// RUN: dataflow-scheduler-opt --handle-cross-core-stages --split-input-file --verify-diagnostics %s

// Two groups, g = 0..1: producer 2g feeds the consumers 2g and 2g + 1, so the
// self-deliveries are cores 0 and 2, which are not one interval.

#groups = affine_set<(g) : (g >= 0, 1 - g >= 0)>
#producers = affine_set<(c)[g] : (c - 2 * g == 0)>
#consumers = affine_set<(c)[g] : (c - 2 * g >= 0, 2 * g + 1 - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @self_deliveries_not_one_interval(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      // expected-error @below {{self-deliveries on tiles that are not one interval are not supported yet}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}

// -----

// A device without a corelet has no same-core route from L1 to L1 through a
// compute unit, so the self-delivery of core 0 cannot be copied locally.

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
module {
  ktdf_arch.device @no_corelet_device {
    %l1 = memory { kind = "L1" }
    %mnisu = exec_unit { kind = "MNISU" }
    %mnilu = exec_unit { kind = "MNILU" }
    datapath %l1 to %mnisu : memory, exec_unit
    datapath %mnilu to %l1 : exec_unit, memory
  }
  func.func @no_local_route(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      // expected-error @below {{expected one same-core route from "L1" to "L1" through a load unit, a compute unit and a store unit for the local copy of the self-deliveries, but found 0}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}

// -----

// Two groups, g = 0..1: producer g feeds only itself, so every consumer is a
// self-delivery and the ring delivers to no tile.

#groups = affine_set<(g) : (g >= 0, 1 - g >= 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - g == 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @no_ring_consumer(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      // expected-error @below {{a cross-core fifo whose consumers are all producers of their group, so that the ring delivers to no tile, is not supported yet}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}

// -----

// Two groups, g = 0..1: producer 0 feeds only itself, and producer 1 feeds
// cores 1..3. Group 0 has no ring consumer, so producer 0 would send over the
// ring to no tile.

#groups = affine_set<(g) : (g >= 0, 1 - g >= 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - g >= 0, 3 * g - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @group_without_ring_consumer(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      // expected-error @below {{group 0 of the cross-core fifo delivers only to its producer, so that the producer sends over the ring to no tile, which is not supported yet}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}

// -----

// The local copy of a channel moves it in the messages of the channel's
// throttle, so the transfers into and out of the channel must agree on it.

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @throttles_differ(%src: memref<64x64xf16, "L1">, %dst: memref<64x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        // expected-error @below {{the transfers into and out of a cross-core fifo with self-deliveries must have the same throttle}}
        ktdf.data_transfer from %src[0, 0] size [64, 64] to %p#0 size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64, 64] to %dst[0, 0] size [64, 64] {dataflow_scheduler.throttle = 32 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}

// -----

// A message of the throttle's 64 elements does not divide the innermost
// dimension of 96 elements.

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @innermost_not_multiple_of_throttle(%src: memref<64x96xf16, "L1">, %dst: memref<64x96xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        // expected-error @below {{the local copy of a cross-core fifo moves it in messages of its throttle of 64, which requires the innermost sizes of the memories it copies between to be multiples of the throttle}}
        ktdf.data_transfer from %src[0, 0] size [64, 96] to %p#0 size [64, 96] {dataflow_scheduler.throttle = 64 : i64} : memref<64x96xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64, 96] to %dst[0, 0] size [64, 96] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x96xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}

// -----

// The producer reads 64 rows of one message and the consumer writes 32 rows of
// two, so the local copy has no one loop over the messages of both.

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @different_walks(%src: memref<64x64xf16, "L1">, %dst: memref<32x128xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        // expected-error @below {{the local copy of a cross-core fifo moves it in messages of its throttle of 64, which requires the memories it copies between to be walked in the same messages}}
        ktdf.data_transfer from %src[0, 0] size [64, 64] to %p#0 size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [32, 128] to %dst[0, 0] size [32, 128] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<32x128xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}
