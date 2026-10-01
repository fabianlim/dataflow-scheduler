// RUN: dataflow-scheduler-opt --path-expansion -split-input-file -verify-diagnostics %s | FileCheck %s

// A pipeline whose stages all have a unit has nothing left for path expansion
// to assign. It is legal if the data two consecutive stages exchange can
// travel between their units: a fifo directly or through switches only, a
// memory buffer through that memory. A legal pipeline comes out unchanged; an
// illegal one is an error, because stages cannot be inserted into it.

// A fifo from MNISU to MNILU over the ring: the units are joined through the
// RING_STOP switch, so the pipeline is legal.

// CHECK-LABEL: func.func @ring_fifo
// CHECK:         ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token)
// CHECK:         ktdf.stage depends_in(none)
// CHECK-NEXT:      ktdf.data_transfer {{.*}} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK-NEXT:      ktdf.data_transfer {{.*}} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:  }
module {
  ktdf_arch.device @ring32_device import("../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @ring_fifo(%src: memref<64x64xf16, "L1">, %dst: memref<64x64xf16, "L1">) {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [64, 64] to %p#0 size [4096] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
      } {applicable_units = ["MNISU"]}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [4096] to %dst[0, 0] size [64, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, memref<64x64xf16, "L1">
      } {applicable_units = ["MNILU"]}
    }
    return
  }
}

// -----

// The same fifo on a device without a ring: MNISU reaches MNILU only through
// DDR, a memory, which a fifo cannot pass through.

module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @fifo_through_memory(%src: memref<64x64xf16, "L1">, %dst: memref<64x64xf16, "L1">) {
    // expected-error @+2 {{path-expansion: two consecutive stages of this fully mapped pipeline are not connected, and stages cannot be inserted into a fully mapped pipeline}}
    // expected-error @+1 {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [64, 64] to %p#0 size [4096] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
      } {applicable_units = ["MNISU"]}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [4096] to %dst[0, 0] size [64, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, memref<64x64xf16, "L1">
      } {applicable_units = ["MNILU"]}
    }
    return
  }
}

// -----

// A fifo from L1LU to L1SU: the units are joined only through the SFU, so a
// stage is missing between them.

module {
  ktdf_arch.device @ring32_device import("../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @missing_compute(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) {
    // expected-error @+2 {{path-expansion: two consecutive stages of this fully mapped pipeline are not connected, and stages cannot be inserted into a fully mapped pipeline}}
    // expected-error @+1 {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
        %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}

// -----

// The local identity chain with every stage mapped: each fifo is a direct
// link, so the pipeline is legal.

// CHECK-LABEL: func.func @local_chain
// CHECK:         ktdf.stage depends_in(none)
// CHECK:         } {applicable_units = ["L1LU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK:         } {applicable_units = ["L1SU"]}
// CHECK-NEXT:  }
#id = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @ring32_device import("../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @local_chain(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) {
    ktdf.pipeline {
      %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1, %t2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %v = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %o = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%v : tensor<64xf16>) outs(%e : tensor<64xf16>) {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %o, %p#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        ktdf.data_transfer from %p#1 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
