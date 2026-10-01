// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering -split-input-file -verify-diagnostics %s

// Stage domains that ktdf-to-ktdflowering rejects (see ring-domains.mlir for
// a legal one). The stages are already mapped, as path expansion leaves them.

// A domain is an integer set.
#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @not_a_set(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [32]} {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-error @+1 {{'dataflow_scheduler.domain' must be an integer set}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = 3}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 31 - c >= 0)>}
    }
    return
  }
}

// -----

// A domain is a set over the compute tile alone.
#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @two_dims(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [32]} {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-error @+1 {{domain must be a set over the compute tile alone, with one dimension and no symbols, but it has 2 dimensions and 0 symbols}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = affine_set<(c, d) : (c >= 0, 7 - c >= 0, d >= 0)>}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 31 - c >= 0)>}
    }
    return
  }
}

// -----

// A domain is within the grid: tiles 32..39 do not exist.
#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @outside_grid(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [32]} {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 7 - c >= 0)>}
      // expected-error @+1 {{domain affine_set<(d0) : (d0 >= 0, -d0 + 39 >= 0)> has tiles outside the grid [0, 32)}}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 39 - c >= 0)>}
    }
    return
  }
}

// -----

// A domain is not empty.
#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @empty(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [32]} {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 7 - c >= 0)>}
      // expected-error @+1 {{domain affine_set<(d0) : (d0 - 4 >= 0, -d0 + 2 >= 0)> is empty}}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c - 4 >= 0, 2 - c >= 0)>}
    }
    return
  }
}

// -----

// The producer stage executes on exactly the tiles that feed the consumer
// stage: consumers 0..31 are fed by producers 0..7, not 0..6.
#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @producers_not_peer_image(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [32]} {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-error @+1 {{stage writes a peer fifo on the tiles {0, 1, 2, 3, 4, 5, 6}, but the tiles that feed the stage reading it are {0, 1, 2, 3, 4, 5, 6, 7}: its domain must be the image of the peer relation}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 6 - c >= 0)>}
      // expected-note @+1 {{the stage reading it}}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 31 - c >= 0)>}
    }
    return
  }
}

// -----

// Each kind runs one program unit over its units, so all its stages have the
// same domain. Both pipelines are legal on their own, but their MNISU stages
// execute on cores 0..7 and 0..3.
#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @kind_with_two_domains(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [32]} {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-note @+1 {{another stage of kind "MNISU"}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 7 - c >= 0)>}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 31 - c >= 0)>}
    }
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-error @+1 {{stages of kind "MNISU" have different domains, which is not supported yet}}
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 3 - c >= 0)>}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = affine_set<(c) : (c >= 0, 15 - c >= 0)>}
    }
    return
  }
}
