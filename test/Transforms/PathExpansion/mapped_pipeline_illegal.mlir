// RUN: dataflow-scheduler-opt --path-expansion -split-input-file -verify-diagnostics %s

// A pipeline whose stages all have a unit cannot have stages inserted into
// it, so it is an error if two consecutive stages cannot exchange their data
// between their units (see mapped_pipeline_legal.mlir for the rule and the
// legal cases).

// A fifo from MNISU to MNILU without a group domain stays on one core, where
// MNISU and MNILU have no direct link: the ring is a cross-core link, which a
// same-core fifo cannot use. (With a group domain it is the legal
// @ring_fifo.)

module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @same_core_ring_fifo(%src: memref<64x64xf16, "L1">, %dst: memref<64x64xf16, "L1">) {
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
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
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
