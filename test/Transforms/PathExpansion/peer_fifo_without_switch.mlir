// RUN: dataflow-scheduler-opt --path-expansion -verify-diagnostics %s

// The ring_peer_fifo pipeline on a device without a switch: MNISU reaches
// MNILU only through DDR, a memory, which a fifo cannot pass through, so
// there is no route across cores and planning fails.

#peer = affine_map<(c) -> (c floordiv 4)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @no_switch(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) {
    // expected-error @+1 {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      }
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      }
    }
    return
  }
}
