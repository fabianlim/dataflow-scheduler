// RUN: dataflow-scheduler-opt %s > %t.before.mlir
// RUN: dataflow-scheduler-opt --handle-cross-core-stages %s > %t.after.mlir
// RUN: diff %t.before.mlir %t.after.mlir

// What the pass leaves unchanged on the 4-core ring device: a cross-core
// channel without self-deliveries, whose producer, core 0, is not among its
// consumers, cores 1..3, and a pipeline without a cross-core fifo.

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g - 1 >= 0, 4 * g + 3 - c >= 0)>
#core0 = affine_set<(c) : (c == 0)>
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_without_self_deliveries(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
  func.func @local_copy(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    ktdf.pipeline {
      %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"], dataflow_scheduler.domain = #core0}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %v = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %o = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%v : tensor<64xf16>) outs(%e : tensor<64xf16>) {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %o, %p#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"], dataflow_scheduler.domain = #core0}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["L1SU"], dataflow_scheduler.domain = #core0}
    }
    return
  }
}
