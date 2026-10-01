// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s
// RUN: dataflow-scheduler-opt --path-expansion --path-expansion %s | FileCheck %s

// A fifo with a peer relation crosses cores. In the frontend form no stage
// has a unit; the peer fifo is what path expansion plans from. It routes the
// pair of stages through the fifo's units and the switch between them,
// L1 -> MNISU -> RING_STOP -> MNILU -> L1, rather than through DDR, and
// assigns MNISU to the producer stage and MNILU to the consumer stage. The
// stages keep their domains (the producer on core 0, which feeds the
// consumers on cores 0..3), and the fifo and the transfers are unchanged;
// the materializer rebuilds the token chain as usual, which drops the
// trailing token, so the consumer stage depends out on none. A second run finds the mapped pipeline legal and
// leaves it unchanged, so both runs share the same checks.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")

// CHECK-LABEL:   func.func @ring_broadcast(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<1x64xf16, "L1">) {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.peer = #[[$ATTR_0]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[CREATE_TOKEN_0]] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#1) {
// CHECK-NEXT:         ktdf.data_transfer from %[[ARG0]][0, 0] size [1, 64] to %[[PRIVATE_0]]#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #[[$ATTR_1]]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[PRIVATE_0]]#1) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[PRIVATE_0]]#0 size [64] to %[[ARG1]][0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #[[$ATTR_2]]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#peer = affine_map<(c) -> (c floordiv 4)>
#producers = affine_set<(c) : (c == 0)>
#consumers = affine_set<(c) : (c >= 0, 3 - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_broadcast(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {dataflow_scheduler.domain = #consumers}
    }
    return
  }
}
