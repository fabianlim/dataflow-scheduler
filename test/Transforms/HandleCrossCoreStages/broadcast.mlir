// RUN: dataflow-scheduler-opt --handle-cross-core-stages %s | FileCheck %s

// A cross-core channel on the 4-core ring device, mapped as path expansion
// leaves it: one group, g = 0, whose producer is core 0 and whose consumers
// are cores 0..3. Core 0 delivers to itself, which the ring cannot carry.
//
// The channel pipeline is wrapped, unchanged, as the first stage of an outer
// pipeline. The second stage wraps the local copy of the self-delivery: the
// producer's transfer into L1LU, an identity on the SFU, and the consumer's
// transfer out of L1SU, with the same memrefs, indices and sizes as the
// channel's transfers, the units of the same-core route of the device, and
// the domain core 0. No token joins the two outer stages, so they run
// concurrently. Each wrapper stage carries the units of its stages.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0)[s0] : (d0 - s0 == 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 >= 0, -d0 + s0 * 4 + 3 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")

// CHECK-LABEL:   func.func @ring_broadcast(
// CHECK-SAME:      %[[ARG0:.*]]: memref<64x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<64x64xf16, "L1">) attributes {grid = [4]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             %[[PRIVATE_0:.*]]:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
// CHECK-NEXT:               %[[FIFO_0:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$ATTR_1]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:               %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf.private_yield %[[FIFO_0]], %[[CREATE_TOKEN_0]] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#1) {
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] size [1, 64] to %[[PRIVATE_0]]#0 size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #[[$ATTR_2]]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#1) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_0]]#0 size [64] to %[[ARG1]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #[[$ATTR_3]]}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {applicable_units = ["MNISU", "MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:               %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:               %[[FIFO_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf.private_yield %[[FIFO_1]], %[[FIFO_2]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(%[[PRIVATE_1]]#2) {
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] size [1, 64] to %[[PRIVATE_1]]#0 size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["L1LU"], dataflow_scheduler.domain = #[[$ATTR_1]]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_1]]#2) depends_out(%[[PRIVATE_1]]#3) {
// CHECK-NEXT:               %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_1]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:               %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:               %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:               ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK-NEXT:                 linalg.yield %[[VAL_5]] : f16
// CHECK-NEXT:               } -> tensor<64xf16>
// CHECK-NEXT:               ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_1]]#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["SFU"], dataflow_scheduler.domain = #[[$ATTR_1]]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_1]]#3) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_1]]#1 size [64] to %[[ARG1]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["L1SU"], dataflow_scheduler.domain = #[[$ATTR_1]]}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_broadcast(%src: memref<64x64xf16, "L1">, %dst: memref<64x64xf16, "L1">) attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c64 step %c1 {
      ktdf.pipeline {
        %p:2 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
          %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %ch, %t0 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#1) {
          ktdf.data_transfer from %src[%i, %c0] size [1, 64] to %p#0 size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
        ktdf.stage depends_in(%p#1) depends_out(none) {
          ktdf.data_transfer from %p#0 size [64] to %dst[%i, %c0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
        } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
