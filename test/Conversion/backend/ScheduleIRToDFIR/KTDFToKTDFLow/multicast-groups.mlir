// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering %s | FileCheck %s

// A cross-core fifo with two groups on the 4-core ring device: producer core
// 2g sends to consumer core 2g + 1, for g = 0, 1. ktdf-to-ktdflowering makes
// each side's end of the multicast group explicit with
// ktdf_lowering.multicast_group, which the stage's transfer uses in place of
// the fifo. The producer names its own MNISU; the consumer names the MNISU
// that feeds its core, from the table %src_map, which maps each ring consumer
// tile to the MNISU on its group's producer tile. Both carry the same
// per-producer attributes: producers 0 and 2, group ids 0 and 1, one ring
// consumer each, and the directions of producers 0 and 2.

// CHECK-DAG: #[[$GROUP_DOMAIN:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK-LABEL:   func.func @ring_pairs(
// CHECK-SAME:      %[[SRC:.*]]: memref<1x64xf16, "L1">, %[[DST:.*]]: memref<1x64xf16, "L1">) attributes {grid = [4]} {
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[TILE:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[SU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU_KEY2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU_KEY0]] -> %[[SU0]]], {{\[}}%[[SU_KEY2]] -> %[[SU2]]]):index
// CHECK-NEXT:     %[[SU:.*]] = uniform.query_map(map:%[[SU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[LU_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[LU_KEY3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU_KEY1]] -> %[[LU1]]], {{\[}}%[[LU_KEY3]] -> %[[LU3]]]):index
// CHECK-NEXT:     %[[LU:.*]] = uniform.query_map(map:%[[LU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[SRC_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[SRC_KEY3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[SRC_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SRC_KEY1]] -> %[[SU0]]], {{\[}}%[[SRC_KEY3]] -> %[[SU2]]]):index
// CHECK-NEXT:     %[[SRC_SU:.*]] = uniform.query_map(map:%[[SRC_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// CHECK-NEXT:       %[[CH:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$GROUP_DOMAIN]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:       %{{.*}} = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       %{{.*}} = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       ktdf_lowering.execute_on %[[SU]] {
// CHECK-NEXT:         %[[TO_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 1>, producers = array<i64: 0, 2>} : <"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[SRC]]{{\[}}%[[C0]], %[[C0]]] size [1, 64] to %[[TO_GROUP]] size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf_lowering.execute_on %[[LU]] {
// CHECK-NEXT:         %[[FROM_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SRC_SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 1>, producers = array<i64: 0, 2>} : <"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[FROM_GROUP]] size [64] to %[[DST]]{{\[}}%[[C0]], %[[C0]]] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(g) : (g >= 0, 1 - g >= 0)>
#producers = affine_set<(c)[g] : (c - 2 * g == 0)>
#consumers = affine_set<(c)[g] : (c - 2 * g - 1 == 0)>
module {
  ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_pairs(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[%c0, %c0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64] to %dst[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }
}
