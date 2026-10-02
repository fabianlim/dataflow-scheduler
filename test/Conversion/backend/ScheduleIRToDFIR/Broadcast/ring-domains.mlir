// RUN: dataflow-scheduler-opt --path-expansion --ktdf-to-ktdflowering %s | FileCheck %s
// RUN: dataflow-scheduler-opt --path-expansion --ktdf-to-ktdflowering --debug-only=ktdf-to-operand-lowering %s 2>&1 >/dev/null | FileCheck %s --check-prefix=GROUPS

// The frontend form of the broadcast relayout on the 4-core ring device:
// the cross-core fifo has the groups g = 0..1, and producer core g sends its
// 64x64 tile to the consumer cores 2g and 2g+1, in messages of one 64-element
// stick, the throttle of its transfers. The producer stage's domain selects
// the producer of each group, P(g) = {g}, and the consumer stage's domain
// its consumers, C(g) = {2g, 2g+1}. Path expansion maps the stages to MNISU
// and MNILU and keeps their domains.
//
// ktdf-to-ktdflowering resolves the groups (checked under GROUPS, from the
// debug output): the producer stage executes on cores 0..1, the union of
// P(g). The consumers are cores 0..3, the union of C(g), and consumer c is
// fed by producer c floordiv 2. Core 1 is both the producer of group 1 and a
// consumer of group 0. Core 0 is the one self-delivery, in both C(0) and
// P(0), which the ring cannot carry: the consumer stage executes on the ring
// consumers, C(g) without P(g), cores 1..3. Without handle-cross-core-stages,
// which carries the self-delivery by a local copy, nothing delivers core 0's
// tile to core 0 here.
//
// Units are materialized per kind on the tiles of its stages: MNISU on cores
// 0..1 and MNILU on cores 1..3, and each kind's map from tile to unit lists
// exactly those tiles: after lowering to DFIR, they are the units the kind's
// program unit runs on.
//
// The token between the two stages gets no signal: the producer stage writes
// no memory, only the fifo, so the stages have no memory conflict.
//
// The fifo is carried by multicast groups, which each side makes explicit
// with ktdf_lowering.multicast_group in place of the fifo: the producer names
// its own MNISU, each consumer the MNISU of its group's producer, from the
// table that maps each ring consumer tile 1..3 to it. Both carry the same
// attributes per producer 0..1: the group g, its number of ring consumers (1
// for group 0, whose producer core 0 is also one of its consumers, 2 for
// group 1) and the direction the producer sends in. broadcast-frontend-form.mlir
// lowers the broadcast on to DFIR.

// GROUPS:      Groups of the cross-core fifo affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// GROUPS-NEXT: group 0: producers {0}, consumers {0, 1}
// GROUPS-NEXT: group 1: producers {1}, consumers {2, 3}
// GROUPS-NEXT: producer tiles {0, 1}, consumer tiles {0, 1, 2, 3}, ring consumer tiles {1, 2, 3}
// GROUPS-NEXT: producer of each consumer: 0 <- 0, 1 <- 0, 2 <- 1, 3 <- 1
// GROUPS-NEXT: self-deliveries {0}

// CHECK: #[[$SET:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$GROUP_DOMAIN:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")

// CHECK-LABEL:   func.func @ring_broadcast() attributes {grid = [4]} {
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[TILE:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[SU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU_KEY0]] -> %[[SU0]]], {{\[}}%[[SU_KEY1]] -> %[[SU1]]]):index
// CHECK-NEXT:     %[[SU:.*]] = uniform.query_map(map:%[[SU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[LU_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[LU_KEY2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[LU_KEY3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU_KEY1]] -> %[[LU1]]], {{\[}}%[[LU_KEY2]] -> %[[LU2]]], {{\[}}%[[LU_KEY3]] -> %[[LU3]]]):index
// CHECK-NEXT:     %[[LU:.*]] = uniform.query_map(map:%[[LU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[SRC_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[SRC_KEY2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[SRC_KEY3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[SRC_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SRC_KEY1]] -> %[[SU0]]], {{\[}}%[[SRC_KEY2]] -> %[[SU1]]], {{\[}}%[[SRC_KEY3]] -> %[[SU1]]]):index
// CHECK-NEXT:     %[[SRC_SU:.*]] = uniform.query_map(map:%[[SRC_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[SRC_VIEW:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[SRC:.*]] = memref.memory_space_cast %[[SRC_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// CHECK-NEXT:     %[[DST_VIEW:.*]] = ktdp.construct_memory_view %[[C128]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[DST:.*]] = memref.memory_space_cast %[[DST_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// CHECK-NEXT:     ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// CHECK-NEXT:       %[[CH:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$GROUP_DOMAIN]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:       %[[TOKEN:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       ktdf_lowering.execute_on %[[SU]] {
// CHECK-NEXT:         %[[TO_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, producers = array<i64: 0, 1>} : <"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[SRC]]{{\[}}%[[C0]], %[[C0]]] size [64, 64] to %[[TO_GROUP]] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf_lowering.execute_on %[[LU]] {
// CHECK-NEXT:         %[[FROM_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SRC_SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, producers = array<i64: 0, 1>} : <"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[FROM_GROUP]] size [64, 64] to %[[DST]]{{\[}}%[[C0]], %[[C0]]] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(g) : (g >= 0, 1 - g >= 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 2 * g >= 0, 2 * g + 1 - c >= 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_broadcast() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c128, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[%c0, %c0] size [64, 64] to %p#0 size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64, 64] to %dst[%c0, %c0] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
      } {dataflow_scheduler.domain = #consumers}
    }
    return
  }
}
