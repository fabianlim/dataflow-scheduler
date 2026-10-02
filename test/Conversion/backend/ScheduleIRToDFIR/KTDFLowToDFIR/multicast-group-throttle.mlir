// RUN: dataflow-scheduler-opt --ktdflowering-to-dfir %s | FileCheck %s

// A transfer into and one out of a multicast group with a throttle, on the
// 4-core ring device: producer core 0 sends a 64x64 tile to consumer core 1
// in messages of 64 elements. Each lowers to one composite op whose time
// dimension walks the tile in its messages, as a transfer between memories
// does: 64 time steps, each advancing one row and moving one 64-element
// stick, which the body sends to or receives from the group.

// CHECK-DAG: #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG: #[[$ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$NEXT_ROW:.+]] = affine_map<(d0) -> (d0, 0)>
// CHECK-DAG: #[[$TIME_ORDER:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$STICK:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$ROWS:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL:   func.func @ring_pair_in_messages() attributes {grid = [4]} {
// CHECK-NEXT:     %[[C4096:.*]] = arith.constant 4096 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[MEM0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index

// The producer: one composite load of 64 steps, each sending a stick.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_SU:.*]] -> (%[[SU0]]) : {
// CHECK-NEXT:       %[[SU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU0]] -> %[[MEM0]]]):index
// CHECK-NEXT:       %[[SU_MEM:.*]] = uniform.query_map(map:%[[SU_MEM_MAP]], key:%[[PU_SU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[SU0]]){
// CHECK-NEXT:           %[[GROUP:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           %[[SRC:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_load %[[SRC]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()(%[[LOADED:.*]]:vector<64xf16>)
// CHECK-NEXT:            {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             dataflow.send %[[GROUP]], %[[LOADED]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The consumer: one composite store of 64 steps, each receiving a stick.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_LU:.*]] -> (%[[LU1]]) : {
// CHECK-NEXT:       %[[LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU1]] -> %[[MEM1]]]):index
// CHECK-NEXT:       %[[LU_MEM:.*]] = uniform.query_map(map:%[[LU_MEM_MAP]], key:%[[PU_LU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[LU1]]){
// CHECK-NEXT:           %[[JOIN:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           %[[DST:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C4096]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_store %[[DST]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()
// CHECK-NEXT:            {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             %[[RECEIVED:.*]] = dataflow.receive %[[JOIN]] : vector<64xf16>
// CHECK-NEXT:             agen.yield %[[RECEIVED]] : vector<64xf16>
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(d0) : (d0 == 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_pair_in_messages() attributes {grid = [4]} {
    %su0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
    %lu1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
    %tile = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %su_map = uniform.def_immutable_mapping([%c0 -> %su0]):index
    %su = uniform.query_map(map:%su_map, key:%tile) : index
    %lu_map = uniform.def_immutable_mapping([%c1 -> %lu1]):index
    %lu = uniform.query_map(map:%lu_map, key:%tile) : index
    %src_map = uniform.def_immutable_mapping([%c1 -> %su0]):index
    %src_su = uniform.query_map(map:%src_map, key:%tile) : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c4096, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    ktdf_lowering.execute_on %su, %lu {
      %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      %token = ktdf.create_token : !ktdf.token
      ktdf_lowering.execute_on %su {
        %group = ktdf_lowering.multicast_group %ch producer(%su) {directions = [#dataflow<direction CounterClockwise>], group_ids = array<i32: 0>, num_consumers = array<i32: 1>, producers = array<i64: 0>} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        ktdf.data_transfer from %src[%c0, %c0] size [64, 64] to %group size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      }
      ktdf_lowering.execute_on %lu {
        %group = ktdf_lowering.multicast_group %ch producer(%src_su) {directions = [#dataflow<direction CounterClockwise>], group_ids = array<i32: 0>, num_consumers = array<i32: 1>, producers = array<i64: 0>} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        ktdf.data_transfer from %group size [64, 64] to %dst[%c0, %c0] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
      }
    }
    return
  }
}
