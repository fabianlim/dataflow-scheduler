// RUN: dataflow-scheduler-opt --ktdflowering-to-dfir %s | FileCheck %s

// A cross-core fifo with two groups on the 4-core ring device, as
// ktdf-to-ktdflowering emits it: producer core 0 sends to consumer core 1
// counter-clockwise, producer core 2 to consumer core 3 clockwise. Each side
// creates its end of the group with ktdf_lowering.multicast_group: the
// producer names its own MNISU, the consumer the MNISU that feeds its core,
// through the table %src_map.
//
// The group's id, number of consumers and direction differ per unit, so the
// MNISU and the MNILU program units get one uniformize_regions region per
// unit, and each region creates the group of its producer's entry: the
// producer resolves its own unit, the consumer the entry of %src_map for its
// core. The transfer into the group becomes a one-step agen.composite_load
// whose body sends with the producer's direction, the transfer out of it a
// one-step agen.composite_store whose body receives from the group.

// CHECK-DAG: #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG: #[[$ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$NO_OFFSET:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK-DAG: #[[$TIME_ORDER:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$STICK:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$ONE_STEP:.+]] = affine_set<(d0) : (d0 == 0)>

// CHECK-LABEL:   func.func @ring_pairs() attributes {grid = [4]} {
// CHECK-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[MEM0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM2:.*]] = dataflow.get_unit {core = 2 : i32, name = "C2-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM3:.*]] = dataflow.get_unit {core = 3 : i32, name = "C3-l1", type = "l1"} : index

// The producers: each region creates its own group and sends to it.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_SU:.*]] -> (%[[SU0]], %[[SU2]]) : {
// CHECK-NEXT:       %[[SU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU0]] -> %[[MEM0]]], {{\[}}%[[SU2]] -> %[[MEM2]]]):index
// CHECK-NEXT:       %[[SU_MEM:.*]] = uniform.query_map(map:%[[SU_MEM_MAP]], key:%[[PU_SU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[SU0]]){
// CHECK-NEXT:           %[[SRC0:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:           %[[GROUP0:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           agen.composite_load %[[SRC0]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()(%[[LOADED0:.*]]:vector<64xf16>)
// CHECK-NEXT:            {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             dataflow.send %[[GROUP0]], %[[LOADED0]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<1x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU2]]){
// CHECK-NEXT:           %[[SRC2:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:           %[[GROUP2:.*]] = dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           agen.composite_load %[[SRC2]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()(%[[LOADED2:.*]]:vector<64xf16>)
// CHECK-NEXT:            {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             dataflow.send %[[GROUP2]], %[[LOADED2]] {dir = #dataflow<direction Clockwise>} : vector<64xf16>
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<1x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The consumers: each region joins the group of the producer feeding its core
// and receives from it.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_LU:.*]] -> (%[[LU1]], %[[LU3]]) : {
// CHECK-NEXT:       %[[LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU1]] -> %[[MEM1]]], {{\[}}%[[LU3]] -> %[[MEM3]]]):index
// CHECK-NEXT:       %[[LU_MEM:.*]] = uniform.query_map(map:%[[LU_MEM_MAP]], key:%[[PU_LU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[LU1]]){
// CHECK-NEXT:           %[[DST1:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:           %[[JOIN0:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           agen.composite_store %[[DST1]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()
// CHECK-NEXT:            {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             %[[RECEIVED1:.*]] = dataflow.receive %[[JOIN0]] : vector<64xf16>
// CHECK-NEXT:             agen.yield %[[RECEIVED1]] : vector<64xf16>
// CHECK-NEXT:           } : memref<1x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU3]]){
// CHECK-NEXT:           %[[DST3:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:           %[[JOIN2:.*]] = dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           agen.composite_store %[[DST3]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()
// CHECK-NEXT:            {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             %[[RECEIVED3:.*]] = dataflow.receive %[[JOIN2]] : vector<64xf16>
// CHECK-NEXT:             agen.yield %[[RECEIVED3]] : vector<64xf16>
// CHECK-NEXT:           } : memref<1x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
#set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_pairs() attributes {grid = [4]} {
    %su0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
    %su2 = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnisu", type = "mnisu"} : index
    %lu1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
    %lu3 = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
    %tile = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %su_map = uniform.def_immutable_mapping([%c0 -> %su0], [%c2 -> %su2]):index
    %su = uniform.query_map(map:%su_map, key:%tile) : index
    %lu_map = uniform.def_immutable_mapping([%c1 -> %lu1], [%c3 -> %lu3]):index
    %lu = uniform.query_map(map:%lu_map, key:%tile) : index
    %src_map = uniform.def_immutable_mapping([%c1 -> %su0], [%c3 -> %su2]):index
    %src_su = uniform.query_map(map:%src_map, key:%tile) : index
    %0 = ktdp.construct_memory_view %c0, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<1x64xf16, #ktdp.memory_space<ct_local>> to memref<1x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c128, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<1x64xf16, #ktdp.memory_space<ct_local>> to memref<1x64xf16, "L1">
    ktdf_lowering.execute_on %su, %lu {
      %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      %token = ktdf.create_token : !ktdf.token
      ktdf_lowering.execute_on %su {
        %group = ktdf_lowering.multicast_group %ch producer(%su) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 1>, producers = array<i64: 0, 2>} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        ktdf.data_transfer from %src[%c0, %c0] size [1, 64] to %group size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      }
      ktdf_lowering.execute_on %lu {
        %group = ktdf_lowering.multicast_group %ch producer(%src_su) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 1>, producers = array<i64: 0, 2>} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        ktdf.data_transfer from %group size [64] to %dst[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<1x64xf16, "L1">
      }
    }
    return
  }
}
