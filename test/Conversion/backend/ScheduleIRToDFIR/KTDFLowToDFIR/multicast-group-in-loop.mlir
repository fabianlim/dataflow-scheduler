// RUN: dataflow-scheduler-opt --ktdflowering-to-dfir %s | FileCheck %s

// Transfers into and out of a multicast group inside an outer loop, on the
// 4-core ring device: producer core 0 sends a 4x128 tile to consumer core 1,
// one row per iteration, as two sticks. The group's producer and attributes
// are the same throughout a unit's region, so each region creates the group
// once, at its top, before the loop, and both transfers of each iteration
// send to it or receive from it.

// CHECK-DAG: #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 128 + d1)>
// CHECK-DAG: #[[$ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$NO_OFFSET:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK-DAG: #[[$TIME_ORDER:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$STICK:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$ONE_STEP:.+]] = affine_set<(d0) : (d0 == 0)>

// CHECK-LABEL:   func.func @ring_pair_in_loop() attributes {grid = [4]} {
// CHECK-NEXT:     %[[C512:.*]] = arith.constant 512 : index
// CHECK-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[MEM0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index

// The producer creates the group before the loop and sends both sticks of
// each row to it.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_SU:.*]] -> (%[[SU0]]) : {
// CHECK-NEXT:       %[[SU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU0]] -> %[[MEM0]]]):index
// CHECK-NEXT:       %[[SU_MEM:.*]] = uniform.query_map(map:%[[SU_MEM_MAP]], key:%[[PU_SU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[SU0]]){
// CHECK-NEXT:           %[[GROUP:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           %[[SRC:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<4x128xf16>
// CHECK-NEXT:           scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-NEXT:             agen.composite_load %[[SRC]]{{\[}}%[[I]], %[[C0]]]
// CHECK-NEXT:              time_symbols()(%[[FIRST:.*]]:vector<64xf16>)
// CHECK-NEXT:              {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               dataflow.send %[[GROUP]], %[[FIRST]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:               agen.yield
// CHECK-NEXT:             } : memref<4x128xf16>
// CHECK-NEXT:             agen.composite_load %[[SRC]]{{\[}}%[[I]], %[[C64]]]
// CHECK-NEXT:              time_symbols()(%[[SECOND:.*]]:vector<64xf16>)
// CHECK-NEXT:              {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               dataflow.send %[[GROUP]], %[[SECOND]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:               agen.yield
// CHECK-NEXT:             } : memref<4x128xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The consumer joins the group before the loop and receives both sticks of
// each row from it.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_LU:.*]] -> (%[[LU1]]) : {
// CHECK-NEXT:       %[[LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU1]] -> %[[MEM1]]]):index
// CHECK-NEXT:       %[[LU_MEM:.*]] = uniform.query_map(map:%[[LU_MEM_MAP]], key:%[[PU_LU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[LU1]]){
// CHECK-NEXT:           %[[JOIN:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           %[[DST:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C512]] {layout_map = #[[$LAYOUT]]} : index, index, memref<4x128xf16>
// CHECK-NEXT:           scf.for %[[J:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-NEXT:             agen.composite_store %[[DST]]{{\[}}%[[J]], %[[C0]]]
// CHECK-NEXT:              time_symbols()
// CHECK-NEXT:              {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               %[[RECEIVED_FIRST:.*]] = dataflow.receive %[[JOIN]] : vector<64xf16>
// CHECK-NEXT:               agen.yield %[[RECEIVED_FIRST]] : vector<64xf16>
// CHECK-NEXT:             } : memref<4x128xf16>
// CHECK-NEXT:             agen.composite_store %[[DST]]{{\[}}%[[J]], %[[C64]]]
// CHECK-NEXT:              time_symbols()
// CHECK-NEXT:              {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               %[[RECEIVED_SECOND:.*]] = dataflow.receive %[[JOIN]] : vector<64xf16>
// CHECK-NEXT:               agen.yield %[[RECEIVED_SECOND]] : vector<64xf16>
// CHECK-NEXT:             } : memref<4x128xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(d0) : (d0 == 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
  ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_pair_in_loop() attributes {grid = [4]} {
    %su0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
    %lu1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
    %tile = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %su_map = uniform.def_immutable_mapping([%c0 -> %su0]):index
    %su = uniform.query_map(map:%su_map, key:%tile) : index
    %lu_map = uniform.def_immutable_mapping([%c1 -> %lu1]):index
    %lu = uniform.query_map(map:%lu_map, key:%tile) : index
    %src_map = uniform.def_immutable_mapping([%c1 -> %su0]):index
    %src_su = uniform.query_map(map:%src_map, key:%tile) : index
    %0 = ktdp.construct_memory_view %c0, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<4x128xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<4x128xf16, #ktdp.memory_space<ct_local>> to memref<4x128xf16, "L1">
    %1 = ktdp.construct_memory_view %c512, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<4x128xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<4x128xf16, #ktdp.memory_space<ct_local>> to memref<4x128xf16, "L1">
    scf.for %i = %c0 to %c4 step %c1 {
      ktdf_lowering.execute_on %su, %lu {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %token = ktdf.create_token : !ktdf.token
        ktdf_lowering.execute_on %su {
          %group = ktdf_lowering.multicast_group %ch producer(%su) {directions = [#dataflow<direction CounterClockwise>], group_ids = array<i32: 0>, num_consumers = array<i32: 1>, producers = array<i64: 0>} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
          ktdf.data_transfer from %src[%i, %c0] size [1, 64] to %group size [64] : memref<4x128xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
          ktdf.data_transfer from %src[%i, %c64] size [1, 64] to %group size [64] : memref<4x128xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        }
        ktdf_lowering.execute_on %lu {
          %group = ktdf_lowering.multicast_group %ch producer(%src_su) {directions = [#dataflow<direction CounterClockwise>], group_ids = array<i32: 0>, num_consumers = array<i32: 1>, producers = array<i64: 0>} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
          ktdf.data_transfer from %group size [64] to %dst[%i, %c0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<4x128xf16, "L1">
          ktdf.data_transfer from %group size [64] to %dst[%i, %c64] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<4x128xf16, "L1">
        }
      }
    }
    return
  }
}
