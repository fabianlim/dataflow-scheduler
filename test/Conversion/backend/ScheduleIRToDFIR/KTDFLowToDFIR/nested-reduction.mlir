// RUN: dataflow-scheduler-opt --ktdflowering-to-dfir -allow-unregistered-dialect %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = {kind = "HBM", size = 1073741824 : i64}
// CHECK: #[[$ATTR_1:.+]] = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
// CHECK: #[[$ATTR_2:.+]] = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}
// CHECK: #[[$ATTR_3:.+]] = {kind = "LX", size = 1048576 : i64}
// CHECK: #[[$ATTR_4:.+]] = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
// CHECK: #[[$ATTR_5:.+]] = {kind = "LXSU", load_store}
// CHECK: #[[$ATTR_6:.+]] = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
// CHECK: #[[$ATTR_7:.+]] = {kind = "core"}
// CHECK: #[[$ATTR_8:.+]] = {kind = "lrfreg", size = 65536 : i64}
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 64 + d2)>
// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 64 + d2 * 64 + d3 * 64 + d4)>
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0) -> (d0, 0, 0)>
// CHECK: #[[$ATTR_13:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[$ATTR_14:.+]] = affine_map<(d0) -> (d0, 0, 0, 0, 0)>
// CHECK: #[[$ATTR_15:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_16:.+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 64 + d2)>
// CHECK: #[[$ATTR_17:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK: #[[$ATTR_18:.+]] = affine_map<(d0) -> (0, 0, 0)>
// CHECK: #[[$ATTR_19:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_20:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ATTR_21:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_22:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK: #[[$ATTR_23:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK: #[[$ATTR_24:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK: #[[$ATTR_25:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_26:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK-LABEL:   ktdf_arch.device @spyre_1core attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.spyre_memory_space<HBM> = "HBM", #ktdp.spyre_memory_space<LX> = "LX", #ktdp.spyre_memory_space<LRF> = "lrfreg">} {
// CHECK-NEXT:     %[[VAL_0:.*]] = memory #[[$ATTR_0]]
// CHECK-NEXT:     group #[[$ATTR_7]] share(%[[VAL_0]]) {
// CHECK-NEXT:       %[[VAL_1:.*]] = memory #[[$ATTR_3]]
// CHECK-NEXT:       %[[VAL_2:.*]] = memory #[[$ATTR_8]]
// CHECK-NEXT:       %[[VAL_3:.*]] = exec_unit #[[$ATTR_1]]
// CHECK-NEXT:       %[[VAL_4:.*]] = exec_unit #[[$ATTR_2]]
// CHECK-NEXT:       datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_0]] to %[[VAL_3]] : memory, exec_unit
// CHECK-NEXT:       datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_3]] to %[[VAL_1]] : exec_unit, memory
// CHECK-NEXT:       datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_1]] to %[[VAL_4]] : memory, exec_unit
// CHECK-NEXT:       datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_4]] to %[[VAL_0]] : exec_unit, memory
// CHECK-NEXT:       group share(%[[VAL_1]]) {
// CHECK-NEXT:         %[[VAL_5:.*]] = exec_unit #[[$ATTR_6]]
// CHECK-NEXT:         %[[VAL_6:.*]] = exec_unit #[[$ATTR_4]]
// CHECK-NEXT:         %[[VAL_7:.*]] = exec_unit #[[$ATTR_5]]
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_1]] to %[[VAL_6]] : memory, exec_unit
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_6]] to %[[VAL_5]] : exec_unit, exec_unit
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_5]] to %[[VAL_7]] : exec_unit, exec_unit
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_7]] to %[[VAL_1]] : exec_unit, memory
// CHECK-NEXT:       }
// CHECK-NEXT:       group share(%[[VAL_1]]) {
// CHECK-NEXT:         %[[VAL_8:.*]] = exec_unit #[[$ATTR_6]]
// CHECK-NEXT:         %[[VAL_9:.*]] = exec_unit #[[$ATTR_4]]
// CHECK-NEXT:         %[[VAL_10:.*]] = exec_unit #[[$ATTR_5]]
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_1]] to %[[VAL_9]] : memory, exec_unit
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_9]] to %[[VAL_8]] : exec_unit, exec_unit
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_8]] to %[[VAL_10]] : exec_unit, exec_unit
// CHECK-NEXT:         datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_10]] to %[[VAL_1]] : exec_unit, memory
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func private @"local-schedule-0"() attributes {grid = [1]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 16384 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3su", type = "l3su"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
// CHECK-NEXT:     %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
// CHECK-NEXT:     %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
// CHECK-NEXT:     %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
// CHECK-NEXT:     %[[GET_UNIT_8:.*]] = dataflow.get_unit {name = "hbm", type = "hbm"} : index
// CHECK-NEXT:     %[[GET_UNIT_9:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_8]], %[[CONSTANT_6]] {layout_map = #[[$ATTR_9]]} : index, index, memref<2x256x64xf16>
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_4]] {layout_map = #[[$ATTR_10]]} : index, index, memref<2x256x1x1x64xf16>
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_6]], %[[VAL_1]], 0] dst:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_6]], %[[VAL_1]], 0, 0, 0]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_2:.*]]:vector<64xf16>)
// CHECK-NEXT:          {load_order = #[[$ATTR_11]], load_set = #[[$ATTR_21]], load_time_addr_map = #[[$ATTR_12]], store_order = #[[$ATTR_13]], store_set = #[[$ATTR_22]], store_time_addr_map = #[[$ATTR_14]], time_order = #[[$ATTR_15]], time_set = #[[$ATTR_23]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<2x256x64xf16>, memref<2x256x1x1x64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_1]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_2]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_1]] : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_2]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_3:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_9]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_3]]) : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[CONSTANT_6]]], {{\[}}%[[GET_UNIT_3]] -> %[[CONSTANT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[VAL_3]]) : index
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[QUERY_MAP_4]], %[[CONSTANT_4]] : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_2:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_3]], %[[ADDI_0]] {layout_map = #[[$ATTR_10]]} : index, index, memref<2x256x1x1x64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_0]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[VAL_3]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_5]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_5]] : index
// CHECK-NEXT:       scf.for %[[VAL_4:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[GET_LOGICAL_MEMORY_VIEW_2]]{{\[}}%[[CONSTANT_6]], %[[VAL_4]], 0, 0, 0] {load_order = #[[$ATTR_13]], load_set = #[[$ATTR_22]]} : memref<2x256x1x1x64xf16>, vector<64xf16>
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[VAL_3]]) : index
// CHECK-NEXT:         dataflow.send %[[QUERY_MAP_6]], %[[VECTOR_LOAD_0]] : vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_5:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:       %[[GET_LOCAL_UNIT_0:.*]] = dataflow.get_local_unit %[[VAL_5]] {name = "lrfreg"} : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_3:.*]] = dataflow.get_logical_memory_view %[[GET_LOCAL_UNIT_0]], %[[CONSTANT_6]] {layout_map = #[[$ATTR_15]]} : index, index, memref<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[VAL_5]]) : index
// CHECK-NEXT:       %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_7]] : vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_0]], %[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[CONSTANT_6]]] {store_order = #[[$ATTR_15]], store_set = #[[$ATTR_24]]} : memref<64xf16>, vector<64xf16>
// CHECK-NEXT:       scf.for %[[VAL_6:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_8:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_8:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_8]], key:%[[VAL_5]]) : index
// CHECK-NEXT:         %[[RECEIVE_1:.*]] = dataflow.receive %[[QUERY_MAP_8]] : vector<64xf16>
// CHECK-NEXT:         %[[VECTOR_LOAD_1:.*]] = agen.vector_load %[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[CONSTANT_6]]] {load_order = #[[$ATTR_15]], load_set = #[[$ATTR_24]]} : memref<64xf16>, vector<64xf16>
// CHECK-NEXT:         %[[BINARY_0:.*]] = vectorchain.binary %[[RECEIVE_1]], %[[VECTOR_LOAD_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_15]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:         agen.vector_store %[[BINARY_0]], %[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[CONSTANT_6]]] {store_order = #[[$ATTR_15]], store_set = #[[$ATTR_24]]} : memref<64xf16>, vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[VECTOR_LOAD_2:.*]] = agen.vector_load %[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[CONSTANT_6]]] {load_order = #[[$ATTR_15]], load_set = #[[$ATTR_24]]} : memref<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_9:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_9:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_9]], key:%[[VAL_5]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_9]], %[[VECTOR_LOAD_2]] : vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_7:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_10:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_9]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_10:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_10]], key:%[[VAL_7]]) : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_11:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[CONSTANT_6]]], {{\[}}%[[GET_UNIT_7]] -> %[[CONSTANT_0]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_11:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_11]], key:%[[VAL_7]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_4:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_10]], %[[QUERY_MAP_11]] {layout_map = #[[$ATTR_16]]} : index, index, memref<2x1x64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_12:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_12:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_12]], key:%[[VAL_7]]) : index
// CHECK-NEXT:       %[[RECEIVE_2:.*]] = dataflow.receive %[[QUERY_MAP_12]] : vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_2]], %[[GET_LOGICAL_MEMORY_VIEW_4]]{{\[}}%[[CONSTANT_6]], 0, 0] {store_order = #[[$ATTR_11]], store_set = #[[$ATTR_21]]} : memref<2x1x64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_13:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_1]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_13:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_13]], key:%[[VAL_7]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_13]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_13]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_8:.*]] -> (%[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_14:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_14:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_14]], key:%[[VAL_8]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_5:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_8]], %[[CONSTANT_5]] {layout_map = #[[$ATTR_17]]} : index, index, memref<2x64xf16>
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_6:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_14]], %[[CONSTANT_6]] {layout_map = #[[$ATTR_16]]} : index, index, memref<2x1x64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_15:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_6]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_15:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_15]], key:%[[VAL_8]]) : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_16:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_16:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_16]], key:%[[VAL_8]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_15]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_16]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_15]] : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_16]] : index
// CHECK-NEXT:       scf.for %[[VAL_9:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_6]]{{\[}}%[[VAL_9]], 0, 0] dst:%[[GET_LOGICAL_MEMORY_VIEW_5]]{{\[}}%[[VAL_9]], 0]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_10:.*]]:vector<64xf16>)
// CHECK-NEXT:          {load_order = #[[$ATTR_11]], load_set = #[[$ATTR_21]], load_time_addr_map = #[[$ATTR_18]], store_order = #[[$ATTR_19]], store_set = #[[$ATTR_25]], store_time_addr_map = #[[$ATTR_20]], time_order = #[[$ATTR_15]], time_set = #[[$ATTR_26]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<2x1x64xf16>, memref<2x64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func private @"local-schedule-0_keep_alive"() {
// CHECK-NEXT:     call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// Pass-20 KTDFLowToDFIR on the nested reduction: the reduction loop is PEELED (first iteration
// hoisted, loop runs from 1), the peeled reset is an UNCONDITIONAL all-lanes vector_store, and
// the in-loop RMW writeback (vector_load -> vectorchain add -> vector_store) is structural.
#HBM = {kind = "HBM", size = 1073741824 : i64}
#L3LU = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
#L3SU = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}
#LX = {kind = "LX", size = 1048576 : i64}
#LXLU = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
#LXSU = {kind = "LXSU", load_store}
#SFP = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
#core = {kind = "core"}
#lrfreg = {kind = "lrfreg", size = 65536 : i64}
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @spyre_1core attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.spyre_memory_space<HBM> = "HBM", #ktdp.spyre_memory_space<LX> = "LX", #ktdp.spyre_memory_space<LRF> = "lrfreg">} {
    %memory = memory #HBM
    group #core share(%memory) {
      %memory_0 = memory #LX
      %memory_1 = memory #lrfreg
      %exec_unit = exec_unit #L3LU
      %exec_unit_2 = exec_unit #L3SU
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory to %exec_unit : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit to %memory_0 : exec_unit, memory
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory_0 to %exec_unit_2 : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit_2 to %memory : exec_unit, memory
      group share(%memory_0) {
        %exec_unit_3 = exec_unit #SFP
        %exec_unit_4 = exec_unit #LXLU
        %exec_unit_5 = exec_unit #LXSU
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %memory_0 to %exec_unit_4 : memory, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_4 to %exec_unit_3 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_3 to %exec_unit_5 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %exec_unit_5 to %memory_0 : exec_unit, memory
      }
      group share(%memory_0) {
        %exec_unit_3 = exec_unit #SFP
        %exec_unit_4 = exec_unit #LXLU
        %exec_unit_5 = exec_unit #LXSU
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %memory_0 to %exec_unit_4 : memory, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_4 to %exec_unit_3 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_3 to %exec_unit_5 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %exec_unit_5 to %memory_0 : exec_unit, memory
      }
    }
  }
  func.func private @"local-schedule-0"() attributes {grid = [1]} {
    %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
    %1 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3su", type = "l3su"} : index
    %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
    %3 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
    %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
    %5 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
    %6 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
    %7 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
    %8 = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %9 = uniform.def_immutable_mapping([%c0 -> %0]):index
    %10 = uniform.query_map(map:%9, key:%8) : index
    %c0_0 = arith.constant 0 : index
    %11 = uniform.def_immutable_mapping([%c0_0 -> %1]):index
    %12 = uniform.query_map(map:%11, key:%8) : index
    %c0_1 = arith.constant 0 : index
    %13 = uniform.def_immutable_mapping([%c0_1 -> %2]):index
    %14 = uniform.query_map(map:%13, key:%8) : index
    %c0_2 = arith.constant 0 : index
    %15 = uniform.def_immutable_mapping([%c0_2 -> %3]):index
    %16 = uniform.query_map(map:%15, key:%8) : index
    %c0_3 = arith.constant 0 : index
    %17 = uniform.def_immutable_mapping([%c0_3 -> %4]):index
    %18 = uniform.query_map(map:%17, key:%8) : index
    %c0_4 = arith.constant 0 : index
    %19 = uniform.def_immutable_mapping([%c0_4 -> %5]):index
    %20 = uniform.query_map(map:%19, key:%8) : index
    %c0_5 = arith.constant 0 : index
    %21 = uniform.def_immutable_mapping([%c0_5 -> %6]):index
    %22 = uniform.query_map(map:%21, key:%8) : index
    %c0_6 = arith.constant 0 : index
    %23 = uniform.def_immutable_mapping([%c0_6 -> %7]):index
    %24 = uniform.query_map(map:%23, key:%8) : index
    %c0_7 = arith.constant 0 : index
    %c8589934592 = arith.constant 8589934592 : index
    %c256 = arith.constant 256 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %25 = ktdp.construct_memory_view %c0_7, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %memspacecast = memref.memory_space_cast %25 : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
    %26 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
    %memspacecast_8 = memref.memory_space_cast %26 : memref<2x64xf16> to memref<2x64xf16, "HBM">
    %reinterpret_cast_9 = memref.reinterpret_cast %memspacecast_8 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1]>, "HBM">
    ktdf_lowering.execute_on %10, %14, %16, %18, %20, %22, %24, %12 {
      %c0_10 = arith.constant 0 : index
      %27 = builtin.unrealized_conversion_cast %c0_10 : index to memref<1x64xf16, "lrfreg">
      %c0_11 = arith.constant 0 : index
      %28 = builtin.unrealized_conversion_cast %c0_11 : index to memref<2x1x64xf16, "LX">
      %29 = ktdf.create_token : !ktdf.token
      ktdf_lowering.execute_on %10, %14, %16, %18, %20, %22, %24 {
        ktdf.parallel (%arg0, %arg1) = (%c0_7) to (%c2) step (%c1) distribute(num_instances = 2) {
          ktdf_lowering.execute_on %18, %20, %22, %24 {
            %30 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
            %31 = ktdf.create_token : !ktdf.token
            ktdf_lowering.execute_on %18, %20 {
              ktdf_lowering.execute_on %10, %14, %16, %18, %20 {
                %c256_12 = arith.constant 256 : index
                %33 = builtin.unrealized_conversion_cast %c256_12 : index to memref<2x256x1x1x64xf16, "LX">
                %34 = ktdf.create_token : !ktdf.token
                ktdf_lowering.execute_on %10 {
                  scf.for %arg2 = %c0_7 to %c256 step %c1 {
                    ktdf.data_transfer from %reinterpret_cast[%arg0, %arg2, 0] size [1, 1, 64] to %33[%arg0, %arg2, 0, 0, 0] size [1, 1, 1, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, memref<2x256x1x1x64xf16, "LX">
                  }
                }
                ktdf_lowering.signal %10, %14, %16
                ktdf_lowering.execute_on %14, %16, %18, %20 {
                  scf.for %arg2 = %c0_7 to %c256 step %c1 {
                    ktdf_lowering.execute_on %14, %16, %18, %20 {
                      %35 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
                      %36 = ktdf.create_token : !ktdf.token
                      ktdf_lowering.execute_on %14, %16 {
                        ktdf.data_transfer from %33[%arg0, %arg2, 0, 0, 0] size [1, 1, 1, 1, 64] to %35 size [64] : memref<2x256x1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
                      }
                      ktdf_lowering.execute_on %18, %20 {
                        %37 = ktdf.read_from_fifo %35 : <"LXLU" -> "SFP", 64xf16> -> tensor<1x64xf16>
                        %38 = bufferization.to_tensor %27 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
                        %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<1x64xf16>) outs(%38 : tensor<1x64xf16>) {
                        ^bb0(%in: f16, %out: f16):
                          %40 = arith.addf %in, %out : f16
                          linalg.yield %40 : f16
                        } -> tensor<1x64xf16>
                        bufferization.materialize_in_destination %39 in writable %27 : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
                      }
                    }
                  }
                }
              }
              %32 = bufferization.to_tensor %27 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
              ktdf.write_to_fifo %32, %30 : tensor<1x64xf16>, <"SFP" -> "LXSU", 64xf16>
            }
            ktdf_lowering.execute_on %22, %24 {
              ktdf.data_transfer from %30 size [64] to %28[%arg0, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<2x1x64xf16, "LX">
            }
          }
          ktdf.parallel_yield
        }
      }
      ktdf_lowering.signal %22, %24, %12
      ktdf_lowering.execute_on %12 {
        scf.for %arg0 = %c0_7 to %c2 step %c1 {
          ktdf.data_transfer from %28[%arg0, 0, 0] size [1, 1, 64] to %reinterpret_cast_9[%arg0, 0] size [1, 64] : memref<2x1x64xf16, "LX">, memref<2x64xf16, strided<[64, 1]>, "HBM">
        }
      }
    }
    return
  }
  func.func private @"local-schedule-0_keep_alive"() {
    call @"local-schedule-0"() : () -> ()
    return
  }
}

