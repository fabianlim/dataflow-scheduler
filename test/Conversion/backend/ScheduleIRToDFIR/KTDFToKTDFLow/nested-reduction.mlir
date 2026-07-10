// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering -allow-unregistered-dialect %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = {kind = "HBM", size = 1073741824 : i64}
// CHECK: #[[$ATTR_1:.+]] = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
// CHECK: #[[$ATTR_2:.+]] = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}
// CHECK: #[[$ATTR_3:.+]] = {kind = "LX", size = 1048576 : i64}
// CHECK: #[[$ATTR_4:.+]] = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
// CHECK: #[[$ATTR_5:.+]] = {kind = "LXSU", load_store}
// CHECK: #[[$ATTR_6:.+]] = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
// CHECK: #[[$ATTR_7:.+]] = {kind = "core"}
// CHECK: #[[$ATTR_8:.+]] = {kind = "lrfreg", size = 65536 : i64}
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_10:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_11:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
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
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3su", type = "l3su"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
// CHECK-NEXT:     %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
// CHECK-NEXT:     %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
// CHECK-NEXT:     %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_2]] -> %[[GET_UNIT_2]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_4]] -> %[[GET_UNIT_4]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_5]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_6]] -> %[[GET_UNIT_6]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_7]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_9:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:     %[[CONSTANT_10:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_11:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_8]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_10]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_9]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_11]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "HBM">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1]>, "HBM">
// CHECK-NEXT:     ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:       %[[CONSTANT_13:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_13]] : index to memref<1x64xf16, "lrfreg">
// CHECK-NEXT:       %[[CONSTANT_14:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_14]] : index to memref<2x1x64xf16, "LX">
// CHECK-NEXT:       %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:         ktdf.parallel (%[[VAL_0:.*]], %[[VAL_1:.*]]) = (%[[CONSTANT_8]]) to (%[[CONSTANT_11]]) step (%[[CONSTANT_12]]) distribute(num_instances = 2) {
// CHECK-NEXT:           ktdf_lowering.execute_on %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:             %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf_lowering.execute_on %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:               ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:                 %[[CONSTANT_15:.*]] = arith.constant 256 : index
// CHECK-NEXT:                 %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_15]] : index to memref<2x256x1x1x64xf16, "LX">
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:                   scf.for %[[VAL_2:.*]] = %[[CONSTANT_8]] to %[[CONSTANT_10]] step %[[CONSTANT_12]] {
// CHECK-NEXT:                     ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_0]], %[[VAL_2]], 0] size [1, 1, 64] to %[[UNREALIZED_CONVERSION_CAST_2]]{{\[}}%[[VAL_0]], %[[VAL_2]], 0, 0, 0] size [1, 1, 1, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, memref<2x256x1x1x64xf16, "LX">
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf_lowering.signal %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK-NEXT:                 ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:                   scf.for %[[VAL_3:.*]] = %[[CONSTANT_8]] to %[[CONSTANT_10]] step %[[CONSTANT_12]] {
// CHECK-NEXT:                     ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:                       %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK-NEXT:                       %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                       ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]] {
// CHECK-NEXT:                         ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_2]]{{\[}}%[[VAL_0]], %[[VAL_3]], 0, 0, 0] size [1, 1, 1, 1, 64] to %[[FIFO_1]] size [64] : memref<2x256x1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK-NEXT:                       }
// CHECK-NEXT:                       ktdf_lowering.execute_on %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:                         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_1]] : <"LXLU" -> "SFP", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:                         %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[UNREALIZED_CONVERSION_CAST_0]] restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
// CHECK-NEXT:                         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_9]], #[[$ATTR_9]]], iterator_types = ["parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x64xf16>) outs(%[[TO_TENSOR_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                         ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:                           %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK-NEXT:                           linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                         } -> tensor<1x64xf16>
// CHECK-NEXT:                         bufferization.materialize_in_destination %[[GENERIC_0]] in writable %[[UNREALIZED_CONVERSION_CAST_0]] : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               %[[TO_TENSOR_1:.*]] = bufferization.to_tensor %[[UNREALIZED_CONVERSION_CAST_0]] restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
// CHECK-NEXT:               ktdf.write_to_fifo %[[TO_TENSOR_1]], %[[FIFO_0]] : tensor<1x64xf16>, <"SFP" -> "LXSU", 64xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf_lowering.execute_on %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:               ktdf.data_transfer from %[[FIFO_0]] size [64] to %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_0]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<2x1x64xf16, "LX">
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.parallel_yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]]
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_1]] {
// CHECK-NEXT:         scf.for %[[VAL_6:.*]] = %[[CONSTANT_8]] to %[[CONSTANT_11]] step %[[CONSTANT_12]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], 0, 0] size [1, 1, 64] to %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_6]], 0] size [1, 64] : memref<2x1x64xf16, "LX">, memref<2x64xf16, strided<[64, 1]>, "HBM">
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func private @"local-schedule-0_keep_alive"() {
// CHECK-NEXT:     call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// Pass-19 KTDFToKTDFLow on the nested reduction: characterizes ComponentClassifier's unit
// assignment (SFP accumulate, LXLU inner-load, L3 load/store) and SignalInsertion placement on
// the nested stages. NOTE: pass-19 strips loop_type -- the output scf.for ops carry no
// loop_type attr, so pass-20 must detect the peelable reduction loop structurally.
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
    %c0 = arith.constant 0 : index
    %c8589934592 = arith.constant 8589934592 : index
    %c256 = arith.constant 256 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
    %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
    %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, "HBM">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1]>, "HBM">
    ktdf.pipeline {
      %2:3 = ktdf.private -> (memref<1x64xf16, "lrfreg">, memref<2x1x64xf16, "LX">, !ktdf.token) {
        %c0_2 = arith.constant 0 : index
        %3 = builtin.unrealized_conversion_cast %c0_2 : index to memref<1x64xf16, "lrfreg">
        %c0_3 = arith.constant 0 : index
        %4 = builtin.unrealized_conversion_cast %c0_3 : index to memref<2x1x64xf16, "LX">
        %5 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %3, %4, %5 : memref<1x64xf16, "lrfreg">, memref<2x1x64xf16, "LX">, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%2#2) {
        ktdf.parallel (%arg0, %arg1) = (%c0) to (%c2) step (%c1) distribute(num_instances = 2) {
          ktdf.pipeline {
            %3:2 = ktdf.private -> (!ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token) {
              %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
              %5 = ktdf.create_token : !ktdf.token
              ktdf.private_yield %4, %5 : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token
            }
            ktdf.stage depends_in(none) depends_out(%3#1) {
              ktdf.pipeline {
                %5:2 = ktdf.private -> (memref<2x256x1x1x64xf16, "LX">, !ktdf.token) {
                  %c256_2 = arith.constant 256 : index
                  %6 = builtin.unrealized_conversion_cast %c256_2 : index to memref<2x256x1x1x64xf16, "LX">
                  %7 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %6, %7 : memref<2x256x1x1x64xf16, "LX">, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%5#1) {
                  scf.for %arg2 = %c0 to %c256 step %c1 {
                    ktdf.data_transfer from %reinterpret_cast[%arg0, %arg2, 0] size [1, 1, 64] to %5#0[%arg0, %arg2, 0, 0, 0] size [1, 1, 1, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, memref<2x256x1x1x64xf16, "LX">
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["L3LU"]}
                ktdf.stage depends_in(%5#1) depends_out(none) {
                  scf.for %arg2 = %c0 to %c256 step %c1 {
                    ktdf.pipeline {
                      %6:2 = ktdf.private -> (!ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token) {
                        %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
                        %8 = ktdf.create_token : !ktdf.token
                        ktdf.private_yield %7, %8 : !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token
                      }
                      ktdf.stage depends_in(none) depends_out(%6#1) {
                        ktdf.data_transfer from %5#0[%arg0, %arg2, 0, 0, 0] size [1, 1, 1, 1, 64] to %6#0 size [64] : memref<2x256x1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
                      } {applicable_units = ["LXLU"]}
                      ktdf.stage depends_in(%6#1) depends_out(none) {
                        %7 = ktdf.read_from_fifo %6#0 : <"LXLU" -> "SFP", 64xf16> -> tensor<1x64xf16>
                        %8 = bufferization.to_tensor %2#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
                        %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<1x64xf16>) outs(%8 : tensor<1x64xf16>) {
                        ^bb0(%in: f16, %out: f16):
                          %10 = arith.addf %in, %out : f16
                          linalg.yield %10 : f16
                        } -> tensor<1x64xf16>
                        bufferization.materialize_in_destination %9 in writable %2#0 : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
                      } {applicable_units = ["SFP"]}
                    }
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["LXLU", "SFP"]}
              }
              %4 = bufferization.to_tensor %2#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
              ktdf.write_to_fifo %4, %3#0 : tensor<1x64xf16>, <"SFP" -> "LXSU", 64xf16>
            } {applicable_units = ["SFP"]}
            ktdf.stage depends_in(%3#1) depends_out(none) {
              ktdf.data_transfer from %3#0 size [64] to %2#1[%arg0, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<2x1x64xf16, "LX">
            } {applicable_units = ["LXSU"]}
          }
          ktdf.parallel_yield
        }
      } {applicable_units = ["L3LU", "LXLU", "SFP", "LXSU"]}
      ktdf.stage depends_in(%2#2) depends_out(none) {
        scf.for %arg0 = %c0 to %c2 step %c1 {
          ktdf.data_transfer from %2#1[%arg0, 0, 0] size [1, 1, 64] to %reinterpret_cast_1[%arg0, 0] size [1, 64] : memref<2x1x64xf16, "LX">, memref<2x64xf16, strided<[64, 1]>, "HBM">
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {applicable_units = ["L3SU"]}
    }
    return
  }
  func.func private @"local-schedule-0_keep_alive"() {
    call @"local-schedule-0"() : () -> ()
    return
  }
}

