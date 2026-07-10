// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s


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
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_10]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_11]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "HBM">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1]>, "HBM">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:5 = ktdf.private -> (memref<1x64xf16, "lrfreg">, memref<1x64xf16, "LX">, !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "lrfreg">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "LX">
// CHECK-NEXT:           %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[FIFO_0]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<1x64xf16, "lrfreg">, memref<1x64xf16, "LX">, !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_1:.*]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_3]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[PRIVATE_1:.*]]:4 = ktdf.private -> (memref<1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[ALLOC_2:.*]] = memref.alloc() : memref<1x1x64xf16, "LX">
// CHECK-NEXT:                 %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[ALLOC_2]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : memref<1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_3:.*]]#2) {
// CHECK-NEXT:                 ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_0]], %[[VAL_2]], 0] size [1, 1, 64] to %[[VAL_3]]#0[0, 0, 0] size [1, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, memref<1x1x64xf16, "LX">
// CHECK-NEXT:               } {applicable_units = ["L3LU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_4:.*]]#2) depends_out(%[[VAL_4]]#3) {
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_4]]#0[0, 0, 0] size [1, 1, 64] to %[[VAL_4]]#1 size [64] : memref<1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK-NEXT:               } {applicable_units = ["LXLU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_5:.*]]#3) depends_out(none) {
// CHECK-NEXT:                 %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_5]]#1 : <"LXLU" -> "SFP", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:                 %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[VAL_1]]#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
// CHECK-NEXT:                 %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_9]], #[[$ATTR_9]]], iterator_types = ["parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x64xf16>) outs(%[[TO_TENSOR_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                 ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:                   %[[ADDF_0:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:                   linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                 } -> tensor<1x64xf16>
// CHECK-NEXT:                 bufferization.materialize_in_destination %[[GENERIC_0]] in writable %[[VAL_1]]#0 : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
// CHECK-NEXT:               } {applicable_units = ["SFP"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:           %[[TO_TENSOR_1:.*]] = bufferization.to_tensor %[[VAL_1]]#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
// CHECK-NEXT:           ktdf.write_to_fifo %[[TO_TENSOR_1]], %[[VAL_1]]#2 : tensor<1x64xf16>, <"SFP" -> "LXSU", 64xf16>
// CHECK-NEXT:         } {applicable_units = ["SFP"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_8:.*]]#3) depends_out(%[[VAL_8]]#4) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_8]]#2 size [64] to %[[VAL_8]]#1[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<1x64xf16, "LX">
// CHECK-NEXT:         } {applicable_units = ["LXSU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_9:.*]]#4) depends_out(none) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_9]]#1[0, 0] size [1, 64] to %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_0]], 0] size [1, 64] : memref<1x64xf16, "LX">, memref<2x64xf16, strided<[64, 1]>, "HBM">
// CHECK-NEXT:         } {applicable_units = ["L3SU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// Pass-03 PathExpansion on the nested reduction: endpoint analysis stops at the nested
// ktdf.pipeline scope boundary, so the outer accumulate/store stages draw their endpoints from
// their own region only. Guards against the spurious HBM->HBM self-hop that previously crashed
// the planner (Planner.cpp assert).
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
    %c8589934592 = arith.constant 8589934592 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
    %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
    %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, "HBM">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1]>, "HBM">
    scf.for %arg0 = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<1x64xf16, "lrfreg">, !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<1x64xf16, "lrfreg">
          %3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>
          %4 = ktdf.create_token : !ktdf.token
          %5 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %3, %4, %5 : memref<1x64xf16, "lrfreg">, !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg1 = %c0 to %c256 step %c1 {
            ktdf.pipeline {
              %4:3 = ktdf.private -> (!ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>, !ktdf.token, !ktdf.token) {
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %5, %6, %7 : !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%4#1) {
                ktdf.data_transfer from %reinterpret_cast[%arg0, %arg1, %c0] size [1, 1, 64] to %4#0 size [64] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>
              }
              ktdf.stage depends_in(%4#1) depends_out(%4#2) {
                %5 = ktdf.read_from_fifo %4#0 : <"HBM" -> "SFP", 64xf16> -> tensor<1x64xf16>
                %6 = bufferization.to_tensor %2#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
                %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x64xf16>) outs(%6 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %8 = arith.addf %in, %out : f16
                  linalg.yield %8 : f16
                } -> tensor<1x64xf16>
                bufferization.materialize_in_destination %7 in writable %2#0 : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
              } {applicable_units = ["SFP"]}
            }
          } {loop_type = #ktdf.loop_type<reduction_loop>}
          %3 = bufferization.to_tensor %2#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
          ktdf.write_to_fifo %3, %2#1 : tensor<1x64xf16>, <"SFP" -> "HBM", 64xf16>
        }
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          ktdf.data_transfer from %2#1 size [64] to %reinterpret_cast_1[%arg0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, memref<2x64xf16, strided<[64, 1]>, "HBM">
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

