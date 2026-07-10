// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s


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

// CHECK-LABEL:   module {
// CHECK-NEXT:     func.func private @"local-schedule-0"() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 256 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_10]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_1]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_11]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "HBM">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1]>, "HBM">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:3 = ktdf.private -> (memref<1x64xf16, "lrfreg">, memref<2x1x64xf16, "LX">, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "lrfreg">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "LX">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]] : memref<1x64xf16, "lrfreg">, memref<2x1x64xf16, "LX">, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[PRIVATE_1:.*]]:2 = ktdf.private -> (!ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token) {
// CHECK-NEXT:                 %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[FIFO_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_2:.*]]#1) {
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[PRIVATE_2:.*]]:2 = ktdf.private -> (memref<2x256x1x1x64xf16, "LX">, !ktdf.token) {
// CHECK-NEXT:                     %[[ALLOC_2:.*]] = memref.alloc() : memref<2x256x1x1x64xf16, "LX">
// CHECK-NEXT:                     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[ALLOC_2]], %[[CREATE_TOKEN_2]] : memref<2x256x1x1x64xf16, "LX">, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_3:.*]]#1) {
// CHECK-NEXT:                     scf.for %[[VAL_4:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:                       %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                       %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                       %[[SUBI_1:.*]] = arith.subi %[[VAL_4]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                       %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                       ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_1]], %[[VAL_4]], 0] size [1, 1, 64] to %[[VAL_3]]#0{{\[}}%[[DIVSI_0]], %[[DIVSI_1]], 0, 0, 0] size [1, 1, 1, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, memref<2x256x1x1x64xf16, "LX">
// CHECK-NEXT:                     } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                   } {applicable_units = ["L3LU"]}
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_5:.*]]#1) depends_out(none) {
// CHECK-NEXT:                     scf.for %[[VAL_6:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:                       ktdf.pipeline {
// CHECK-NEXT:                         %[[PRIVATE_3:.*]]:2 = ktdf.private -> (!ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token) {
// CHECK-NEXT:                           %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK-NEXT:                           %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                           ktdf.private_yield %[[FIFO_1]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token
// CHECK-NEXT:                         }
// CHECK-NEXT:                         ktdf.stage depends_in(none) depends_out(%[[VAL_7:.*]]#1) {
// CHECK-NEXT:                           %[[SUBI_2:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                           %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                           %[[SUBI_3:.*]] = arith.subi %[[VAL_6]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                           %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                           ktdf.data_transfer from %[[VAL_5]]#0{{\[}}%[[DIVSI_2]], %[[DIVSI_3]], 0, 0, 0] size [1, 1, 1, 1, 64] to %[[VAL_7]]#0 size [64] : memref<2x256x1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK-NEXT:                         } {applicable_units = ["LXLU"]}
// CHECK-NEXT:                         ktdf.stage depends_in(%[[VAL_8:.*]]#1) depends_out(none) {
// CHECK-NEXT:                           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_8]]#0 : <"LXLU" -> "SFP", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:                           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[VAL_0]]#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
// CHECK-NEXT:                           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_9]], #[[$ATTR_9]]], iterator_types = ["parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x64xf16>) outs(%[[TO_TENSOR_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                           ^bb0(%[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK-NEXT:                             %[[ADDF_0:.*]] = arith.addf %[[VAL_9]], %[[VAL_10]] : f16
// CHECK-NEXT:                             linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                           } -> tensor<1x64xf16>
// CHECK-NEXT:                           bufferization.materialize_in_destination %[[GENERIC_0]] in writable %[[VAL_0]]#0 : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
// CHECK-NEXT:                         } {applicable_units = ["SFP"]}
// CHECK-NEXT:                       }
// CHECK-NEXT:                     } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                   } {applicable_units = ["LXLU", "SFP"]}
// CHECK-NEXT:                 }
// CHECK-NEXT:                 %[[TO_TENSOR_1:.*]] = bufferization.to_tensor %[[VAL_0]]#0 restrict : memref<1x64xf16, "lrfreg"> to tensor<1x64xf16>
// CHECK-NEXT:                 ktdf.write_to_fifo %[[TO_TENSOR_1]], %[[VAL_2]]#0 : tensor<1x64xf16>, <"SFP" -> "LXSU", 64xf16>
// CHECK-NEXT:               } {applicable_units = ["SFP"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_11:.*]]#1) depends_out(none) {
// CHECK-NEXT:                 %[[SUBI_4:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_4:.*]] = arith.divsi %[[SUBI_4]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_11]]#0 size [64] to %[[VAL_0]]#1{{\[}}%[[DIVSI_4]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<2x1x64xf16, "LX">
// CHECK-NEXT:               } {applicable_units = ["LXSU"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L3LU", "LXLU", "SFP", "LXSU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_12:.*]]#2) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_13:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             %[[SUBI_5:.*]] = arith.subi %[[VAL_13]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_5:.*]] = arith.divsi %[[SUBI_5]], %[[CONSTANT_4]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_12]]#1{{\[}}%[[DIVSI_5]], 0, 0] size [1, 1, 64] to %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_13]], 0] size [1, 64] : memref<2x1x64xf16, "LX">, memref<2x64xf16, strided<[64, 1]>, "HBM">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L3SU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @"local-schedule-0_keep_alive"() {
// CHECK-NEXT:       call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// Pass-09 StageCoarsening on the nested reduction: a ktdf.stage whose child is a nested
// ktdf.pipeline is treated as a coarsening LEAF (verbatim-cloned, not re-entered). The outer
// tile loop is distributed around it, but no tile loop is sunk into its interior and its
// buffers are not double-expanded. Guards the Materializer isLoopNode assert.
#map = affine_map<() -> (0, 0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<() -> (0, 0)>
#map4 = affine_map<(d0) -> (d0, 0)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
"builtin.module"() ({
  "ktdf_arch.device"() <{sym_name = "spyre_1core"}> ({
    %27 = "ktdf_arch.memory"() <{kind = "HBM", size = 1073741824 : i64}> : () -> !ktdf_arch.memory
    "ktdf_arch.group"(%27) <{kind = "core"}> ({
    ^bb0(%arg4: !ktdf_arch.memory):
      %28 = "ktdf_arch.memory"() <{kind = "LX", size = 1048576 : i64}> : () -> !ktdf_arch.memory
      %29 = "ktdf_arch.memory"() <{kind = "lrfreg", size = 65536 : i64}> : () -> !ktdf_arch.memory
      %30 = "ktdf_arch.exec_unit"() <{kind = "L3LU", load_store}> {dataflow_scheduler.double_buffer_last} : () -> !ktdf_arch.exec_unit
      %31 = "ktdf_arch.exec_unit"() <{kind = "L3SU", load_store}> {dataflow_scheduler.double_buffer_last} : () -> !ktdf_arch.exec_unit
      "ktdf_arch.datapath"(%arg4, %30) {ktdf_arch.transfer_granularity = array<i64: 64>} : (!ktdf_arch.memory, !ktdf_arch.exec_unit) -> ()
      "ktdf_arch.datapath"(%30, %28) {ktdf_arch.transfer_granularity = array<i64: 64>} : (!ktdf_arch.exec_unit, !ktdf_arch.memory) -> ()
      "ktdf_arch.datapath"(%28, %31) {ktdf_arch.transfer_granularity = array<i64: 64>} : (!ktdf_arch.memory, !ktdf_arch.exec_unit) -> ()
      "ktdf_arch.datapath"(%31, %arg4) {ktdf_arch.transfer_granularity = array<i64: 64>} : (!ktdf_arch.exec_unit, !ktdf_arch.memory) -> ()
      "ktdf_arch.group"(%28) ({
      ^bb0(%arg6: !ktdf_arch.memory):
        %35 = "ktdf_arch.exec_unit"() <{kind = "SFP"}> {ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}} : () -> !ktdf_arch.exec_unit
        %36 = "ktdf_arch.exec_unit"() <{kind = "LXLU", load_store}> {ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}} : () -> !ktdf_arch.exec_unit
        %37 = "ktdf_arch.exec_unit"() <{kind = "LXSU", load_store}> : () -> !ktdf_arch.exec_unit
        "ktdf_arch.datapath"(%arg6, %36) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} : (!ktdf_arch.memory, !ktdf_arch.exec_unit) -> ()
        "ktdf_arch.datapath"(%36, %35) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} : (!ktdf_arch.exec_unit, !ktdf_arch.exec_unit) -> ()
        "ktdf_arch.datapath"(%35, %37) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} : (!ktdf_arch.exec_unit, !ktdf_arch.exec_unit) -> ()
        "ktdf_arch.datapath"(%37, %arg6) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} : (!ktdf_arch.exec_unit, !ktdf_arch.memory) -> ()
        "ktdf_arch.yield"() : () -> ()
      }) : (!ktdf_arch.memory) -> ()
      "ktdf_arch.group"(%28) ({
      ^bb0(%arg5: !ktdf_arch.memory):
        %32 = "ktdf_arch.exec_unit"() <{kind = "SFP"}> {ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}} : () -> !ktdf_arch.exec_unit
        %33 = "ktdf_arch.exec_unit"() <{kind = "LXLU", load_store}> {ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}} : () -> !ktdf_arch.exec_unit
        %34 = "ktdf_arch.exec_unit"() <{kind = "LXSU", load_store}> : () -> !ktdf_arch.exec_unit
        "ktdf_arch.datapath"(%arg5, %33) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} : (!ktdf_arch.memory, !ktdf_arch.exec_unit) -> ()
        "ktdf_arch.datapath"(%33, %32) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} : (!ktdf_arch.exec_unit, !ktdf_arch.exec_unit) -> ()
        "ktdf_arch.datapath"(%32, %34) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} : (!ktdf_arch.exec_unit, !ktdf_arch.exec_unit) -> ()
        "ktdf_arch.datapath"(%34, %arg5) {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} : (!ktdf_arch.exec_unit, !ktdf_arch.memory) -> ()
        "ktdf_arch.yield"() : () -> ()
      }) : (!ktdf_arch.memory) -> ()
      "ktdf_arch.yield"() : () -> ()
    }) : (!ktdf_arch.memory) -> ()
  }) {mem_space_mapping = #ktdf_arch.map<#ktdp.spyre_memory_space<HBM> = "HBM", #ktdp.spyre_memory_space<LX> = "LX", #ktdp.spyre_memory_space<LRF> = "lrfreg">} : () -> ()
  "builtin.module"() ({
    "func.func"() <{function_type = () -> (), sym_name = "local-schedule-0", sym_visibility = "private"}> ({
      %0 = "arith.constant"() <{value = 0 : index}> : () -> index
      %1 = "arith.constant"() <{value = 8589934592 : index}> : () -> index
      %2 = "arith.constant"() <{value = 256 : index}> : () -> index
      %3 = "arith.constant"() <{value = 2 : index}> : () -> index
      %4 = "arith.constant"() <{value = 1 : index}> : () -> index
      %5 = "ktdp.construct_memory_view"(%0) <{coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>, operandSegmentSizes = array<i32: 1, 0, 0>, static_sizes = array<i64: 2, 256, 64>, static_strides = array<i64: 16384, 64, 1>}> : (index) -> memref<2x256x64xf16>
      %6 = "memref.memory_space_cast"(%5) : (memref<2x256x64xf16>) -> memref<2x256x64xf16, "HBM">
      %7 = "memref.reinterpret_cast"(%6) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 2, 256, 64>, static_strides = array<i64: 16384, 64, 1>}> : (memref<2x256x64xf16, "HBM">) -> memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">
      %8 = "ktdp.construct_memory_view"(%1) <{coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>, operandSegmentSizes = array<i32: 1, 0, 0>, static_sizes = array<i64: 2, 64>, static_strides = array<i64: 64, 1>}> : (index) -> memref<2x64xf16>
      %9 = "memref.memory_space_cast"(%8) : (memref<2x64xf16>) -> memref<2x64xf16, "HBM">
      %10 = "memref.reinterpret_cast"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 2, 64>, static_strides = array<i64: 64, 1>}> : (memref<2x64xf16, "HBM">) -> memref<2x64xf16, strided<[64, 1]>, "HBM">
      "scf.for"(%0, %3, %4) ({
      ^bb0(%arg0: index):
        "ktdf.pipeline"() ({
          %11:5 = "ktdf.private"() ({
            %22 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x64xf16, "lrfreg">
            %23 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x64xf16, "LX">
            %24 = "ktdf.fifo.allocate"() : () -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
            %25 = "ktdf.create_token"() <{unique_id = 140734776153120 : index}> : () -> !ktdf.token
            %26 = "ktdf.create_token"() <{unique_id = 140734776153120 : index}> : () -> !ktdf.token
            "ktdf.private_yield"(%22, %23, %24, %25, %26) : (memref<1x64xf16, "lrfreg">, memref<1x64xf16, "LX">, !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token, !ktdf.token) -> ()
          }) : () -> (memref<1x64xf16, "lrfreg">, memref<1x64xf16, "LX">, !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token, !ktdf.token)
          "ktdf.stage"(%11#3) <{applicable_units = ["SFP"], operandSegmentSizes = array<i32: 0, 1>}> ({
            "scf.for"(%0, %2, %4) ({
            ^bb0(%arg1: index):
              "ktdf.pipeline"() ({
                %13:4 = "ktdf.private"() ({
                  %18 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x64xf16, "LX">
                  %19 = "ktdf.fifo.allocate"() : () -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
                  %20 = "ktdf.create_token"() <{unique_id = 140734776153120 : index}> : () -> !ktdf.token
                  %21 = "ktdf.create_token"() <{unique_id = 140734776153120 : index}> : () -> !ktdf.token
                  "ktdf.private_yield"(%18, %19, %20, %21) : (memref<1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token, !ktdf.token) -> ()
                }) : () -> (memref<1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.token, !ktdf.token)
                "ktdf.stage"(%13#2) <{applicable_units = ["L3LU"], operandSegmentSizes = array<i32: 0, 1>}> ({
                  "ktdf.data_transfer"(%7, %arg0, %arg1, %13#0) <{dest_map = #map, operandSegmentSizes = array<i32: 1, 2, 0, 1, 0, 0>, source_map = #map1, static_dest_sizes = array<i64: 1, 1, 64>, static_source_sizes = array<i64: 1, 1, 64>}> : (memref<2x256x64xf16, strided<[16384, 64, 1]>, "HBM">, index, index, memref<1x1x64xf16, "LX">) -> ()
                }) : (!ktdf.token) -> ()
                "ktdf.stage"(%13#2, %13#3) <{applicable_units = ["LXLU"], operandSegmentSizes = array<i32: 1, 1>}> ({
                  "ktdf.data_transfer"(%13#0, %13#1) <{operandSegmentSizes = array<i32: 1, 0, 0, 1, 0, 0>, source_map = #map, static_dest_sizes = array<i64: 64>, static_source_sizes = array<i64: 1, 1, 64>}> : (memref<1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>) -> ()
                }) : (!ktdf.token, !ktdf.token) -> ()
                "ktdf.stage"(%13#3) <{applicable_units = ["SFP"], operandSegmentSizes = array<i32: 1, 0>}> ({
                  %14 = "ktdf.read_from_fifo"(%13#1) : (!ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>) -> tensor<1x64xf16>
                  %15 = "bufferization.to_tensor"(%11#0) <{restrict}> : (memref<1x64xf16, "lrfreg">) -> tensor<1x64xf16>
                  %16 = "linalg.generic"(%14, %15) <{indexing_maps = [#map2, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
                  ^bb0(%arg2: f16, %arg3: f16):
                    %17 = "arith.addf"(%arg2, %arg3) <{fastmath = #arith.fastmath<none>}> : (f16, f16) -> f16
                    "linalg.yield"(%17) : (f16) -> ()
                  }) : (tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
                  "bufferization.materialize_in_destination"(%16, %11#0) <{writable}> : (tensor<1x64xf16>, memref<1x64xf16, "lrfreg">) -> ()
                }) : (!ktdf.token) -> ()
              }) : () -> ()
              "scf.yield"() : () -> ()
            }) {loop_type = #ktdf.loop_type<reduction_loop>} : (index, index, index) -> ()
            %12 = "bufferization.to_tensor"(%11#0) <{restrict}> : (memref<1x64xf16, "lrfreg">) -> tensor<1x64xf16>
            "ktdf.write_to_fifo"(%12, %11#2) : (tensor<1x64xf16>, !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>) -> ()
          }) : (!ktdf.token) -> ()
          "ktdf.stage"(%11#3, %11#4) <{applicable_units = ["LXSU"], operandSegmentSizes = array<i32: 1, 1>}> ({
            "ktdf.data_transfer"(%11#2, %11#1) <{dest_map = #map3, operandSegmentSizes = array<i32: 1, 0, 0, 1, 0, 0>, static_dest_sizes = array<i64: 1, 64>, static_source_sizes = array<i64: 64>}> : (!ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<1x64xf16, "LX">) -> ()
          }) : (!ktdf.token, !ktdf.token) -> ()
          "ktdf.stage"(%11#4) <{applicable_units = ["L3SU"], operandSegmentSizes = array<i32: 1, 0>}> ({
            "ktdf.data_transfer"(%11#1, %10, %arg0) <{dest_map = #map4, operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 0>, source_map = #map3, static_dest_sizes = array<i64: 1, 64>, static_source_sizes = array<i64: 1, 64>}> : (memref<1x64xf16, "LX">, memref<2x64xf16, strided<[64, 1]>, "HBM">, index) -> ()
          }) : (!ktdf.token) -> ()
        }) : () -> ()
        "scf.yield"() : () -> ()
      }) {loop_type = #ktdf.loop_type<parallel_loop>} : (index, index, index) -> ()
      "func.return"() : () -> ()
    }) {grid = [1]} : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "local-schedule-0_keep_alive", sym_visibility = "private"}> ({
      "func.call"() <{callee = @"local-schedule-0"}> : () -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
