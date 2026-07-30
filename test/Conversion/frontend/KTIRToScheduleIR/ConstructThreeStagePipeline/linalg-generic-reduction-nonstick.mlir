// RUN: dataflow-scheduler-opt -construct-three-stage-pipeline %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = {kind = "HBM", size = 1073741824 : i64}
// CHECK: #[[$ATTR_1:.+]] = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
// CHECK: #[[$ATTR_2:.+]] = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}
// CHECK: #[[$ATTR_3:.+]] = {kind = "LX", size = 1048576 : i64}
// CHECK: #[[$ATTR_4:.+]] = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
// CHECK: #[[$ATTR_5:.+]] = {kind = "LXSU", load_store}
// CHECK: #[[$ATTR_6:.+]] = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
// CHECK: #[[$ATTR_7:.+]] = {kind = "core"}
// CHECK: #[[$ATTR_8:.+]] = {kind = "lrfreg", ktdf_arch.features = {ktdf_arch.feature.local_to_compute}, size = 65536 : i64}
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_11:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_12:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @spyre_1core attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.spyre_memory_space<HBM> = "HBM", #ktdp.spyre_memory_space<LX> = "LX">} {
// CHECK:           %[[VAL_0:.*]] = memory #[[$ATTR_0]]
// CHECK:           group #[[$ATTR_7]] share(%[[VAL_0]]) {
// CHECK:             %[[VAL_1:.*]] = memory #[[$ATTR_3]]
// CHECK:             %[[VAL_2:.*]] = exec_unit #[[$ATTR_1]]
// CHECK:             %[[VAL_3:.*]] = exec_unit #[[$ATTR_2]]
// CHECK:             datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_0]] to %[[VAL_2]] : memory, exec_unit
// CHECK:             datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_2]] to %[[VAL_1]] : exec_unit, memory
// CHECK:             datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_1]] to %[[VAL_3]] : memory, exec_unit
// CHECK:             datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %[[VAL_3]] to %[[VAL_0]] : exec_unit, memory
// CHECK:             group share(%[[VAL_1]]) {
// CHECK:               %[[VAL_4:.*]] = memory #[[$ATTR_8]]
// CHECK:               %[[VAL_5:.*]] = exec_unit #[[$ATTR_6]]
// CHECK:               %[[VAL_6:.*]] = exec_unit #[[$ATTR_4]]
// CHECK:               %[[VAL_7:.*]] = exec_unit #[[$ATTR_5]]
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_1]] to %[[VAL_6]] : memory, exec_unit
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_6]] to %[[VAL_5]] : exec_unit, exec_unit
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_5]] to %[[VAL_7]] : exec_unit, exec_unit
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_7]] to %[[VAL_1]] : exec_unit, memory
// CHECK:             }
// CHECK:             group share(%[[VAL_1]]) {
// CHECK:               %[[VAL_8:.*]] = memory #[[$ATTR_8]]
// CHECK:               %[[VAL_9:.*]] = exec_unit #[[$ATTR_6]]
// CHECK:               %[[VAL_10:.*]] = exec_unit #[[$ATTR_4]]
// CHECK:               %[[VAL_11:.*]] = exec_unit #[[$ATTR_5]]
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_1]] to %[[VAL_10]] : memory, exec_unit
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_10]] to %[[VAL_9]] : exec_unit, exec_unit
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %[[VAL_9]] to %[[VAL_11]] : exec_unit, exec_unit
// CHECK:               datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %[[VAL_11]] to %[[VAL_1]] : exec_unit, memory
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK-LABEL:   func.func private @"local-schedule-0"() attributes {grid = [1]} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 8589934592 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_11:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_11]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_13:.*]] = memref.memory_space_cast %[[VAL_11]] : memref<2x256x64xf16> to memref<2x256x64xf16, "HBM">
// CHECK:           %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_13]] to offset: {{\[}}%[[VAL_12]]], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "HBM"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "HBM">
// CHECK:           %[[VAL_15:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_12]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = memref.memory_space_cast %[[VAL_15]] : memref<2x64xf16> to memref<2x64xf16, "HBM">
// CHECK:           %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_17]] to offset: {{\[}}%[[VAL_16]]], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "HBM"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "HBM">
// CHECK:           scf.for %[[VAL_19:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] {
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_10]] {
// CHECK:               ktdf.pipeline {
// CHECK:                 %[[VAL_21:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>, !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK:                   %[[VAL_22:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>
// CHECK:                   %[[VAL_23:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>
// CHECK:                   %[[VAL_24:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                   %[[VAL_25:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                   ktdf.private_yield %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]] : !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>, !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, !ktdf.token, !ktdf.token
// CHECK:                 }
// CHECK:                 ktdf.stage depends_in(none) depends_out(%[[VAL_26:.*]]#2) {
// CHECK:                   %[[VAL_27:.*]] = arith.constant 0 : index
// CHECK:                   %[[VAL_28:.*]] = arith.constant 256 : index
// CHECK:                   %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:                   scf.for %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_29]] {
// CHECK:                     ktdf.data_transfer from %[[VAL_14]]{{\[}}%[[VAL_19]], %[[VAL_30]], %[[VAL_20]]] size [1, 1, 64] to %[[VAL_26]]#0 size [64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "HBM">, !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:                 }
// CHECK:                 ktdf.stage depends_in(%[[VAL_31:.*]]#2) depends_out(%[[VAL_31]]#3) {
// CHECK:                   %[[VAL_32:.*]] = memref.alloc() : memref<1x64xf16, "lrfreg">
// CHECK:                   %[[VAL_33:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:                   linalg.fill ins(%[[VAL_33]] : f16) outs(%[[VAL_32]] : memref<1x64xf16, "lrfreg">)
// CHECK:                   %[[VAL_34:.*]] = arith.constant 0 : index
// CHECK:                   %[[VAL_35:.*]] = arith.constant 256 : index
// CHECK:                   %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:                   scf.for %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_35]] step %[[VAL_36]] {
// CHECK:                     %[[VAL_38:.*]] = ktdf.read_from_fifo %[[VAL_31]]#0 : <"HBM" -> "SFP", 64xf16> -> memref<1x1x64xf16>
// CHECK:                     linalg.generic {indexing_maps = [#[[$ATTR_9]], #[[$ATTR_10]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[VAL_38]] : memref<1x1x64xf16>) outs(%[[VAL_32]] : memref<1x64xf16, "lrfreg">) {
// CHECK:                     ^bb0(%[[VAL_39:.*]]: f16, %[[VAL_40:.*]]: f16):
// CHECK:                       %[[VAL_41:.*]] = arith.addf %[[VAL_39]], %[[VAL_40]] : f16
// CHECK:                       linalg.yield %[[VAL_41]] : f16
// CHECK:                     }
// CHECK:                     %[[VAL_42:.*]] = arith.constant 255 : index
// CHECK:                     %[[VAL_43:.*]] = arith.cmpi eq, %[[VAL_37]], %[[VAL_42]] : index
// CHECK:                     scf.if %[[VAL_43]] {
// CHECK:                       ktdf.write_to_fifo %[[VAL_32]], %[[VAL_31]]#1 : memref<1x64xf16, "lrfreg">, <"SFP" -> "HBM", 64xf16>
// CHECK:                     }
// CHECK:                   } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:                 } {applicable_units = ["SFP"]}
// CHECK:                 ktdf.stage depends_in(%[[VAL_44:.*]]#3) depends_out(none) {
// CHECK:                   %[[VAL_45:.*]] = arith.constant 0 : index
// CHECK:                   ktdf.data_transfer from %[[VAL_44]]#1 size [64] to %[[VAL_18]]{{\[}}%[[VAL_19]], %[[VAL_20]]] size [1, 64] : !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, memref<2x64xf16, strided<[64, 1], offset: ?>, "HBM">
// CHECK:                 }
// CHECK:               }
// CHECK:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:           return
// CHECK:         }


// ---- Input IR (post compute-group-extraction, with inlined spyre_1core device) ----

#HBM = {kind = "HBM", size = 1073741824 : i64}
#L3LU = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
#L3SU = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}
#LX = {kind = "LX", size = 1048576 : i64}
#LXLU = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
#LXSU = {kind = "LXSU", load_store}
#SFP = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
#core = {kind = "core"}
#lrfreg = {kind = "lrfreg", size = 65536 : i64, ktdf_arch.features = {ktdf_arch.feature.local_to_compute}}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @spyre_1core attributes {
    mem_space_mapping = #ktdf_arch.map<
      #ktdp.spyre_memory_space<HBM> = "HBM",
      #ktdp.spyre_memory_space<LX>  = "LX"
    >
  } {
    %memory = memory #HBM
    group #core share(%memory) {
      %memory_0 = memory #LX
      %exec_unit = exec_unit #L3LU
      %exec_unit_2 = exec_unit #L3SU
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory to %exec_unit : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit to %memory_0 : exec_unit, memory
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory_0 to %exec_unit_2 : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit_2 to %memory : exec_unit, memory
      group share(%memory_0) {
        %lrfreg_0 = memory #lrfreg
        %exec_unit_3 = exec_unit #SFP
        %exec_unit_4 = exec_unit #LXLU
        %exec_unit_5 = exec_unit #LXSU
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %memory_0 to %exec_unit_4 : memory, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_4 to %exec_unit_3 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_3 to %exec_unit_5 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %exec_unit_5 to %memory_0 : exec_unit, memory
      }
      group share(%memory_0) {
        %lrfreg_1 = memory #lrfreg
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

  // Post-compute-group-extraction form of the loopless nonstick sum:
  //   a: [2, 256, 64] f16, HBM base 0
  //   c: [2, 64] f16, HBM base 8589934592
  // linalg.generic with iterator_types = ["parallel", "reduction", "parallel"]
  // Pass 01 leaves the linalg.generic intact (no scf.for wrapping) because the
  // KTIR had no enclosing loop -- the reduction dim is expressed as a
  // reduction iterator inside the generic itself.
  func.func private @"local-schedule-0"() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %a_view = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {
      coordinate_set = #set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x256x64xf16>
    %c0_idx = arith.constant 0 : index
    %a_tile = ktdp.construct_access_tile %a_view[%c0_idx, %c0_idx, %c0_idx] {
      access_tile_order = #map,
      access_tile_set = #set
    } : memref<2x256x64xf16> -> !ktdp.access_tile<2x256x64xindex>
    %a_tensor = ktdp.load %a_tile : !ktdp.access_tile<2x256x64xindex> -> tensor<2x256x64xf16>
    %zero_2d = tensor.empty() : tensor<2x64xf16>
    %c8589934592 = arith.constant 8589934592 : index
    %c_view = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {
      coordinate_set = #set1,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x64xf16>
    %c_result = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "reduction", "parallel"]
      } ins(%a_tensor : tensor<2x256x64xf16>)
        outs(%zero_2d : tensor<2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    } -> tensor<2x64xf16>
    %c_tile = ktdp.construct_access_tile %c_view[%c0_idx, %c0_idx] {
      access_tile_order = #map2,
      access_tile_set = #set1
    } : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
    ktdp.store %c_result, %c_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
    return
  }
}
