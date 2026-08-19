// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s

// How much of the SIMD register the parallel dims may claim. They get the whole
// budget only when the output's fastest-varying axis is also the loop's, i.e.
// when the axis it lands on is the innermost stride-1 run; otherwise the budget
// goes to an outer, non-unit-stride axis and the transfer becomes a gather.
//
//   @onstick_lane_innermost -- iterator_types = ["reduction","parallel",
//     "reduction"], output map (d1). Innermost d2 is the reduced lane axis, which
//     the output does not vary along, so the register is already accounted for
//     and d1 is walked one slice at a time: tile [0,1,0], <2x1x64> compute tile,
//     128-element input FIFO. d1 has stride 64, so tiling it to 64 would ask for
//     64 rows strided 64 apart -- the gather to avoid.
//
//   @onstick_output_lane_innermost -- iterator_types = ["reduction","parallel",
//     "reduction","parallel"], output map (d1,d3). Innermost d3 is parallel *and*
//     the output's fastest axis, so it takes the full budget: tile [0,1,0,64].
//     Same compute tile and FIFO, but the output tile is a whole <1x64> stick.
//
// The distinction is about the output map, not iterator types: both reduce their
// innermost-but-one axis and differ only in whether the output has a lane axis.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   func.func @onstick_lane_innermost() attributes {grid = [1]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [256], strides: [64] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<256xf16>
// CHECK-NEXT:     %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[CONSTANT_8]]], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<256xf16> to memref<256xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[CONSTANT_9]]], sizes: [256], strides: [1] : memref<256xf16, "DDR"> to memref<256xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_6]] step %[[CONSTANT_7]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 1xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>
// CHECK-NEXT:           %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 1xf16>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 1xf16>, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_1:.*]]#2) {
// CHECK-NEXT:           ktdf.data_transfer from %[[REINTERPRET_CAST_0]][0, %[[VAL_0]], 0] size [2, 1, 64] to %[[VAL_1]]#0 size [2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_2]]#0 : <"DDR" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
// CHECK-NEXT:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1xf16>
// CHECK-NEXT:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "reduction"]} ins(%[[READ_FROM_FIFO_0]] : tensor<2x1x64xf16>) outs(%[[EMPTY_0]] : tensor<1xf16>) {
// CHECK-NEXT:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:           } -> tensor<1xf16>
// CHECK-NEXT:           ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_2]]#1 : tensor<1xf16>, <"SFU" -> "DDR", 1xf16>
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_5:.*]]#3) depends_out(%[[VAL_5]]#4) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_5]]#1 size [1] to %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_0]]] size [1] : !ktdf.fifo.slot<"SFU" -> "DDR", 1xf16>, memref<256xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-LABEL:   func.func @onstick_output_lane_innermost() attributes {grid = [1]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_10:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_11:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [256, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_6]], memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
// CHECK-NEXT:     %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[CONSTANT_12]]], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTANT_13:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<256x64xf16> to memref<256x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[CONSTANT_13]]], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, "DDR"> to memref<256x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_9]] to %[[CONSTANT_10]] step %[[CONSTANT_11]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>
// CHECK-NEXT:             %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_2:.*]]#2) {
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_0]][0, %[[VAL_0]], 0] size [2, 1, 64] to %[[VAL_2]]#0 size [2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_3:.*]]#2) depends_out(%[[VAL_3]]#3) {
// CHECK-NEXT:             %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_3]]#0 : <"DDR" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:             %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<2x1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:             ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:               %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK-NEXT:               linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:             } -> tensor<1x64xf16>
// CHECK-NEXT:             ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_3]]#1 : tensor<1x64xf16>, <"SFU" -> "DDR", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["SFU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_6:.*]]#3) depends_out(%[[VAL_6]]#4) {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_6]]#1 size [1, 64] to %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<256x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @onstick_lane_innermost() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %a_base = arith.constant 0 : index
    %c_base = arith.constant 8589934592 : index

    %a_view = ktdp.construct_memory_view %a_base, sizes: [2, 256, 64], strides: [16384, 64, 1] {
      coordinate_set = affine_set<(d0,d1,d2):(d0>=0,-d0+1>=0,d1>=0,-d1+255>=0,d2>=0,-d2+63>=0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<2x256x64xf16>

    %c_view = ktdp.construct_memory_view %c_base, sizes: [256], strides: [64] {
      coordinate_set = affine_set<(d0):(d0>=0,-d0+255>=0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<256xf16>

    %a_tile = ktdp.construct_access_tile %a_view[%c0, %c0, %c0] {
      access_tile_set = affine_set<(d0,d1,d2):(d0>=0,-d0+1>=0,d1>=0,-d1+255>=0,d2>=0,-d2+63>=0)>,
      access_tile_order = affine_map<(d0,d1,d2)->(d0,d1,d2)>
    } : memref<2x256x64xf16> -> !ktdp.access_tile<2x256x64xindex>
    %a_tensor = ktdp.load %a_tile : !ktdp.access_tile<2x256x64xindex> -> tensor<2x256x64xf16>

    %init = tensor.empty() : tensor<256xf16>
    %c_result = linalg.generic {
        indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>,
          affine_map<(d0,d1,d2) -> (d1)>],
        iterator_types = ["reduction", "parallel", "reduction"]
      } ins(%a_tensor : tensor<2x256x64xf16>)
        outs(%init : tensor<256xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    } -> tensor<256xf16>

    %c_tile = ktdp.construct_access_tile %c_view[%c0] {
      access_tile_set = affine_set<(d0):(d0>=0,-d0+255>=0)>,
      access_tile_order = affine_map<(d0)->(d0)>
    } : memref<256xf16> -> !ktdp.access_tile<256xindex>
    ktdp.store %c_result, %c_tile : tensor<256xf16>, !ktdp.access_tile<256xindex>

    return
  }

  func.func @onstick_output_lane_innermost() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %a_base = arith.constant 0 : index
    %c_base = arith.constant 8589934592 : index

    %a_view = ktdp.construct_memory_view %a_base, sizes: [2, 256, 64], strides: [16384, 64, 1] {
      coordinate_set = affine_set<(d0,d1,d2):(d0>=0,-d0+1>=0,d1>=0,-d1+255>=0,d2>=0,-d2+63>=0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<2x256x64xf16>

    %c_view = ktdp.construct_memory_view %c_base, sizes: [256, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0,d1):(d0>=0,-d0+255>=0,d1>=0,-d1+63>=0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<256x64xf16>

    %a_tile = ktdp.construct_access_tile %a_view[%c0, %c0, %c0] {
      access_tile_set = affine_set<(d0,d1,d2):(d0>=0,-d0+1>=0,d1>=0,-d1+255>=0,d2>=0,-d2+63>=0)>,
      access_tile_order = affine_map<(d0,d1,d2)->(d0,d1,d2)>
    } : memref<2x256x64xf16> -> !ktdp.access_tile<2x256x64xindex>
    %a_tensor = ktdp.load %a_tile : !ktdp.access_tile<2x256x64xindex> -> tensor<2x256x64xf16>

    %init = tensor.empty() : tensor<256x64xf16>
    %c_result = linalg.generic {
        indexing_maps = [affine_map<(d0,d1,d2,d3) -> (d0,d1,d2)>,
          affine_map<(d0,d1,d2,d3) -> (d1,d3)>],
        iterator_types = ["reduction", "parallel", "reduction", "parallel"]
      } ins(%a_tensor : tensor<2x256x64xf16>)
        outs(%init : tensor<256x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    } -> tensor<256x64xf16>

    %c_tile = ktdp.construct_access_tile %c_view[%c0, %c0] {
      access_tile_set = affine_set<(d0,d1):(d0>=0,-d0+255>=0,d1>=0,-d1+63>=0)>,
      access_tile_order = affine_map<(d0,d1)->(d0,d1)>
    } : memref<256x64xf16> -> !ktdp.access_tile<256x64xindex>
    ktdp.store %c_result, %c_tile : tensor<256x64xf16>, !ktdp.access_tile<256x64xindex>

    return
  }
}
