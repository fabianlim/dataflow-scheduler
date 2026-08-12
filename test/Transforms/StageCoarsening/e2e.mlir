// XFAIL: *
// RUN: dataflow-scheduler-opt --tile-scf-for-loops="normalize-loops=true" --canonicalize --loop-invariant-code-motion --stage-coarsening --canonicalize %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (1 ceildiv s0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (64 ceildiv s0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0)[s0, s1] -> ((-(d0 * s1) + s0) ceildiv 64)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 64 + d1 * s0)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK: #[[$ATTR_7:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_7]], memory_space = "DDR"} : memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_7]], memory_space = "DDR"} : memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_6]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_7]], memory_space = "DDR"} : memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<96x64xf16, "DDR"> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<96x64xf16, "DDR"> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<96x64xf16, "DDR"> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[TILING_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[TILING_1:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[TILING_0]]]
// CHECK-NEXT:     %[[MULI_2:.*]] = arith.muli %[[TILING_1]], %[[CONSTANT_2]] : index
// CHECK-NEXT:     %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[MULI_2]]]
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[APPLY_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[APPLY_2:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_0]]){{\[}}%[[TILING_0]]]
// CHECK-NEXT:       %[[ADDI_1:.*]] = arith.addi %[[APPLY_2]], %[[TILING_0]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_1]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[APPLY_3:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_0]]){{\[}}%[[MINSI_0]], %[[TILING_0]]]
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[APPLY_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:         %[[APPLY_4:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_1]]){{\[}}%[[MULI_2]]]
// CHECK-NEXT:         %[[ADDI_2:.*]] = arith.addi %[[APPLY_4]], %[[MULI_2]] : index
// CHECK-NEXT:         %[[MINSI_1:.*]] = arith.minsi %[[ADDI_2]], %[[CONSTANT_2]] : index
// CHECK-NEXT:         %[[APPLY_5:.*]] = affine.apply #[[$ATTR_4]](%[[VAL_1]]){{\[}}%[[MINSI_1]], %[[MULI_2]]]
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:5 = ktdf.private -> (memref<?x?x64xf16, "L1">, memref<?x?x64xf16, "L1">, memref<?x?x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[ALLOC_0:.*]] = memref.alloc(%[[APPLY_3]], %[[APPLY_5]]) : memref<?x?x64xf16, "L1">
// CHECK-NEXT:             %[[ALLOC_1:.*]] = memref.alloc(%[[APPLY_3]], %[[APPLY_5]]) : memref<?x?x64xf16, "L1">
// CHECK-NEXT:             %[[ALLOC_2:.*]] = memref.alloc(%[[APPLY_3]], %[[APPLY_5]]) : memref<?x?x64xf16, "L1">
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[ALLOC_2]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<?x?x64xf16, "L1">, memref<?x?x64xf16, "L1">, memref<?x?x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_2:.*]]#3) {
// CHECK-NEXT:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[APPLY_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               scf.for %[[VAL_4:.*]] = %[[CONSTANT_0]] to %[[APPLY_5]] step %[[CONSTANT_1]] {
// CHECK-NEXT:                 %[[APPLY_6:.*]] = affine.apply #[[$ATTR_5]](%[[VAL_4]], %[[VAL_1]]){{\[}}%[[MULI_2]]]
// CHECK-NEXT:                 %[[APPLY_7:.*]] = affine.apply #[[$ATTR_6]](%[[VAL_3]], %[[VAL_0]]){{\[}}%[[TILING_0]]]
// CHECK-NEXT:                 ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_7]], %[[APPLY_6]]] size [1, 64] to %[[VAL_2]]#0{{\[}}%[[VAL_3]], %[[VAL_4]], %[[CONSTANT_0]]] size [1, 1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<?x?x64xf16, "L1">
// CHECK-NEXT:                 %[[APPLY_8:.*]] = affine.apply #[[$ATTR_5]](%[[VAL_4]], %[[VAL_1]]){{\[}}%[[MULI_2]]]
// CHECK-NEXT:                 %[[APPLY_9:.*]] = affine.apply #[[$ATTR_6]](%[[VAL_3]], %[[VAL_0]]){{\[}}%[[TILING_0]]]
// CHECK-NEXT:                 ktdf.data_transfer from %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_9]], %[[APPLY_8]]] size [1, 64] to %[[VAL_2]]#1{{\[}}%[[VAL_3]], %[[VAL_4]], %[[CONSTANT_0]]] size [1, 1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<?x?x64xf16, "L1">
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_5:.*]]#3) depends_out(%[[VAL_5]]#4) {
// CHECK-NEXT:             scf.for %[[VAL_6:.*]] = %[[CONSTANT_0]] to %[[APPLY_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               scf.for %[[VAL_7:.*]] = %[[CONSTANT_0]] to %[[APPLY_5]] step %[[CONSTANT_1]] {
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[PRIVATE_1:.*]]:5 = ktdf.private -> (memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[ALLOC_3:.*]] = memref.alloc() : memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     %[[ALLOC_4:.*]] = memref.alloc() : memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     %[[ALLOC_5:.*]] = memref.alloc() : memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[ALLOC_3]], %[[ALLOC_4]], %[[ALLOC_5]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_8:.*]]#3) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_5]]#0{{\[}}%[[VAL_6]], %[[VAL_7]], %[[CONSTANT_0]]] size [1, 1, 64] to %[[VAL_8]]#0{{\[}}%[[CONSTANT_0]]] size [64] : memref<?x?x64xf16, "L1">, memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_5]]#1{{\[}}%[[VAL_6]], %[[VAL_7]], %[[CONSTANT_0]]] size [1, 1, 64] to %[[VAL_8]]#1{{\[}}%[[CONSTANT_0]]] size [64] : memref<?x?x64xf16, "L1">, memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(%[[VAL_9]]#4) {
// CHECK-NEXT:                     linalg.add ins(%[[VAL_9]]#0, %[[VAL_9]]#1 : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">) outs(%[[VAL_9]]#2 : memref<64xf16, "SFP_LRFREG">)
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_10:.*]]#4) depends_out(none) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_10]]#2{{\[}}%[[CONSTANT_0]]] size [64] to %[[VAL_5]]#2{{\[}}%[[VAL_6]], %[[VAL_7]], %[[CONSTANT_0]]] size [1, 1, 64] : memref<64xf16, "SFP_LRFREG">, memref<?x?x64xf16, "L1">
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_11:.*]]#4) depends_out(none) {
// CHECK-NEXT:             scf.for %[[VAL_12:.*]] = %[[CONSTANT_0]] to %[[APPLY_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               scf.for %[[VAL_13:.*]] = %[[CONSTANT_0]] to %[[APPLY_5]] step %[[CONSTANT_1]] {
// CHECK-NEXT:                 %[[APPLY_10:.*]] = affine.apply #[[$ATTR_5]](%[[VAL_13]], %[[VAL_1]]){{\[}}%[[MULI_2]]]
// CHECK-NEXT:                 %[[APPLY_11:.*]] = affine.apply #[[$ATTR_6]](%[[VAL_12]], %[[VAL_0]]){{\[}}%[[TILING_0]]]
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_11]]#2{{\[}}%[[VAL_12]], %[[VAL_13]], %[[CONSTANT_0]]] size [1, 1, 64] to %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_11]], %[[APPLY_10]]] size [1, 64] : memref<?x?x64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }







module {
    ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %tile_size = arith.constant 32 : index
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index

        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
        %start_row_i = arith.addi %start_row, %i : index

        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = "DDR"
        } : memref<96x64xf16, "DDR">

        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = "DDR"
        } : memref<96x64xf16, "DDR">

        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = "DDR"
        } : memref<96x64xf16, "DDR">

        %offset = arith.muli %start_row_i, %c64 : index
        %A_ms_cast = memref.memory_space_cast %A_view : memref<96x64xf16, "DDR"> to memref<96x64xf16, "DDR">
        %B_ms_cast = memref.memory_space_cast %B_view : memref<96x64xf16, "DDR"> to memref<96x64xf16, "DDR">
        %C_ms_cast = memref.memory_space_cast %C_view : memref<96x64xf16, "DDR"> to memref<96x64xf16, "DDR">

        %A = memref.reinterpret_cast %A_ms_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        %B = memref.reinterpret_cast %B_ms_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        %C = memref.reinterpret_cast %C_ms_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">

        scf.for %m = %c0 to %c1 step %c1 {
          scf.for %n = %c0 to %c64 step %c64 {
           ktdf.pipeline  {
              %res_l1_A, %res_l1_B, %res_l1_C, %res_sfu_A, %res_sfu_B, %res_sfu_out,
              %res_t1, %res_t2, %res_t3, %res_t4, %res_t5 = ktdf.private -> (
                memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "L1">,
                memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">,
                !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              ) {
                %l1_A    = memref.alloc() : memref<64xf16, "L1">
                %l1_B    = memref.alloc() : memref<64xf16, "L1">
                %l1_C    = memref.alloc() : memref<64xf16, "L1">
                %sfu_A   = memref.alloc() : memref<64xf16, "SFP_LRFREG">
                %sfu_B   = memref.alloc() : memref<64xf16, "SFP_LRFREG">
                %sfu_out = memref.alloc() : memref<64xf16, "SFP_LRFREG">
                %t1 = ktdf.create_token : !ktdf.token
                %t2 = ktdf.create_token : !ktdf.token
                %t3 = ktdf.create_token : !ktdf.token
                %t4 = ktdf.create_token : !ktdf.token
                %t5 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %l1_A, %l1_B, %l1_C, %sfu_A, %sfu_B, %sfu_out,
                                   %t1, %t2, %t3, %t4, %t5
                    : memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "L1">,
                      memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">,
                      !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }

              ktdf.stage depends_in(none) depends_out(%res_t1) {
                ktdf.data_transfer from %A[%m, %n] size [1, 64] to %res_l1_A[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<64xf16, "L1">
                ktdf.data_transfer from %B[%m, %n] size [1, 64] to %res_l1_B[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<64xf16, "L1">
              }

              ktdf.stage depends_in(%res_t1) depends_out(%res_t2) {
                ktdf.data_transfer from %res_l1_A[%c0] size [64] to %res_sfu_A[%c0] size [64] : memref<64xf16, "L1">, memref<64xf16, "SFP_LRFREG">
                ktdf.data_transfer from %res_l1_B[%c0] size [64] to %res_sfu_B[%c0] size [64] : memref<64xf16, "L1">, memref<64xf16, "SFP_LRFREG">
              }

              ktdf.stage depends_in(%res_t2) depends_out(%res_t3) {
                linalg.add ins(%res_sfu_A, %res_sfu_B : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">) outs(%res_sfu_out : memref<64xf16, "SFP_LRFREG">)
              }

              ktdf.stage depends_in(%res_t3) depends_out(%res_t4) {
                ktdf.data_transfer from %res_sfu_out[%c0] size [64] to %res_l1_C[%c0] size [64] : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "L1">
              }

              ktdf.stage depends_in(%res_t4) depends_out(%res_t5) {
                ktdf.data_transfer from %res_l1_C[%c0] size [64] to %C[%m, %n] size [1, 64] : memref<64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
              }
            } // ktdf.pipeline
          } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n parallel
        } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m parallel
        return
    }
}
