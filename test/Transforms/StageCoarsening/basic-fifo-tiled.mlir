// CAUTION: Be careful when updating this test to make sure SSA results from ktdf.private are reusing the appropriate FileCheck matches.

// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_6]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_7]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_3]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:5 = ktdf.private -> (memref<2x1x64xf16, "L1">, memref<2x1x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-NEXT:             %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-NEXT:             %[[ALLOC_2:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[ALLOC_2]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x64xf16, "L1">, memref<2x1x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#3) {
// CHECK-NEXT:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               scf.for %[[VAL_4:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_3]] {
// CHECK-NEXT:                 %[[ADDI_1:.*]] = arith.addi %[[VAL_0]], %[[VAL_3]] : index
// CHECK-NEXT:                 %[[ADDI_2:.*]] = arith.addi %[[VAL_1]], %[[VAL_4]] : index
// CHECK-NEXT:                 %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[SUBI_1:.*]] = arith.subi %[[VAL_4]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[ADDI_1]], %[[ADDI_2]]] size [1, 64] to %[[PRIVATE_0]]#0{{\[}}%[[DIVSI_0]], %[[DIVSI_1]], %[[CONSTANT_0]]] size [1, 1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<2x1x64xf16, "L1">
// CHECK-NEXT:                 %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[SUBI_3:.*]] = arith.subi %[[VAL_4]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[REINTERPRET_CAST_1]]{{\[}}%[[ADDI_1]], %[[ADDI_2]]] size [1, 64] to %[[PRIVATE_0]]#1{{\[}}%[[DIVSI_2]], %[[DIVSI_3]], %[[CONSTANT_0]]] size [1, 1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<2x1x64xf16, "L1">
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_5:.*]]#3) depends_out(%[[VAL_5]]#4) {
// CHECK-NEXT:             scf.for %[[VAL_6:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               scf.for %[[VAL_7:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_3]] {
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[PRIVATE_1:.*]]:8 = ktdf.private -> (memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[ALLOC_3:.*]] = memref.alloc() : memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     %[[ALLOC_4:.*]] = memref.alloc() : memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     %[[ALLOC_5:.*]] = memref.alloc() : memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[ALLOC_3]], %[[ALLOC_4]], %[[ALLOC_5]], %[[FIFO_0]]#0, %[[FIFO_0]]#1, %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_8:.*]]#6) {
// CHECK-NEXT:                     %[[SUBI_4:.*]] = arith.subi %[[VAL_6]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                     %[[DIVSI_4:.*]] = arith.divsi %[[SUBI_4]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                     %[[SUBI_5:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                     %[[DIVSI_5:.*]] = arith.divsi %[[SUBI_5]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[PRIVATE_0]]#0{{\[}}%[[DIVSI_4]], %[[DIVSI_5]], %[[CONSTANT_0]]] size [1, 1, 64] to %[[PRIVATE_1]]#3 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     %[[SUBI_6:.*]] = arith.subi %[[VAL_6]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                     %[[DIVSI_6:.*]] = arith.divsi %[[SUBI_6]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                     %[[SUBI_7:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                     %[[DIVSI_7:.*]] = arith.divsi %[[SUBI_7]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[PRIVATE_0]]#1{{\[}}%[[DIVSI_6]], %[[DIVSI_7]], %[[CONSTANT_0]]] size [1, 1, 64] to %[[PRIVATE_1]]#4 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[PRIVATE_1]]#6) depends_out(%[[PRIVATE_1]]#7) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[PRIVATE_1]]#3 to %[[PRIVATE_1]]#0{{\[}}%[[CONSTANT_0]]] size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     ktdf.data_transfer from %[[PRIVATE_1]]#4 to %[[PRIVATE_1]]#1{{\[}}%[[CONSTANT_0]]] size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, memref<64xf16, "SFP_LRFREG">
// CHECK-NEXT:                     linalg.add ins(%[[PRIVATE_1]]#0, %[[PRIVATE_1]]#1 : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">) outs(%[[PRIVATE_1]]#2 : memref<64xf16, "SFP_LRFREG">)
// CHECK-NEXT:                     ktdf.data_transfer from %[[PRIVATE_1]]#2{{\[}}%[[CONSTANT_0]]] size [64] to %[[PRIVATE_1]]#5 size [64] : memref<64xf16, "SFP_LRFREG">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[PRIVATE_1]]#7) depends_out(none) {
// CHECK-NEXT:                     %[[SUBI_8:.*]] = arith.subi %[[VAL_6]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                     %[[DIVSI_8:.*]] = arith.divsi %[[SUBI_8]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                     %[[SUBI_9:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                     %[[DIVSI_9:.*]] = arith.divsi %[[SUBI_9]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[PRIVATE_1]]#5 size [64] to %[[PRIVATE_0]]#2{{\[}}%[[DIVSI_8]], %[[DIVSI_9]], %[[CONSTANT_0]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0]]#4) depends_out(none) {
// CHECK-NEXT:             scf.for %[[VAL_12:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               scf.for %[[VAL_13:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_3]] {
// CHECK-NEXT:                 %[[ADDI_3:.*]] = arith.addi %[[VAL_0]], %[[VAL_12]] : index
// CHECK-NEXT:                 %[[ADDI_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_13]] : index
// CHECK-NEXT:                 %[[SUBI_10:.*]] = arith.subi %[[VAL_12]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_10:.*]] = arith.divsi %[[SUBI_10]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[SUBI_11:.*]] = arith.subi %[[VAL_13]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_11:.*]] = arith.divsi %[[SUBI_11]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[PRIVATE_0]]#2{{\[}}%[[DIVSI_10]], %[[DIVSI_11]], %[[CONSTANT_0]]] size [1, 1, 64] to %[[REINTERPRET_CAST_2]]{{\[}}%[[ADDI_3]], %[[ADDI_4]]] size [1, 64] : memref<2x1x64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
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
        %c2 = arith.constant 2 : index
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
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>

        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>

        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>

        %offset = arith.muli %start_row_i, %c64 : index
        %A_ms_cast = memref.memory_space_cast %A_view : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
        %B_ms_cast = memref.memory_space_cast %B_view : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
        %C_ms_cast = memref.memory_space_cast %C_view : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">

        %A = memref.reinterpret_cast %A_ms_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        %B = memref.reinterpret_cast %B_ms_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        %C = memref.reinterpret_cast %C_ms_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">

        // Outer tile loops (m0, n0) - tile size 2x64
        scf.for %m0 = %c0 to %c1 step %c2 {
          scf.for %n0 = %c0 to %c64 step %c64 {
            // Inner loops (m1, n1) - execution granularity
            scf.for %m1 = %c0 to %c2 step %c1 {
              scf.for %n1 = %c0 to %c64 step %c64 {
               ktdf.pipeline  {
                  %res_l1_A, %res_l1_B, %res_l1_C, %res_sfu_A, %res_sfu_B, %res_sfu_out,
                  %res_fifo_slot_a, %res_fifo_slot_b, %res_fifo_slot_c,
                  %res_t1, %res_t2, %res_t3, %res_t4, %res_t5 = ktdf.private -> (
                    memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "L1">,
                    memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">,
                    !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                    !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                    !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
                  ) {
                    %l1_A    = memref.alloc() : memref<64xf16, "L1">
                    %l1_B    = memref.alloc() : memref<64xf16, "L1">
                    %l1_C    = memref.alloc() : memref<64xf16, "L1">
                    %sfu_A   = memref.alloc() : memref<64xf16, "SFP_LRFREG">
                    %sfu_B   = memref.alloc() : memref<64xf16, "SFP_LRFREG">
                    %sfu_out = memref.alloc() : memref<64xf16, "SFP_LRFREG">
                    %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                                                                        !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                    %t1 = ktdf.create_token : !ktdf.token
                    %t2 = ktdf.create_token : !ktdf.token
                    %t3 = ktdf.create_token : !ktdf.token
                    %t4 = ktdf.create_token : !ktdf.token
                    %t5 = ktdf.create_token : !ktdf.token
                    ktdf.private_yield %l1_A, %l1_B, %l1_C, %sfu_A, %sfu_B, %sfu_out,
                                       %fifo_slot_a, %fifo_slot_b, %fifo_slot_c,
                                       %t1, %t2, %t3, %t4, %t5
                        : memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "L1">,
                          memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">,
                          !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                          !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                          !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
                  }

                  ktdf.stage depends_in(none) depends_out(%res_t1) {
                    %m_idx = arith.addi %m0, %m1 : index
                    %n_idx = arith.addi %n0, %n1 : index
                    ktdf.data_transfer from %A[%m_idx, %n_idx] size [1, 64] to %res_l1_A[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<64xf16, "L1">
                    ktdf.data_transfer from %B[%m_idx, %n_idx] size [1, 64] to %res_l1_B[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<64xf16, "L1">
                  }

                  ktdf.stage depends_in(%res_t1) depends_out(%res_t2) {
                    ktdf.data_transfer from %res_l1_A[%c0] size [64] to %res_fifo_slot_a size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    ktdf.data_transfer from %res_l1_B[%c0] size [64] to %res_fifo_slot_b size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  }

                  ktdf.stage depends_in(%res_t2) depends_out(%res_t3) {
                    ktdf.data_transfer from %res_fifo_slot_a to %res_sfu_A[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, memref<64xf16, "SFP_LRFREG">
                    ktdf.data_transfer from %res_fifo_slot_b to %res_sfu_B[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, memref<64xf16, "SFP_LRFREG">
                    linalg.add ins(%res_sfu_A, %res_sfu_B : memref<64xf16, "SFP_LRFREG">, memref<64xf16, "SFP_LRFREG">) outs(%res_sfu_out : memref<64xf16, "SFP_LRFREG">)
                    ktdf.data_transfer from %res_sfu_out[%c0] size [64] to %res_fifo_slot_c size [64] : memref<64xf16, "SFP_LRFREG">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  }

                  ktdf.stage depends_in(%res_t3) depends_out(%res_t4) {
                    ktdf.data_transfer from %res_fifo_slot_c size [64] to %res_l1_C[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
                  }

                  ktdf.stage depends_in(%res_t4) depends_out(%res_t5) {
                    %m_idx = arith.addi %m0, %m1 : index
                    %n_idx = arith.addi %n0, %n1 : index
                    ktdf.data_transfer from %res_l1_C[%c0] size [64] to %C[%m_idx, %n_idx] size [1, 64] : memref<64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
                  }
                } // ktdf.pipeline
              } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n1 parallel
            } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m1 parallel
          } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n0 parallel
        } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m0 parallel
        return
    }
}
