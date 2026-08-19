// RUN: dataflow-scheduler-opt --reduction-lane-collapse %s | FileCheck %s

// Two local schedules in the shape the structural passes leave a reduction
// pipeline in, differing only in which loop dimension is reduced.
//
//   @local_schedule_0 -- on-stick. iterator_types = ["reduction", "parallel",
//     "reduction"]: innermost d2 (extent 64, the stride-1 run occupying one SIMD
//     register) and d0 (extent 2, the stick-group axis) are both reduced. Only
//     d2 is rewritten; d0 is a sequence of whole registers.
//
//   @local_schedule_1 -- non-stick. iterator_types = ["parallel", "reduction",
//     "parallel"]: innermost dim is parallel, so nothing is rewritten.
//
// Why the rewrite is needed: the pass that materialises a reduction dimension as
// a loop multiplies every reduction extent into one factor and divides the input
// FIFO by it. With both dims in the product the factor is 2 * 64 = 128, the
// 128-element FIFO divides evenly, and the result is a 1-element FIFO driving 128
// scalar iterations. With d2 taken out first the factor is 2 and the FIFO is 64.
//
// Three things about the rewritten shape:
//
//   * The generic's `outs` is widened, tensor<1xf16> -> tensor<1x64xf16>, with d2
//     appended to the output map. Making d2 parallel without widening would leave
//     a parallel dimension absent from the output map (ill-formed) and discard
//     the lanes the collapse consumes.
//
//   * The widening stops at the opaque, which hands back tensor<1xf16> -- the
//     shape the original reduction produced -- so the store path behind it sees
//     the element count it was already sized for.
//
//   * The opaque's `outs` is a tensor of its *result* type: the op is
//     destination-passing, so neither a buffer there nor an `outs` of the input
//     type type-checks. A scratchpad rides as an extra input instead.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_onstick_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:     func.func @sum_nonstick_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_1() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_1()
// CHECK:     func.func @sum_onstick_4d_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_2() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_2()
// CHECK:   }
// CHECK:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 256 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [256], strides: [64] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<256xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<256xf16> to memref<256xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [256], strides: [1] : memref<256xf16, "DDR"> to memref<256xf16, strided<[1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<256xf16, strided<[1]>, "DDR"> to memref<256xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<256x2x1x64xf16, "L1">, memref<256x1xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<256x2x1x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<256x1xf16, "L1">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<256x2x1x64xf16, "L1">, memref<256x1xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]][0, %[[VAL_1]], 0] size [2, 1, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<256x2x1x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:                 %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_4:.*]]#2) {
// CHECK-NEXT:                 %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], 0, 0, 0] size [1, 2, 1, 64] to %[[VAL_4]]#0 size [128] : memref<256x2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:               } {applicable_units = ["L1LU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_5:.*]]#2) depends_out(%[[VAL_5]]#3) {
// CHECK-NEXT:                 %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_5]]#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
// CHECK-NEXT:                 %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                 %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<2x1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                 ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:                   %[[ADDF_0:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:                   linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                 } -> tensor<1x64xf16>
// CHECK-NEXT:                 %[[EMPTY_1:.*]] = tensor.empty() : tensor<1xf16>
// CHECK-NEXT:                 %[[OPAQUE_0:.*]] = ktdf.opaque "LANE_REDUCE" -> tensor<1xf16>
// CHECK-NEXT:                   ins(%[[GENERIC_0]]: tensor<1x64xf16>)
// CHECK-NEXT:                   outs(%[[EMPTY_1]]: tensor<1xf16>)
// CHECK-NEXT:                 ktdf.write_to_fifo %[[OPAQUE_0]], %[[VAL_5]]#1 : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
// CHECK-NEXT:               } {applicable_units = ["SFU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_8:.*]]#3) depends_out(none) {
// CHECK-NEXT:                 %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_8]]#1 size [1] to %[[VAL_2]]#1{{\[}}%[[DIVSI_2]], 0] size [1, 1] : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<256x1xf16, "L1">
// CHECK-NEXT:               } {applicable_units = ["L1SU"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_10:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[VAL_10]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_9]]#1{{\[}}%[[DIVSI_3]], 0] size [1, 1] to %[[CAST_1]]{{\[}}%[[VAL_10]]] size [1] : memref<256x1xf16, "L1">, memref<256xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-LABEL:   module @local_schedule_1 {
// CHECK-NEXT:     func.func @local_schedule_1() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], 0, %[[CONSTANT_0]] * 64] size [1, 256, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
// CHECK-NEXT:                 %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_4:.*]]#2) {
// CHECK-NEXT:                 %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], 0, 0, 0] size [1, 1, 256, 64] to %[[VAL_4]]#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
// CHECK-NEXT:               } {applicable_units = ["L1LU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_5:.*]]#2) depends_out(%[[VAL_5]]#3) {
// CHECK-NEXT:                 %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_5]]#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
// CHECK-NEXT:                 %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                 %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_2]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x256x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                 ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:                   %[[ADDF_0:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:                   linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                 } -> tensor<1x64xf16>
// CHECK-NEXT:                 ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_5]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               } {applicable_units = ["SFU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_8:.*]]#3) depends_out(none) {
// CHECK-NEXT:                 %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_8]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_2]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:               } {applicable_units = ["L1SU"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_10:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[VAL_10]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_9]]#1{{\[}}%[[DIVSI_3]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_10]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-LABEL:   module @local_schedule_2 {
// CHECK-NEXT:     func.func @local_schedule_2() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 256 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [256, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_6]], memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<256x64xf16> to memref<256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, "DDR"> to memref<256x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<256x64xf16, strided<[64, 1]>, "DDR"> to memref<256x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<256x2x1x64xf16, "L1">, memref<256x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<256x2x1x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<256x1x64xf16, "L1">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<256x2x1x64xf16, "L1">, memref<256x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]][0, %[[VAL_1]], 0] size [2, 1, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<256x2x1x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:                 %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_4:.*]]#2) {
// CHECK-NEXT:                 %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], 0, 0, 0] size [1, 2, 1, 64] to %[[VAL_4]]#0 size [128] : memref<256x2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:               } {applicable_units = ["L1LU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_5:.*]]#2) depends_out(%[[VAL_5]]#3) {
// CHECK-NEXT:                 %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_5]]#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
// CHECK-NEXT:                 %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                 %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<2x1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                 ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:                   %[[ADDF_0:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:                   linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                 } -> tensor<1x64xf16>
// CHECK-NEXT:                 %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                 %[[OPAQUE_0:.*]] = ktdf.opaque "LANE_REDUCE" -> tensor<1x64xf16>
// CHECK-NEXT:                   ins(%[[GENERIC_0]]: tensor<1x64xf16>)
// CHECK-NEXT:                   outs(%[[EMPTY_1]]: tensor<1x64xf16>)
// CHECK-NEXT:                 ktdf.write_to_fifo %[[OPAQUE_0]], %[[VAL_5]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               } {applicable_units = ["SFU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_8:.*]]#3) depends_out(none) {
// CHECK-NEXT:                 %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_8]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_2]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, "L1">
// CHECK-NEXT:               } {applicable_units = ["L1SU"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_10:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[VAL_10]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_9]]#1{{\[}}%[[DIVSI_3]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_10]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<256x1x64xf16, "L1">, memref<256x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @sum_onstick_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
    func.func @sum_nonstick_1core() attributes {grid = [1]} {
      call @local_schedule_1() : () -> ()
      return
    }
    func.func private @local_schedule_1()
    func.func @sum_onstick_4d_1core() attributes {grid = [1]} {
      call @local_schedule_2() : () -> ()
      return
    }
    func.func private @local_schedule_2()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

  // On-stick: the innermost dim is reduced. Rewritten.
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c256 = arith.constant 256 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [256], strides: [64] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<256xf16> to memref<256xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [256], strides: [1] : memref<256xf16, "DDR"> to memref<256xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<256xf16, strided<[1]>, "DDR"> to memref<256xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<256x2x1x64xf16, "L1">, memref<256x1xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<256x2x1x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<256x1xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<256x2x1x64xf16, "L1">, memref<256x1xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[0, %arg0, 0] size [2, 1, 64] to %2#0[%4, 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<256x2x1x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %2#0[%5, 0, 0, 0] size [1, 2, 1, 64] to %3#0 size [128] : memref<256x2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
                %5 = tensor.empty() : tensor<1xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "reduction"]} ins(%4 : tensor<2x1x64xf16>) outs(%5 : tensor<1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %3#1 size [1] to %2#1[%5, 0] size [1, 1] : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<256x1xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0] size [1, 1] to %cast_2[%arg0] size [1] : memref<256x1xf16, "L1">, memref<256xf16, strided<[1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }

  // Non-stick: the innermost dim is parallel. Left untouched.
  module @local_schedule_1 {
    func.func @local_schedule_1() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x1x256x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<2x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[%arg0, 0, %c0 * 64] size [1, 256, 64] to %2#0[%4, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %2#0[%5, 0, 0, 0] size [1, 1, 256, 64] to %3#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
                %5 = tensor.empty() : tensor<1x64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4 : tensor<1x256x64xf16>) outs(%5 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }

  // On-stick with an output lane axis: the innermost dim is parallel and is the
  // output's fastest axis, so the reduced lane axis is d2, one in from the end.
  module @local_schedule_2 {
    func.func @local_schedule_2() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c256 = arith.constant 256 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 64], strides: [64, 1] {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<256x64xf16> to memref<256x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, "DDR"> to memref<256x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<256x64xf16, strided<[64, 1]>, "DDR"> to memref<256x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<256x2x1x64xf16, "L1">, memref<256x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<256x2x1x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<256x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<256x2x1x64xf16, "L1">, memref<256x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[0, %arg0, 0] size [2, 1, 64] to %2#0[%4, 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<256x2x1x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %2#0[%5, 0, 0, 0] size [1, 2, 1, 64] to %3#0 size [128] : memref<256x2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
                %5 = tensor.empty() : tensor<1x64xf16>
                %6 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%4 : tensor<2x1x64xf16>) outs(%5 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<256x1x64xf16, "L1">, memref<256x64xf16, strided<[64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
