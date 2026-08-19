// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// On-stick (lane-axis) reduction: the innermost, stride-1 SIMD lane axis reduced
// within one vector register. One compute program unit, no loop, no
// inter-compute-unit FIFO. `sample_device`'s single SFU compute unit matches the
// production device, which declares SFP as its only compute unit.
//
// Both result widths share one lowering and reduce at the *input* width,
// differing only in the closing shuffle's repetition: @onstick_lane_reduce ->
// tensor<1x64xf16>, repetition 64; @onstick_lane_reduce_narrow -> tensor<1xf16>,
// repetition 1.
//
// An arithmetic operand must be something an address generator can name, which
// another arithmetic op's result is not. So every stage runs memory-to-memory
// through the memref `ins` buffer, rewritten in place; that store/load pair is
// also what orders the stages. (`ins` and not `outs` because ktdf.opaque is
// destination-passing: an `outs` would have to be a tensor of the result type.)
//
// The butterfly reuses two permutations, both at repetition = 8:
//   p1 = [2, 3, 0, 1, 6, 7, 6, 7]      p2 = [4, 5, 3, 3, 5, 5, 6, 7]
//   v1 = shuffle(v, p1) + v; v2 = v1 + shuffle(v1, p2);
//   v3 = shuffle(v2, p1) + shuffle(v2, p2)
// leaving each group of 8 lanes reduced into its lane 0 and junk in the other 7;
// only lane 0 is read again, which is why p1 and p2 need not be permutations.
//
// OPERAND PAIRING -- the two rules this op stream exists to lock in. The backend
// does not lower these shuffles into ops of their own: it skips them
// (`isShuffleNFWDVersion0` / `isShuffleNFWDVersion2`) and fuses each into its
// consumer as an NFWD per-operand modifier. A shuffle it cannot place is dropped
// SILENTLY -- the consumer reads the unpermuted value, no diagnostic -- and the
// reduction returns a wrong sum whose total is still plausible. So:
//   1. p1 fuses only onto operand 0, p2 only onto operand 1. Hence
//      `shuffle(v, p1) + v` in stage 1 and not `v + shuffle(v, p1)`: the latter
//      puts p1 on operand 1, where it is dropped and the stage degenerates to
//      v + v. Do NOT normalise these to a uniform order.
//   2. Each operand needs its own load, so a stage loads the same buffer twice
//      for what is the same value -- not redundancy to clean up. Two operands
//      sharing one load leave the second shuffle with no operand to fuse onto.
// Verified on device with a per-lane probe. An all-ones or sum-only check cannot
// catch a violation: any three-stage tree returns the correct total.
//
// The adds are `vectorchain.binary {binary_op = add}` with an identity
// op_specific_map: all lane movement lives in the shuffles. Then one shuffle
// broadcasts each group's lane 0 across its group -- its own stage, since
// `scan_with_gap` has no operand slot to fold a permutation into -- and
// `scan_with_gap {gap = 8, eval_order = left_to_right}` combines the groups,
// leaving the total at lane 0. That eval_order does not match the op's own
// prefix-scan description; do not "correct" it from that text.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 == 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   func.func @onstick_lane_reduce() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:     %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x64xf16, "L1">
// CHECK-NEXT:       %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_3]]} : memref<1x1x64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_0]], %[[VECTOR_LOAD_0]] : vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_1]] : vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_1:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_0:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_1]]) {indices = [2 : i32, 3 : i32, 0 : i32, 1 : i32, 6 : i32, 7 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_2:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_0:.*]] = vectorchain.binary %[[SHUFFLE_0]], %[[VECTOR_LOAD_2]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_3:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_1:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_3]]) {indices = [4 : i32, 5 : i32, 3 : i32, 3 : i32, 5 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_4:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_1:.*]] = vectorchain.binary %[[VECTOR_LOAD_4]], %[[SHUFFLE_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_1]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_5:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_2:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_5]]) {indices = [2 : i32, 3 : i32, 0 : i32, 1 : i32, 6 : i32, 7 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_6:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_3:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_6]]) {indices = [4 : i32, 5 : i32, 3 : i32, 3 : i32, 5 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_2:.*]] = vectorchain.binary %[[SHUFFLE_2]], %[[SHUFFLE_3]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_2]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_7:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_4:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_7]]) {indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SHUFFLE_4]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_8:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SCAN_WITH_GAP_0:.*]] = vectorchain.scan_with_gap %[[VECTOR_LOAD_8]] {eval_order = #vectorchain<eval_order left_to_right>, gap = 8 : index, reduction_op = #vectorchain<binary_operator add>} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SCAN_WITH_GAP_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_9:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_5:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_9]]) {indices = [0 : i32], repetition = 64 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_2]], %[[SHUFFLE_5]] : vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:       %[[ALLOC_2:.*]] = memref.alloc() : memref<1x1x64xf16, "L1">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_2]]) : index
// CHECK-NEXT:       %[[RECEIVE_1:.*]] = dataflow.receive %[[QUERY_MAP_3]] : vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_1]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_0]], store_set = #[[$ATTR_3]]} : memref<1x1x64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-LABEL:   func.func @onstick_lane_reduce_narrow() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:     %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x64xf16, "L1">
// CHECK-NEXT:       %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_3]]} : memref<1x1x64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_0]], %[[VECTOR_LOAD_0]] : vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_1]] : vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_1:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_0:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_1]]) {indices = [2 : i32, 3 : i32, 0 : i32, 1 : i32, 6 : i32, 7 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_2:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_0:.*]] = vectorchain.binary %[[SHUFFLE_0]], %[[VECTOR_LOAD_2]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_3:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_1:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_3]]) {indices = [4 : i32, 5 : i32, 3 : i32, 3 : i32, 5 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_4:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_1:.*]] = vectorchain.binary %[[VECTOR_LOAD_4]], %[[SHUFFLE_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_1]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_5:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_2:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_5]]) {indices = [2 : i32, 3 : i32, 0 : i32, 1 : i32, 6 : i32, 7 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_6:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_3:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_6]]) {indices = [4 : i32, 5 : i32, 3 : i32, 3 : i32, 5 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_2:.*]] = vectorchain.binary %[[SHUFFLE_2]], %[[SHUFFLE_3]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_2]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_7:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_4:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_7]]) {indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SHUFFLE_4]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_8:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SCAN_WITH_GAP_0:.*]] = vectorchain.scan_with_gap %[[VECTOR_LOAD_8]] {eval_order = #vectorchain<eval_order left_to_right>, gap = 8 : index, reduction_op = #vectorchain<binary_operator add>} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SCAN_WITH_GAP_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_9:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_5:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_9]]) {indices = [0 : i32], repetition = 1 : i32} : vector<64xf16>, vector<1xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_2]], %[[SHUFFLE_5]] : vector<1xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:       %[[ALLOC_2:.*]] = memref.alloc() : memref<1x1x1xf16, "L1">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_2]]) : index
// CHECK-NEXT:       %[[RECEIVE_1:.*]] = dataflow.receive %[[QUERY_MAP_3]] : vector<1xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_1]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_0]], store_set = #[[$ATTR_5]]} : memref<1x1x1xf16, "L1">, vector<1xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-LABEL:   func.func @onstick_lane_reduce_from_accumulator() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:       %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_0:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_0]]) {indices = [2 : i32, 3 : i32, 0 : i32, 1 : i32, 6 : i32, 7 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_1:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_0:.*]] = vectorchain.binary %[[SHUFFLE_0]], %[[VECTOR_LOAD_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_0]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_2:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_1:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_2]]) {indices = [4 : i32, 5 : i32, 3 : i32, 3 : i32, 5 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_3:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_1:.*]] = vectorchain.binary %[[VECTOR_LOAD_3]], %[[SHUFFLE_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_1]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_4:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_2:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_4]]) {indices = [2 : i32, 3 : i32, 0 : i32, 1 : i32, 6 : i32, 7 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_5:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_3:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_5]]) {indices = [4 : i32, 5 : i32, 3 : i32, 3 : i32, 5 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[BINARY_2:.*]] = vectorchain.binary %[[SHUFFLE_2]], %[[SHUFFLE_3]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[BINARY_2]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_6:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_4:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_6]]) {indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SHUFFLE_4]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_7:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SCAN_WITH_GAP_0:.*]] = vectorchain.scan_with_gap %[[VECTOR_LOAD_7]] {eval_order = #vectorchain<eval_order left_to_right>, gap = 8 : index, reduction_op = #vectorchain<binary_operator add>} : vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SCAN_WITH_GAP_0]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[VECTOR_LOAD_8:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       %[[SHUFFLE_5:.*]] = vectorchain.shuffle input(%[[VECTOR_LOAD_8]]) {indices = [0 : i32], repetition = 1 : i32} : vector<64xf16>, vector<1xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_0]], %[[SHUFFLE_5]] : vector<1xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<1x1x1xf16, "L1">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_1]] : vector<1xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_0]], store_set = #[[$ATTR_5]]} : memref<1x1x1xf16, "L1">, vector<1xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @onstick_lane_reduce() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %l1su0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %l1su1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index

    %tile_id = ktdp.get_compute_tile_id : index
    %m_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu = uniform.query_map(map:%m_l1lu, key:%tile_id) : index
    %m_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0], [%c1 -> %sfu1]) : index
    %u_sfu  = uniform.query_map(map:%m_sfu, key:%tile_id) : index
    %m_l1su = uniform.def_immutable_mapping([%c0 -> %l1su0], [%c1 -> %l1su1]) : index
    %u_l1su = uniform.query_map(map:%m_l1su, key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x1x64xf16, "L1">
    %out_l1   = memref.alloc() : memref<1x1x64xf16, "L1">

    %fifo_in  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %fifo_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>

    // Load: one 64-lane stick (one row of a[2,256,64]) into the compute unit.
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0, %c0] size [1, 1, 64]
                         to %fifo_in size [64]
        : memref<1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }

    // Compute: collapse the 64 lanes of one stick into lane 0.
    ktdf_lowering.execute_on %u_sfu {
      %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %v = ktdf.read_from_fifo %fifo_in
        : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
      %e = tensor.empty() : tensor<1x64xf16>
      %r = ktdf.opaque "LANE_REDUCE" -> tensor<1x64xf16>
        ins(%v, %acc : tensor<1x64xf16>, memref<1x64xf16, "SFU_REG">)
        outs(%e : tensor<1x64xf16>)
      ktdf.write_to_fifo %r, %fifo_out
        : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
    }

    // Store: full 64-lane stick back to L1. Only lane 0 is logically live.
    ktdf_lowering.execute_on %u_l1su {
      ktdf.data_transfer from %fifo_out size [64]
                         to %out_l1[%c0, %c0, %c0] size [1, 1, 64]
        : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x64xf16, "L1">
    }
    return
  }

  func.func @onstick_lane_reduce_narrow() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %l1su0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %l1su1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index

    %tile_id = ktdp.get_compute_tile_id : index
    %m_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu = uniform.query_map(map:%m_l1lu, key:%tile_id) : index
    %m_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0], [%c1 -> %sfu1]) : index
    %u_sfu  = uniform.query_map(map:%m_sfu, key:%tile_id) : index
    %m_l1su = uniform.def_immutable_mapping([%c0 -> %l1su0], [%c1 -> %l1su1]) : index
    %u_l1su = uniform.query_map(map:%m_l1su, key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x1x64xf16, "L1">
    %out_l1   = memref.alloc() : memref<1x1x1xf16, "L1">

    %fifo_in  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %fifo_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>

    // Load: one 64-lane stick (one row of a[2,256,64]) into the compute unit.
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0, %c0] size [1, 1, 64]
                         to %fifo_in size [64]
        : memref<1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }

    // Compute: collapse the 64 lanes of one stick into lane 0.
    ktdf_lowering.execute_on %u_sfu {
      %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %v = ktdf.read_from_fifo %fifo_in
        : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
      %e = tensor.empty() : tensor<1xf16>
      %r = ktdf.opaque "LANE_REDUCE" -> tensor<1xf16>
        ins(%v, %acc : tensor<1x64xf16>, memref<1x64xf16, "SFU_REG">)
        outs(%e : tensor<1xf16>)
      ktdf.write_to_fifo %r, %fifo_out
        : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
    }

    // Store: just the reduced element.
    ktdf_lowering.execute_on %u_l1su {
      ktdf.data_transfer from %fifo_out size [1]
                         to %out_l1[%c0, %c0, %c0] size [1, 1, 1]
        : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<1x1x1xf16, "L1">
    }
    return
  }

  // Third form: the value to collapse is already sitting in a buffer in the
  // compute unit's local memory rather than in a register. That is what a
  // preceding reduction leaves behind -- it accumulates its running total into
  // such a buffer -- so the lowering must read it out before combining lanes.
  // No input FIFO here: isolating this case means the buffer is the only
  // producer, so nothing else can supply the register.
  func.func @onstick_lane_reduce_from_accumulator() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %l1su0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %l1su1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index

    %tile_id = ktdp.get_compute_tile_id : index
    %m_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0], [%c1 -> %sfu1]) : index
    %u_sfu  = uniform.query_map(map:%m_sfu, key:%tile_id) : index
    %m_l1su = uniform.def_immutable_mapping([%c0 -> %l1su0], [%c1 -> %l1su1]) : index
    %u_l1su = uniform.query_map(map:%m_l1su, key:%tile_id) : index

    %out_l1 = memref.alloc() : memref<1x1x1xf16, "L1">

    %fifo_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>

    ktdf_lowering.execute_on %u_sfu {
      %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %e = tensor.empty() : tensor<1xf16>
      %r = ktdf.opaque "LANE_REDUCE" -> tensor<1xf16>
        ins(%acc : memref<1x64xf16, "SFU_REG">)
        outs(%e : tensor<1xf16>)
      ktdf.write_to_fifo %r, %fifo_out
        : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
    }

    // Store: just the reduced element.
    ktdf_lowering.execute_on %u_l1su {
      ktdf.data_transfer from %fifo_out size [1]
                         to %out_l1[%c0, %c0, %c0] size [1, 1, 1]
        : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<1x1x1xf16, "L1">
    }
    return
  }
}
