// RUN: dataflow-scheduler-opt --split-input-file --path-expansion %s | FileCheck %s

// The load stage reads from an L1 view straight into the compute stage: the
// route is L1 -> L1LU -> SFU, so no intermediate stage is inserted on the load
// side. The load stage still gets its unit (L1LU), a FIFO naming that unit
// ("L1LU" -> "SFU", shared with the compute stage's read_from_fifo) and the
// flat slot size on its FIFO side. The store side goes through L1 to DDR.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @l1_endpoint_load(
// CHECK-SAME:      %[[ARG0:[^:]*]]: memref<96x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:[^:]*]]: memref<96x64xf16, "L1">,
// CHECK-SAME:      %[[ARG2:[^:]*]]: memref<96x64xf16, "DDR">) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:         %[[CONSTANT_3:.*]] = arith.constant 96 : index
// CHECK-NEXT:         scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] {
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             %[[PRIVATE_0:.*]]:7 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:               %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:               %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:               %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf.private_yield %[[FIFO_0]]#0, %[[FIFO_0]]#1, %[[FIFO_1]], %[[ALLOC_0]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#4) {
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_0]], 0] size [1, 64] to %[[PRIVATE_0]]#0 size [64] : memref<96x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG1]]{{\[}}%[[VAL_0]], 0] size [1, 64] to %[[PRIVATE_0]]#1 size [64] : memref<96x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["L1LU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#4) depends_out(%[[PRIVATE_0]]#5) {
// CHECK-NEXT:               %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:               %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:               %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:               %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]] : tensor<64xf16>, tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:               ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:                 %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK-NEXT:                 linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:               } -> tensor<64xf16>
// CHECK-NEXT:               ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_0]]#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["SFU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#5) depends_out(%[[PRIVATE_0]]#6) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_0]]#2 size [64] to %[[PRIVATE_0]]#3[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["L1SU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#6) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_0]]#3[0, 0] size [1, 64] to %[[ARG2]]{{\[}}%[[VAL_0]], 0] size [1, 64] : memref<1x64xf16, "L1">, memref<96x64xf16, "DDR">
// CHECK-NEXT:             } {applicable_units = ["MNISU"]}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         return
// CHECK-NEXT:       }


#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @l1_endpoint_load(%x: memref<96x64xf16, "L1">, %y: memref<96x64xf16, "L1">, %out: memref<96x64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c96 = arith.constant 96 : index
    scf.for %arg0 = %c0 to %c96 step %c1 {
      ktdf.pipeline {
        %0:6 = ktdf.private -> (!ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
          %1:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
          %2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          %5 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %1#0, %1#1, %2, %3, %4, %5 : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%0#3) {
          ktdf.data_transfer from %x[%arg0, %c0] size [1, 64] to %0#0 size [1, 64] : memref<96x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
          ktdf.data_transfer from %y[%arg0, %c0] size [1, 64] to %0#1 size [1, 64] : memref<96x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
        }
        ktdf.stage depends_in(%0#3) depends_out(%0#4) {
          %1 = ktdf.read_from_fifo %0#0 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
          %2 = ktdf.read_from_fifo %0#1 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
          %3 = tensor.empty() : tensor<64xf16>
          %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %2 : tensor<64xf16>, tensor<64xf16>) outs(%3 : tensor<64xf16>) {
          ^bb0(%in: f16, %in_0: f16, %o: f16):
            %5 = arith.addf %in, %in_0 : f16
            linalg.yield %5 : f16
          } -> tensor<64xf16>
          ktdf.write_to_fifo %4, %0#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%0#4) depends_out(%0#5) {
          ktdf.data_transfer from %0#2 size [64] to %out[%arg0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<96x64xf16, "DDR">
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// The store stage writes from the compute stage straight into an L1 view: the
// route is SFU -> L1SU -> L1, so no intermediate stage is inserted on the
// store side. The store stage still gets its unit (L1SU), a FIFO naming that
// unit ("SFU" -> "L1SU", shared with the compute stage's write_to_fifo) and
// the flat slot size on its FIFO side. The load side comes from DDR through L1.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @l1_endpoint_store(
// CHECK-SAME:      %[[ARG0:[^:]*]]: memref<96x64xf16, "DDR">,
// CHECK-SAME:      %[[ARG1:[^:]*]]: memref<96x64xf16, "L1">) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:         %[[CONSTANT_3:.*]] = arith.constant 96 : index
// CHECK-NEXT:         scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] {
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             %[[PRIVATE_0:.*]]:6 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:               %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:               %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:               %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf.private_yield %[[FIFO_0]], %[[ALLOC_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#3) {
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_0]], 0] size [1, 64] to %[[PRIVATE_0]]#1[0, 0] size [1, 64] : memref<96x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["MNILU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#3) depends_out(%[[PRIVATE_0]]#4) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_0]]#1[0, 0] size [1, 64] to %[[PRIVATE_0]]#2 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["L1LU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#4) depends_out(%[[PRIVATE_0]]#5) {
// CHECK-NEXT:               %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#2 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:               %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:               %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:               ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:                 %[[NEGF_0:.*]] = arith.negf %[[VAL_4]] : f16
// CHECK-NEXT:                 linalg.yield %[[NEGF_0]] : f16
// CHECK-NEXT:               } -> tensor<64xf16>
// CHECK-NEXT:               ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_0]]#0 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["SFU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#5) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_0]]#0 size [64] to %[[ARG1]]{{\[}}%[[VAL_0]], 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<96x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["L1SU"]}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         return
// CHECK-NEXT:       }


#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @l1_endpoint_store(%x: memref<96x64xf16, "DDR">, %out: memref<96x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c96 = arith.constant 96 : index
    scf.for %arg0 = %c0 to %c96 step %c1 {
      ktdf.pipeline {
        %0:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
          %1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
          %2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          %5 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %1, %2, %3, %4, %5 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%0#2) {
          ktdf.data_transfer from %x[%arg0, %c0] size [1, 64] to %0#0 size [64] : memref<96x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        }
        ktdf.stage depends_in(%0#2) depends_out(%0#3) {
          %1 = ktdf.read_from_fifo %0#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
          %2 = tensor.empty() : tensor<64xf16>
          %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<64xf16>) outs(%2 : tensor<64xf16>) {
          ^bb0(%in: f16, %o: f16):
            %4 = arith.negf %in : f16
            linalg.yield %4 : f16
          } -> tensor<64xf16>
          ktdf.write_to_fifo %3, %0#1 : tensor<64xf16>, <"SFU" -> "L1", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%0#3) depends_out(%0#4) {
          ktdf.data_transfer from %0#1 size [1, 64] to %out[%arg0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<96x64xf16, "L1">
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// L1 -> SFU -> L1: the route (L1 -> L1LU -> SFU -> L1SU -> L1) already matches
// the stages, so no stage is inserted, but both transfer stages name the
// memory on their FIFOs. Each still gets its unit, a FIFO naming that unit and
// the flat slot size on its FIFO side.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @l1_only(
// CHECK-SAME:      %[[ARG0:[^:]*]]: memref<96x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:[^:]*]]: memref<96x64xf16, "L1">) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:         %[[CONSTANT_2:.*]] = arith.constant 96 : index
// CHECK-NEXT:         scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             %[[PRIVATE_0:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:               %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:               %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#2) {
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_0]], 0] size [1, 64] to %[[PRIVATE_0]]#0 size [64] : memref<96x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["L1LU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#2) depends_out(%[[PRIVATE_0]]#3) {
// CHECK-NEXT:               %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:               %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:               %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:               ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK-NEXT:                 %[[NEGF_0:.*]] = arith.negf %[[VAL_3]] : f16
// CHECK-NEXT:                 linalg.yield %[[NEGF_0]] : f16
// CHECK-NEXT:               } -> tensor<64xf16>
// CHECK-NEXT:               ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_0]]#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             } {applicable_units = ["SFU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIVATE_0]]#3) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIVATE_0]]#1 size [64] to %[[ARG1]]{{\[}}%[[VAL_0]], 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<96x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["L1SU"]}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         return
// CHECK-NEXT:       }


#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @l1_only(%x: memref<96x64xf16, "L1">, %out: memref<96x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c96 = arith.constant 96 : index
    scf.for %arg0 = %c0 to %c96 step %c1 {
      ktdf.pipeline {
        %0:5 = ktdf.private -> (!ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
          %1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
          %2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          %5 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %1, %2, %3, %4, %5 : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%0#2) {
          ktdf.data_transfer from %x[%arg0, %c0] size [1, 64] to %0#0 size [1, 64] : memref<96x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
        }
        ktdf.stage depends_in(%0#2) depends_out(%0#3) {
          %1 = ktdf.read_from_fifo %0#0 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
          %2 = tensor.empty() : tensor<64xf16>
          %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<64xf16>) outs(%2 : tensor<64xf16>) {
          ^bb0(%in: f16, %o: f16):
            %4 = arith.negf %in : f16
            linalg.yield %4 : f16
          } -> tensor<64xf16>
          ktdf.write_to_fifo %3, %0#1 : tensor<64xf16>, <"SFU" -> "L1", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%0#3) depends_out(%0#4) {
          ktdf.data_transfer from %0#1 size [1, 64] to %out[%arg0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<96x64xf16, "L1">
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
