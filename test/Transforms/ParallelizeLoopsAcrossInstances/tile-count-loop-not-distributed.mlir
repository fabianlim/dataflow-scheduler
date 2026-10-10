// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// A loop whose upper bound is arith.ceildivui(%total, %reserve_size) counts
// tiles whose size tile-size-selection has yet to choose. That pass finds the
// loop only as an scf.for with this bound, so the loop is never distributed;
// the element loop inside it, bounded by ktdf.tiling.derive_size, still is.

// Nested per-sub-core pipelines, as stage-coarsening leaves a group whose
// stages all run on one sub-core. The outer pipeline has only the tile-count
// loop in scope and is left alone; the inner pipeline distributes its element
// loop. Exactly one ktdf.parallel results.

// Flat case: one pipeline with the element loop and the tile-count loop both
// in scope. The outermost dynamic loop is the tile-count loop, which is
// skipped, so the element loop is distributed.

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   func.func @nested_pipelines() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[TILING_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK:           %[[CEILDIVUI_0:.*]] = arith.ceildivui %[[CONSTANT_2]], %[[TILING_0]] : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CEILDIVUI_0]] step %[[CONSTANT_1]] {
// CHECK:             %[[TILING_1:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_0]] : %[[TILING_0]]], total_size = %[[CONSTANT_2]] : index
// CHECK:             ktdf.pipeline {
// CHECK:               ktdf.stage depends_in(none) depends_out(none) {
// CHECK:                 ktdf.parallel (%[[VAL_1:.*]], %[[VAL_2:.*]]) = (%[[CONSTANT_0]]) to (%[[TILING_1]]) step (%[[CONSTANT_1]]) distribute(num_instances = 2) {
// CHECK:                   %[[TILING_2:.*]] = ktdf.tiling.linearize_index {{\[}}%[[VAL_0]] : %[[TILING_0]]], {{\[}}%[[VAL_1]] : %[[CONSTANT_1]]] : index
// CHECK:                   ktdf.pipeline {
// CHECK:                     ktdf.stage depends_in(none) depends_out(none) {
// CHECK:                       "test.load"(%[[TILING_2]]) : (index) -> ()
// CHECK:                     } {applicable_units = ["L1LU"]}
// CHECK:                     ktdf.stage depends_in(none) depends_out(none) {
// CHECK:                       "test.body"(%[[TILING_2]]) : (index) -> ()
// CHECK:                     } {applicable_units = ["SFU"]}
// CHECK:                     ktdf.stage depends_in(none) depends_out(none) {
// CHECK:                       "test.store"(%[[TILING_2]]) : (index) -> ()
// CHECK:                     } {applicable_units = ["L1SU"]}
// CHECK:                   }
// CHECK:                   ktdf.parallel_yield
// CHECK:                 }
// CHECK:               } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK:             }
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:           return
// CHECK:         }
  func.func @nested_pipelines() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %1 = arith.ceildivui %c4, %0 : index
    scf.for %t = %c0 to %1 step %c1 {
      %2 = ktdf.tiling.derive_size [%t : %0], total_size = %c4 : index
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          scf.for %e = %c0 to %2 step %c1 {
            %3 = ktdf.tiling.linearize_index [%t : %0], [%e : %c1] : index
            ktdf.pipeline {
              ktdf.stage depends_in(none) depends_out(none) {
                "test.load"(%3) : (index) -> ()
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(none) depends_out(none) {
                "test.body"(%3) : (index) -> ()
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(none) depends_out(none) {
                "test.store"(%3) : (index) -> ()
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }

// CHECK-LABEL:   func.func @flat_pipeline() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[TILING_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK:           %[[CEILDIVUI_0:.*]] = arith.ceildivui %[[CONSTANT_2]], %[[TILING_0]] : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CEILDIVUI_0]] step %[[CONSTANT_1]] {
// CHECK:             %[[TILING_1:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_0]] : %[[TILING_0]]], total_size = %[[CONSTANT_2]] : index
// CHECK:             ktdf.parallel (%[[VAL_1:.*]], %[[VAL_2:.*]]) = (%[[CONSTANT_0]]) to (%[[TILING_1]]) step (%[[CONSTANT_1]]) distribute(num_instances = 2) {
// CHECK:               %[[TILING_2:.*]] = ktdf.tiling.linearize_index {{\[}}%[[VAL_0]] : %[[TILING_0]]], {{\[}}%[[VAL_1]] : %[[CONSTANT_1]]] : index
// CHECK:               ktdf.pipeline {
// CHECK:                 ktdf.stage depends_in(none) depends_out(none) {
// CHECK:                   "test.body"(%[[TILING_2]]) : (index) -> ()
// CHECK:                 } {applicable_units = ["SFU"]}
// CHECK:               }
// CHECK:               ktdf.parallel_yield
// CHECK:             }
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:           return
// CHECK:         }
  func.func @flat_pipeline() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %1 = arith.ceildivui %c4, %0 : index
    scf.for %t = %c0 to %1 step %c1 {
      %2 = ktdf.tiling.derive_size [%t : %0], total_size = %c4 : index
      scf.for %e = %c0 to %2 step %c1 {
        %3 = ktdf.tiling.linearize_index [%t : %0], [%e : %c1] : index
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            "test.body"(%3) : (index) -> ()
          } {applicable_units = ["SFU"]}
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
