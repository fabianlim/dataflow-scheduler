// RUN: dataflow-scheduler-opt --compute-group-extraction %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK:   module {
// CHECK:     func.func @row_reduction() attributes {grid = [1]} {
// CHECK:       call @"local-schedule-0"() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @"local-schedule-0"()
// CHECK:   }

// CHECK:   module {
// CHECK:     func.func private @"local-schedule-0"() attributes {grid = [1]} {
// CHECK:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK:       %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:       %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK:       %[[CONSTANT_4:.*]] = arith.constant 256 : index
// CHECK:       %[[CONSTANT_5:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf16>
// CHECK:       %[[CONSTANT_6:.*]] = arith.constant 8589934592 : index
// CHECK:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_6]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK:       scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] {
// CHECK:         %[[FOR_0:.*]] = scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_5]]) -> (tensor<64xf16>) {
// CHECK:           %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[VAL_0]], %[[VAL_1]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_5]]} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
// CHECK:           %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <1x1x64xindex> -> tensor<1x1x64xf16>
// CHECK:           %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[LOAD_0]][0, 0, 0] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf16> to tensor<64xf16>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[EXTRACT_SLICE_0]] : tensor<64xf16>) outs(%[[VAL_2]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[ADDF_0]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           scf.yield %[[GENERIC_0]] : tensor<64xf16>
// CHECK:         }
// CHECK:         %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_2]], access_tile_set = #[[$ATTR_6]]} : memref<2x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:         %[[INSERT_SLICE_0:.*]] = tensor.insert_slice %[[FOR_0]] into %[[EMPTY_0]][0, 0] [1, 64] [1, 1] : tensor<64xf16> into tensor<1x64xf16>
// CHECK:         ktdp.store %[[INSERT_SLICE_0]], %[[CONSTRUCT_ACCESS_TILE_1]] : tensor<1x64xf16>, <1x64xindex>
// CHECK:       }
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @"local-schedule-0_keep_alive"() {
// CHECK:       call @"local-schedule-0"() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:   }

// Pass-01 ComputeGroupExtraction: the reduction loop nest (inner scf.for reduction over rows,
// with the ktdp.load inside the loop body and the ktdp.store outside it) is extracted into its
// own compute group / local_schedule.
//
// A 1D row-reduction: c[stick, :] = sum over rows of a[stick, :, :].
//   a: [2, 256, 64] f16, d1 = reduction axis;  c: [2, 64] f16.
module {
  func.func @row_reduction() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64xf16>

    %a_base = arith.constant 0 : index
    %c_base = arith.constant 8589934592 : index

    %a_view = ktdp.construct_memory_view %a_base, sizes: [2, 256, 64], strides: [16384, 64, 1] {
      coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x256x64xf16>

    %c_view = ktdp.construct_memory_view %c_base, sizes: [2, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x64xf16>

    // Outer: sweep the 2 output rows.
    scf.for %n_stick = %c0 to %c2 step %c1 {

      // Inner: reduction over the 256-row axis, 64-lane accumulator.
      %acc_final = scf.for %m = %c0 to %c256 step %c1
          iter_args(%acc = %zero) -> tensor<64xf16> {

        %a_tile = ktdp.construct_access_tile %a_view[%n_stick, %m, %c0] {
          access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        } : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %a_raw = ktdp.load %a_tile : !ktdp.access_tile<1x1x64xindex> -> tensor<1x1x64xf16>
        %a_1d = tensor.extract_slice %a_raw[0, 0, 0][1, 1, 64][1, 1, 1]
            : tensor<1x1x64xf16> to tensor<64xf16>

        // acc[i] += a_1d[i]
        %new_acc = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%a_1d : tensor<64xf16>) outs(%acc : tensor<64xf16>) {
        ^bb0(%a: f16, %o: f16):
          %add = arith.addf %a, %o : f16
          linalg.yield %add : f16
        } -> tensor<64xf16>
        scf.yield %new_acc : tensor<64xf16>
      }

      // Post-loop: store the reduced row.
      %c_tile = ktdp.construct_access_tile %c_view[%n_stick, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<2x64xf16> -> !ktdp.access_tile<1x64xindex>
      %c_2d = tensor.empty() : tensor<1x64xf16>
      %c_inserted = tensor.insert_slice %acc_final into %c_2d[0, 0][1, 64][1, 1]
          : tensor<64xf16> into tensor<1x64xf16>
      ktdp.store %c_inserted, %c_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }
}
