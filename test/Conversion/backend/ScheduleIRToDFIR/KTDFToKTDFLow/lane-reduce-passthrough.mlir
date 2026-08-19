// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)" %s | FileCheck %s

// `ktdf.opaque "LANE_REDUCE"` -- the carrier marking an on-stick (lane-axis)
// reduction -- must survive the KTDF -> KTDFLowering structural conversion
// unchanged, so that the DFIR lowering can find it inside a
// `ktdf_lowering.execute_on` body.
//
// The pass moves everything that is not a `ktdf.data_transfer` wholesale, so the
// carry needs no special handling; this test keeps that true -- template name,
// operands, result type and attributes must all come out identical.
//
// Note this pass also strips `loop_type` from every `scf.for`, which is why no
// downstream consumer can rely on `loop_type` to carry the classification.

// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   func.func @lane_reduce_passthrough() attributes {grid = [2]} {
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu", type = "l1lu"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu", type = "sfu"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// CHECK-NEXT:     %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1su", type = "l1su"} : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_2]] -> %[[GET_UNIT_2]]], {{\[}}%[[CONSTANT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_4]] -> %[[GET_UNIT_4]]], {{\[}}%[[CONSTANT_5]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_1]], %[[QUERY_MAP_2]] {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x64xf16, "L1">
// CHECK-NEXT:       %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:       %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:       %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_0]]{{\[}}%[[CONSTANT_6]], %[[CONSTANT_6]], %[[CONSTANT_6]]] size [1, 1, 64] to %[[FIFO_0]] size [64] : memref<1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_1]] {
// CHECK-NEXT:         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]] : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:         %[[OPAQUE_0:.*]] = ktdf.opaque "LANE_REDUCE" -> tensor<1x64xf16>
// CHECK-NEXT:           ins(%[[READ_FROM_FIFO_0]]: tensor<1x64xf16>)
// CHECK-NEXT:           outs(%[[EMPTY_0]]: tensor<1x64xf16>)
// CHECK-NEXT:         ktdf.write_to_fifo %[[OPAQUE_0]], %[[FIFO_1]] : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_2]] {
// CHECK-NEXT:         ktdf.data_transfer from %[[FIFO_1]] size [64] to %[[ALLOC_0]]{{\[}}%[[CONSTANT_6]], %[[CONSTANT_6]], %[[CONSTANT_6]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#set     = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @lane_reduce_passthrough() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index

    ktdf.pipeline {
      %p:5 = ktdf.private -> (memref<1x1x64xf16, "L1">,
                              !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                              !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                              !ktdf.token, !ktdf.token) {
        %alloc = memref.alloc() : memref<1x1x64xf16, "L1">
        %fin  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %fout = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %alloc, %fin, %fout, %t0, %t1
          : memref<1x1x64xf16, "L1">,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
            !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
            !ktdf.token, !ktdf.token
      }

      // Load stage: one 64-lane row into the L1LU -> SFU FIFO.
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %p#0[%c0, %c0, %c0] size [1, 1, 64]
                           to %p#1 size [64]
          : memref<1x1x64xf16, "L1">,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"]}

      // Compute stage: the lane reduction, carried as ktdf.opaque.
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        %v = ktdf.read_from_fifo %p#1
          : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
        %e = tensor.empty() : tensor<1x64xf16>
        %r = ktdf.opaque "LANE_REDUCE" -> tensor<1x64xf16>
          ins(%v : tensor<1x64xf16>)
          outs(%e : tensor<1x64xf16>)
        ktdf.write_to_fifo %r, %p#2
          : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"]}

      // Store stage.
      ktdf.stage depends_in(%p#4) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64]
                           to %p#0[%c0, %c0, %c0] size [1, 1, 64]
          : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
            memref<1x1x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}

