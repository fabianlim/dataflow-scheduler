// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2 * 64 + d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 64 + d2)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>



// CHECK-LABEL:   ktdf_arch.device @spyre_1core import("Inputs/spyre_1core_device.mlir")

// CHECK-LABEL:   func.func @corelet_parallel_fifo_mapping() attributes {grid = [1]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 65536 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 255 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
// CHECK-NEXT:     %[[VAL_7:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
// CHECK-NEXT:     %[[VAL_8:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
// CHECK-NEXT:     %[[VAL_9:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
// CHECK-NEXT:     %[[VAL_10:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
// CHECK-NEXT:     %[[VAL_11:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
// CHECK-NEXT:     %[[VAL_12:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_13:.*]] -> (%[[VAL_6]], %[[VAL_7]]) : {
// CHECK-NEXT:       %[[VAL_14:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_6]] -> %[[VAL_12]]], {{\[}}%[[VAL_7]] -> %[[VAL_12]]]):index
// CHECK-NEXT:       %[[VAL_15:.*]] = uniform.query_map(map:%[[VAL_14]], key:%[[VAL_13]]) : index
// CHECK-NEXT:       %[[VAL_16:.*]] = dataflow.get_logical_memory_view %[[VAL_15]], %[[VAL_5]] {layout_map = #[[$ATTR_0]]} : index, index, memref<1x1x1x64xf16>
// CHECK-NEXT:       scf.for %[[VAL_17:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:         %[[VAL_18:.*]] = agen.vector_load %[[VAL_16]][0, 0, 0, 0] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_5]]} : memref<1x1x1x64xf16>, vector<64xf16>
// CHECK-NEXT:         %[[VAL_19:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_6]] -> %[[VAL_8]]], {{\[}}%[[VAL_7]] -> %[[VAL_9]]]):index
// CHECK-NEXT:         %[[VAL_20:.*]] = uniform.query_map(map:%[[VAL_19]], key:%[[VAL_13]]) : index
// CHECK-NEXT:         dataflow.send %[[VAL_20]], %[[VAL_18]] : vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_21:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK-NEXT:       scf.for %[[VAL_22:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:         %[[VAL_23:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_6]]], {{\[}}%[[VAL_9]] -> %[[VAL_7]]]):index
// CHECK-NEXT:         %[[VAL_24:.*]] = uniform.query_map(map:%[[VAL_23]], key:%[[VAL_21]]) : index
// CHECK-NEXT:         %[[VAL_25:.*]] = dataflow.receive %[[VAL_24]] : vector<64xf16>
// CHECK-NEXT:         %[[VAL_26:.*]] = vectorchain.binary %[[VAL_0]], %[[VAL_25]] {binary_op = {{.*}}<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:         %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_22]], %[[VAL_3]] : index
// CHECK-NEXT:         scf.if %[[VAL_27]] {
// CHECK-NEXT:           %[[VAL_28:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_10]]], {{\[}}%[[VAL_9]] -> %[[VAL_11]]]):index
// CHECK-NEXT:           %[[VAL_29:.*]] = uniform.query_map(map:%[[VAL_28]], key:%[[VAL_21]]) : index
// CHECK-NEXT:           dataflow.send %[[VAL_29]], %[[VAL_26]] : vector<64xf16>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_30:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK-NEXT:       %[[VAL_31:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_10]] -> %[[VAL_12]]], {{\[}}%[[VAL_11]] -> %[[VAL_12]]]):index
// CHECK-NEXT:       %[[VAL_32:.*]] = uniform.query_map(map:%[[VAL_31]], key:%[[VAL_30]]) : index
// CHECK-NEXT:       %[[VAL_33:.*]] = dataflow.get_logical_memory_view %[[VAL_32]], %[[VAL_1]] {layout_map = #[[$ATTR_3]]} : index, index, memref<2x1x64xf16>
// CHECK-NEXT:       scf.for %[[VAL_34:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:         %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_34]], %[[VAL_3]] : index
// CHECK-NEXT:         scf.if %[[VAL_35]] {
// CHECK-NEXT:           %[[VAL_36:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_10]] -> %[[VAL_8]]], {{\[}}%[[VAL_11]] -> %[[VAL_9]]]):index
// CHECK-NEXT:           %[[VAL_37:.*]] = uniform.query_map(map:%[[VAL_36]], key:%[[VAL_30]]) : index
// CHECK-NEXT:           %[[VAL_38:.*]] = dataflow.receive %[[VAL_37]] : vector<64xf16>
// CHECK-NEXT:           agen.vector_store %[[VAL_38]], %[[VAL_33]][0, 0, 0] {store_order = #[[$ATTR_4]], store_set = #[[$ATTR_6]]} : memref<2x1x64xf16>, vector<64xf16>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Verify that FIFO endpoint unit mappings are corelet-parallel: each source
// corelet maps to its own destination corelet, not always corelet 0.
//
// Input: 2-corelet LXLU -> SFP -> LXSU reduction loop (derived from pass-20
// output of sum/1core_loopless).
//
// The bug: createQueryMapForComponent matched target units by core only,
// causing both LXLU-CL0 and LXLU-CL1 to map to SFP-CL0.
//
// Expected: corelet-parallel mappings
//   LXLU->SFP:  [lxlu_cl0 -> sfp_cl0], [lxlu_cl1 -> sfp_cl1]
//   SFP->LXLU:  [sfp_cl0  -> lxlu_cl0], [sfp_cl1  -> lxlu_cl1]
//   SFP->LXSU:  [sfp_cl0  -> lxsu_cl0], [sfp_cl1  -> lxsu_cl1]
//   LXSU->SFP:  [lxsu_cl0 -> sfp_cl0], [lxsu_cl1 -> sfp_cl1]


// LXLU program_unit sends to SFP: each LXLU corelet maps to its own SFP corelet

// SFP program_unit receives from LXLU and sends to LXSU, both corelet-parallel

// LXSU program_unit receives from SFP: each LXSU corelet maps to its own SFP corelet

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @spyre_1core import("Inputs/spyre_1core_device.mlir")
  func.func @corelet_parallel_fifo_mapping() attributes {grid = [1]} {
    %lxlu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
    %lxlu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
    %sfp0  = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0",  type = "sfp"}  : index
    %sfp1  = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1",  type = "sfp"}  : index
    %lxsu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
    %lxsu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index

    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c255 = arith.constant 255 : index
    %c256 = arith.constant 256 : index

    // Query maps for each unit type (corelet-indexed)
    %lxlu_map = uniform.def_immutable_mapping([%c0 -> %lxlu0], [%c1 -> %lxlu1]):index
    %lxlu_q   = uniform.query_map(map:%lxlu_map, key:%c0) : index
    %sfp_map  = uniform.def_immutable_mapping([%c0 -> %sfp0],  [%c1 -> %sfp1]):index
    %sfp_q    = uniform.query_map(map:%sfp_map,  key:%c0) : index
    %lxsu_map = uniform.def_immutable_mapping([%c0 -> %lxsu0], [%c1 -> %lxsu1]):index
    %lxsu_q   = uniform.query_map(map:%lxsu_map, key:%c0) : index

    // Outer execute_on containing the full pipeline
    ktdf_lowering.execute_on %lxlu_q, %sfp_q, %lxsu_q {
      // FIFO slots for the LXLU->SFP and SFP->LXSU routes
      %fifo_lxlu_sfp  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
      %fifo_sfp_lxsu  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>

      // LX buffer (source for LXLU transfers); 1x1x1x64 so element count matches
      // the FIFO slot width of 64
      %lx_buf_offset = arith.constant 0 : index
      %lx_buf = builtin.unrealized_conversion_cast %lx_buf_offset : index to memref<1x1x1x64xf16, "LX">

      // LXLU execute_on: load from LX buffer and send to SFP via FIFO
      ktdf_lowering.execute_on %lxlu_q {
        scf.for %iv = %c0 to %c256 step %c1 {
          ktdf.data_transfer from %lx_buf[0, 0, 0, 0] size [1, 1, 1, 64]
            to %fifo_lxlu_sfp size [64]
            : memref<1x1x1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
        }
      }

      // SFP execute_on: receive from LXLU FIFO, accumulate, send to LXSU FIFO on last iter
      ktdf_lowering.execute_on %sfp_q {
        %init = tensor.empty() : tensor<1x64xf16>
        %result = scf.for %iv = %c0 to %c256 step %c1 iter_args(%acc = %init) -> (tensor<1x64xf16>) {
          %chunk = ktdf.read_from_fifo %fifo_lxlu_sfp
            : <"LXLU" -> "SFP", 64xf16> -> tensor<1x1x64xf16>
          %reduced = linalg.generic {
            indexing_maps = [#map, #map1],
            iterator_types = ["parallel", "reduction", "parallel"]
          } ins(%chunk : tensor<1x1x64xf16>) outs(%acc : tensor<1x64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %s = arith.addf %in, %out : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          %is_last = arith.cmpi eq, %iv, %c255 : index
          scf.if %is_last {
            ktdf.write_to_fifo %reduced, %fifo_sfp_lxsu
              : tensor<1x64xf16>, <"SFP" -> "LXSU", 64xf16>
          }
          scf.yield %reduced : tensor<1x64xf16>
        }
      }

      // LXSU execute_on: receive from SFP FIFO and store to LX buffer
      %lx_out_offset = arith.constant 65536 : index
      %lx_out = builtin.unrealized_conversion_cast %lx_out_offset : index to memref<2x1x64xf16, "LX">
      ktdf_lowering.execute_on %lxsu_q {
        scf.for %iv = %c0 to %c256 step %c1 {
          %is_last = arith.cmpi eq, %iv, %c255 : index
          scf.if %is_last {
            ktdf.data_transfer from %fifo_sfp_lxsu size [64]
              to %lx_out[0, 0, 0] size [1, 1, 64]
              : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<2x1x64xf16, "LX">
          }
        }
      }
    }
    return
  }
}
