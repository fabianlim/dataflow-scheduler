// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2 * 64 + d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ACC_ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 64 + d2)>
// CHECK: #[[$ATTR_3D_ID:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$SET_ACC:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>



// CHECK-LABEL:   ktdf_arch.device @spyre_1core import("Inputs/spyre_1core_device.mlir")

// CHECK-LABEL:   func.func @corelet_parallel_fifo_mapping() attributes {grid = [1]} {
// CHECK-DAG:       %[[C65536:.*]] = arith.constant 65536 : index
// CHECK-DAG:       %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[LXLU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
// CHECK-DAG:       %[[LXLU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
// CHECK-DAG:       %[[SFP0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
// CHECK-DAG:       %[[SFP1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
// CHECK-DAG:       %[[LXSU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
// CHECK-DAG:       %[[LXSU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
// CHECK-DAG:       %[[LX:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index

// LXLU program_unit: sends to SFP (corelet-parallel)
// CHECK:       dataflow.program_unit iter_arg : %[[LXLU_ARG:.*]] -> (%[[LXLU0]], %[[LXLU1]]) : {
// CHECK:         %[[LXLU_LX_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LXLU0]] -> %[[LX]]], {{\[}}%[[LXLU1]] -> %[[LX]]]):index
// CHECK:         %[[LXLU_LX:.*]] = uniform.query_map(map:%[[LXLU_LX_MAP]], key:%[[LXLU_ARG]]) : index
// CHECK:         %[[LXLU_VIEW:.*]] = dataflow.get_logical_memory_view %[[LXLU_LX]], %[[C0]] {layout_map = #[[$ATTR_0]]} : index, index, memref<1x1x1x64xf16>
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C256]] step %[[C1]] {
// CHECK:           %{{.*}} = agen.vector_load %[[LXLU_VIEW]]
// CHECK:           %[[SEND_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LXLU0]] -> %[[SFP0]]], {{\[}}%[[LXLU1]] -> %[[SFP1]]]):index
// CHECK:           %[[SEND_DST:.*]] = uniform.query_map(map:%[[SEND_MAP]], key:%[[LXLU_ARG]]) : index
// CHECK:           dataflow.send %[[SEND_DST]],
// CHECK:         }
// CHECK:       }

// SFP program_unit: receives from LXLU (corelet-parallel), accumulates in the
// register-file buffer, sends to LXSU every iteration
// CHECK:       dataflow.program_unit iter_arg : %[[SFP_ARG:.*]] -> (%[[SFP0]], %[[SFP1]]) : {
// CHECK:         %[[ACC:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK:         %[[SEED:.*]] = vectorchain.constant_bitstream {value = [0x0]} : vector<1xf16>
// CHECK:         %[[ZERO:.*]] = vectorchain.shuffle %[[SEED]] {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:         agen.vector_store %[[ZERO]], %[[ACC]][%[[C0]], %[[C0]]] {{.*}} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK:         scf.for %[[IV:.*]] = %[[C0]] to %[[C256]] step %[[C1]] {
// CHECK:           %[[RECV_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SFP0]] -> %[[LXLU0]]], {{\[}}%[[SFP1]] -> %[[LXLU1]]]):index
// CHECK:           %[[RECV_SRC:.*]] = uniform.query_map(map:%[[RECV_MAP]], key:%[[SFP_ARG]]) : index
// CHECK:           %[[RECV:.*]] = dataflow.receive %[[RECV_SRC]] : vector<64xf16>
// CHECK:           %[[LOAD:.*]] = agen.vector_load %[[ACC]][%[[C0]], %[[C0]]] {{.*}} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK:           %[[SUM:.*]] = vectorchain.binary %[[RECV]], %[[LOAD]] {binary_op = {{.*}}<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK:           agen.vector_store %[[SUM]], %[[ACC]][%[[C0]], %[[C0]]] {{.*}} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK:           %[[SEND_MAP2:.*]] = uniform.def_immutable_mapping({{\[}}%[[SFP0]] -> %[[LXSU0]]], {{\[}}%[[SFP1]] -> %[[LXSU1]]]):index
// CHECK:           %[[SEND_DST2:.*]] = uniform.query_map(map:%[[SEND_MAP2]], key:%[[SFP_ARG]]) : index
// CHECK:           %[[OUT:.*]] = agen.vector_load %[[ACC]][%[[C0]], %[[C0]]] {{.*}} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK:           dataflow.send %[[SEND_DST2]], %[[OUT]] : vector<64xf16>
// CHECK:         }
// CHECK:       }

// LXSU program_unit: receives from SFP (corelet-parallel)
// CHECK:       dataflow.program_unit iter_arg : %[[LXSU_ARG:.*]] -> (%[[LXSU0]], %[[LXSU1]]) : {
// CHECK:         %[[LXSU_LX_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LXSU0]] -> %[[LX]]], {{\[}}%[[LXSU1]] -> %[[LX]]]):index
// CHECK:         %[[LXSU_LX:.*]] = uniform.query_map(map:%[[LXSU_LX_MAP]], key:%[[LXSU_ARG]]) : index
// CHECK:         %[[LXSU_VIEW:.*]] = dataflow.get_logical_memory_view %[[LXSU_LX]], %[[C65536]] {layout_map = #[[$ATTR_3]]} : index, index, memref<2x1x64xf16>
// CHECK:         scf.for %[[LXSU_IV:.*]] = %[[C0]] to %[[C256]] step %[[C1]] {
// CHECK:           %[[RECV_MAP3:.*]] = uniform.def_immutable_mapping({{\[}}%[[LXSU0]] -> %[[SFP0]]], {{\[}}%[[LXSU1]] -> %[[SFP1]]]):index
// CHECK:           %[[RECV_SRC3:.*]] = uniform.query_map(map:%[[RECV_MAP3]], key:%[[LXSU_ARG]]) : index
// CHECK:           %[[RECV3:.*]] = dataflow.receive %[[RECV_SRC3]] : vector<64xf16>
// CHECK:           agen.vector_store %[[RECV3]], %[[LXSU_VIEW]][0, 0, 0] {{.*}} : memref<2x1x64xf16>, vector<64xf16>
// CHECK:         }
// CHECK:       }



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

      // SFP execute_on: receive from LXLU FIFO, accumulate into the register-file
      // buffer, send to LXSU FIFO every iteration
      ktdf_lowering.execute_on %sfp_q {
        %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
        %zero = arith.constant 0.0 : f16
        linalg.fill ins(%zero : f16) outs(%acc : memref<1x64xf16, "SFU_REG">)
        scf.for %iv = %c0 to %c256 step %c1 {
          %chunk = ktdf.read_from_fifo %fifo_lxlu_sfp
            : <"LXLU" -> "SFP", 64xf16> -> memref<1x1x64xf16>
          linalg.generic {
            indexing_maps = [#map, #map1],
            iterator_types = ["parallel", "reduction", "parallel"]
          } ins(%chunk : memref<1x1x64xf16>) outs(%acc : memref<1x64xf16, "SFU_REG">) {
          ^bb0(%in: f16, %out: f16):
            %s = arith.addf %in, %out : f16
            linalg.yield %s : f16
          }
          ktdf.write_to_fifo %acc, %fifo_sfp_lxsu
            : memref<1x64xf16, "SFU_REG">, <"SFP" -> "LXSU", 64xf16>
        }
      }

      // LXSU execute_on: receive from SFP FIFO and store to LX buffer
      %lx_out_offset = arith.constant 65536 : index
      %lx_out = builtin.unrealized_conversion_cast %lx_out_offset : index to memref<2x1x64xf16, "LX">
      ktdf_lowering.execute_on %lxsu_q {
        scf.for %iv = %c0 to %c256 step %c1 {
          ktdf.data_transfer from %fifo_sfp_lxsu size [64]
            to %lx_out[0, 0, 0] size [1, 1, 64]
            : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<2x1x64xf16, "LX">
        }
      }
    }
    return
  }
}
