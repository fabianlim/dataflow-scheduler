// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering %s | FileCheck %s --check-prefix=LOWERING
// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering --ktdflowering-to-dfir %s | FileCheck %s
// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering --ktdflowering-to-dfir %s | FileCheck %s --check-prefix=NOSYNC

// The local copy of the broadcast relayout, on its own: a 64x64xf16 L1 tile
// is copied from element offset 0 to element offset 128, one stick per
// iteration, through L1LU -> SFU -> L1SU. The SFU stage is an identity
// linalg.generic, so it must lower to a plain receive and send.
//
// Every stage is mapped. Path expansion is not part of the RUN lines: it
// does not recognise a fully mapped pipeline as legal yet, and the backend
// needs every stage mapped.
//
// No stage has a domain yet, so each program unit runs on the units of all
// 32 cores.

// The stages are joined by tokens only, with no scratchpad dependency between
// them, so no signals are inserted.
// LOWERING-NOT:  ktdf_lowering.signal
// LOWERING:      ktdf_lowering.execute_on
// LOWERING-NOT:  ktdf_lowering.signal

// NOSYNC-NOT:    dataflow.sync
// NOSYNC-NOT:    ktdf_lowering

// CHECK-DAG:     #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG:     #[[$ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:     #[[$STICK:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @local_copy()
// CHECK-DAG:       %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG:       %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-DAG:       %[[LU31:.*]] = dataflow.get_unit {core = 31 : i32, corelet = 0 : i32, name = "C31-l1lu", type = "l1lu"} : index
// CHECK-DAG:       %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// CHECK-DAG:       %[[SFU31:.*]] = dataflow.get_unit {core = 31 : i32, corelet = 0 : i32, name = "C31-sfu", type = "sfu"} : index
// CHECK-DAG:       %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// CHECK-DAG:       %[[SU31:.*]] = dataflow.get_unit {core = 31 : i32, corelet = 0 : i32, name = "C31-l1su", type = "l1su"} : index
// CHECK-DAG:       %[[MEM0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index

// L1LU: load a stick from the view at offset 0 and send it to the SFU.
// CHECK:           dataflow.program_unit iter_arg : %[[PU0:.*]] -> (%[[LU0]], {{(%[0-9]+, ){30}%}}[[LU31]]) : {
// CHECK-NEXT:        %[[MAP0:.*]] = uniform.def_immutable_mapping([%[[LU0]] -> %[[MEM0]]], {{.*}}
// CHECK-NEXT:        %[[L1_0:.*]] = uniform.query_map(map:%[[MAP0]], key:%[[PU0]]) : index
// CHECK-NEXT:        %[[SRC:.*]] = dataflow.get_logical_memory_view %[[L1_0]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:        scf.for %[[ROW0:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:          %[[LOADED:.*]] = agen.vector_load %[[SRC]][%[[ROW0]], %[[C0]]] {load_order = #[[$ORDER]], load_set = #[[$STICK]]} : memref<64x64xf16>, vector<64xf16>
// CHECK-NEXT:          %[[MAP1:.*]] = uniform.def_immutable_mapping([%[[LU0]] -> %[[SFU0]]], {{.*}}
// CHECK-NEXT:          %[[TO_SFU:.*]] = uniform.query_map(map:%[[MAP1]], key:%[[PU0]]) : index
// CHECK-NEXT:          dataflow.send %[[TO_SFU]], %[[LOADED]] : vector<64xf16>
// CHECK-NEXT:        }
// CHECK-NEXT:      }

// SFU: receive and forward; the identity generic leaves no arithmetic.
// CHECK-NEXT:      dataflow.program_unit iter_arg : %[[PU1:.*]] -> (%[[SFU0]], {{(%[0-9]+, ){30}%}}[[SFU31]]) : {
// CHECK-NEXT:        scf.for %{{.*}} = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:          %[[MAP2:.*]] = uniform.def_immutable_mapping([%[[SFU0]] -> %[[LU0]]], {{.*}}
// CHECK-NEXT:          %[[FROM_LU:.*]] = uniform.query_map(map:%[[MAP2]], key:%[[PU1]]) : index
// CHECK-NEXT:          %[[FORWARDED:.*]] = dataflow.receive %[[FROM_LU]] : vector<64xf16>
// CHECK-NEXT:          %[[MAP3:.*]] = uniform.def_immutable_mapping([%[[SFU0]] -> %[[SU0]]], {{.*}}
// CHECK-NEXT:          %[[TO_SU:.*]] = uniform.query_map(map:%[[MAP3]], key:%[[PU1]]) : index
// CHECK-NEXT:          dataflow.send %[[TO_SU]], %[[FORWARDED]] : vector<64xf16>
// CHECK-NEXT:        }
// CHECK-NEXT:      }

// L1SU: receive a stick from the SFU and store it to the view at offset 128.
// CHECK-NEXT:      dataflow.program_unit iter_arg : %[[PU2:.*]] -> (%[[SU0]], {{(%[0-9]+, ){30}%}}[[SU31]]) : {
// CHECK-NEXT:        %[[MAP4:.*]] = uniform.def_immutable_mapping([%[[SU0]] -> %[[MEM0]]], {{.*}}
// CHECK-NEXT:        %[[L1_2:.*]] = uniform.query_map(map:%[[MAP4]], key:%[[PU2]]) : index
// CHECK-NEXT:        %[[DST:.*]] = dataflow.get_logical_memory_view %[[L1_2]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:        scf.for %[[ROW2:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:          %[[MAP5:.*]] = uniform.def_immutable_mapping([%[[SU0]] -> %[[SFU0]]], {{.*}}
// CHECK-NEXT:          %[[FROM_SFU:.*]] = uniform.query_map(map:%[[MAP5]], key:%[[PU2]]) : index
// CHECK-NEXT:          %[[STORED:.*]] = dataflow.receive %[[FROM_SFU]] : vector<64xf16>
// CHECK-NEXT:          agen.vector_store %[[STORED]], %[[DST]][%[[ROW2]], %[[C0]]] {store_order = #[[$ORDER]], store_set = #[[$STICK]]} : memref<64x64xf16>, vector<64xf16>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return

#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring32_device attributes {} import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @local_copy() attributes {grid = [32]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c128, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    scf.for %row = %c0 to %c64 step %c1 {
      ktdf.pipeline {
        %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
          %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          %t2 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %f0, %f1, %t0, %t1, %t2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%row, %c0] size [1, 64] to %p#0 size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %v = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
          %e = tensor.empty() : tensor<64xf16>
          %o = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%v : tensor<64xf16>) outs(%e : tensor<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            linalg.yield %in : f16
          } -> tensor<64xf16>
          ktdf.write_to_fifo %o, %p#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(%p#4) {
          ktdf.data_transfer from %p#1 size [64] to %dst[%row, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x64xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
