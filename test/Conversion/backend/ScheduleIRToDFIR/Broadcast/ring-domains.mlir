// RUN: dataflow-scheduler-opt --path-expansion --ktdf-to-ktdflowering %s | FileCheck %s

// The frontend form of the broadcast relayout on the 32-core ring device:
// producer core k sends its slice to the consumer cores c with
// c floordiv 4 == k. The producer stage has the domain cores 0..7, the
// consumer stage cores 0..31, and the producer domain is the image of the
// peer relation over the consumer domain. Path expansion maps the stages to
// MNISU and MNILU and keeps their domains.
//
// Units are materialized per kind on the tiles of the domain of its stages:
// MNISU on cores 0..7 and MNILU on cores 0..31, and each kind's map from tile
// to unit lists exactly those tiles: after lowering to DFIR, they are the
// units the kind's program unit runs on.
//
// The test stops before DFIR: that lowering pairs the units of the two stages
// core by core (for the signal and for the send and receive of the fifo),
// which consumers 8..31 cannot do, having no producer unit on their core. Ring
// lowering will pair them through the peer relation instead.

// CHECK: #[[$PEER:.+]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK: #[[$SET:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")

// CHECK-LABEL:   func.func @ring_broadcast() attributes {grid = [32]} {
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU4:.*]] = dataflow.get_unit {core = 4 : i32, corelet = 0 : i32, name = "C4-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU5:.*]] = dataflow.get_unit {core = 5 : i32, corelet = 0 : i32, name = "C5-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU6:.*]] = dataflow.get_unit {core = 6 : i32, corelet = 0 : i32, name = "C6-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU7:.*]] = dataflow.get_unit {core = 7 : i32, corelet = 0 : i32, name = "C7-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU4:.*]] = dataflow.get_unit {core = 4 : i32, corelet = 0 : i32, name = "C4-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU5:.*]] = dataflow.get_unit {core = 5 : i32, corelet = 0 : i32, name = "C5-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU6:.*]] = dataflow.get_unit {core = 6 : i32, corelet = 0 : i32, name = "C6-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU7:.*]] = dataflow.get_unit {core = 7 : i32, corelet = 0 : i32, name = "C7-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU8:.*]] = dataflow.get_unit {core = 8 : i32, corelet = 0 : i32, name = "C8-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU9:.*]] = dataflow.get_unit {core = 9 : i32, corelet = 0 : i32, name = "C9-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU10:.*]] = dataflow.get_unit {core = 10 : i32, corelet = 0 : i32, name = "C10-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU11:.*]] = dataflow.get_unit {core = 11 : i32, corelet = 0 : i32, name = "C11-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU12:.*]] = dataflow.get_unit {core = 12 : i32, corelet = 0 : i32, name = "C12-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU13:.*]] = dataflow.get_unit {core = 13 : i32, corelet = 0 : i32, name = "C13-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU14:.*]] = dataflow.get_unit {core = 14 : i32, corelet = 0 : i32, name = "C14-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU15:.*]] = dataflow.get_unit {core = 15 : i32, corelet = 0 : i32, name = "C15-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU16:.*]] = dataflow.get_unit {core = 16 : i32, corelet = 0 : i32, name = "C16-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU17:.*]] = dataflow.get_unit {core = 17 : i32, corelet = 0 : i32, name = "C17-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU18:.*]] = dataflow.get_unit {core = 18 : i32, corelet = 0 : i32, name = "C18-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU19:.*]] = dataflow.get_unit {core = 19 : i32, corelet = 0 : i32, name = "C19-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU20:.*]] = dataflow.get_unit {core = 20 : i32, corelet = 0 : i32, name = "C20-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU21:.*]] = dataflow.get_unit {core = 21 : i32, corelet = 0 : i32, name = "C21-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU22:.*]] = dataflow.get_unit {core = 22 : i32, corelet = 0 : i32, name = "C22-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU23:.*]] = dataflow.get_unit {core = 23 : i32, corelet = 0 : i32, name = "C23-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU24:.*]] = dataflow.get_unit {core = 24 : i32, corelet = 0 : i32, name = "C24-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU25:.*]] = dataflow.get_unit {core = 25 : i32, corelet = 0 : i32, name = "C25-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU26:.*]] = dataflow.get_unit {core = 26 : i32, corelet = 0 : i32, name = "C26-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU27:.*]] = dataflow.get_unit {core = 27 : i32, corelet = 0 : i32, name = "C27-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU28:.*]] = dataflow.get_unit {core = 28 : i32, corelet = 0 : i32, name = "C28-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU29:.*]] = dataflow.get_unit {core = 29 : i32, corelet = 0 : i32, name = "C29-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU30:.*]] = dataflow.get_unit {core = 30 : i32, corelet = 0 : i32, name = "C30-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU31:.*]] = dataflow.get_unit {core = 31 : i32, corelet = 0 : i32, name = "C31-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[TILE:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[SU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[SU_KEY2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[SU_KEY3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[SU_KEY4:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[SU_KEY5:.*]] = arith.constant 5 : index
// CHECK-NEXT:     %[[SU_KEY6:.*]] = arith.constant 6 : index
// CHECK-NEXT:     %[[SU_KEY7:.*]] = arith.constant 7 : index
// CHECK-NEXT:     %[[SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU_KEY0]] -> %[[SU0]]], {{\[}}%[[SU_KEY1]] -> %[[SU1]]], {{\[}}%[[SU_KEY2]] -> %[[SU2]]], {{\[}}%[[SU_KEY3]] -> %[[SU3]]], {{\[}}%[[SU_KEY4]] -> %[[SU4]]], {{\[}}%[[SU_KEY5]] -> %[[SU5]]], {{\[}}%[[SU_KEY6]] -> %[[SU6]]], {{\[}}%[[SU_KEY7]] -> %[[SU7]]]):index
// CHECK-NEXT:     %[[SU:.*]] = uniform.query_map(map:%[[SU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[LU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[LU_KEY1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[LU_KEY2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[LU_KEY3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[LU_KEY4:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[LU_KEY5:.*]] = arith.constant 5 : index
// CHECK-NEXT:     %[[LU_KEY6:.*]] = arith.constant 6 : index
// CHECK-NEXT:     %[[LU_KEY7:.*]] = arith.constant 7 : index
// CHECK-NEXT:     %[[LU_KEY8:.*]] = arith.constant 8 : index
// CHECK-NEXT:     %[[LU_KEY9:.*]] = arith.constant 9 : index
// CHECK-NEXT:     %[[LU_KEY10:.*]] = arith.constant 10 : index
// CHECK-NEXT:     %[[LU_KEY11:.*]] = arith.constant 11 : index
// CHECK-NEXT:     %[[LU_KEY12:.*]] = arith.constant 12 : index
// CHECK-NEXT:     %[[LU_KEY13:.*]] = arith.constant 13 : index
// CHECK-NEXT:     %[[LU_KEY14:.*]] = arith.constant 14 : index
// CHECK-NEXT:     %[[LU_KEY15:.*]] = arith.constant 15 : index
// CHECK-NEXT:     %[[LU_KEY16:.*]] = arith.constant 16 : index
// CHECK-NEXT:     %[[LU_KEY17:.*]] = arith.constant 17 : index
// CHECK-NEXT:     %[[LU_KEY18:.*]] = arith.constant 18 : index
// CHECK-NEXT:     %[[LU_KEY19:.*]] = arith.constant 19 : index
// CHECK-NEXT:     %[[LU_KEY20:.*]] = arith.constant 20 : index
// CHECK-NEXT:     %[[LU_KEY21:.*]] = arith.constant 21 : index
// CHECK-NEXT:     %[[LU_KEY22:.*]] = arith.constant 22 : index
// CHECK-NEXT:     %[[LU_KEY23:.*]] = arith.constant 23 : index
// CHECK-NEXT:     %[[LU_KEY24:.*]] = arith.constant 24 : index
// CHECK-NEXT:     %[[LU_KEY25:.*]] = arith.constant 25 : index
// CHECK-NEXT:     %[[LU_KEY26:.*]] = arith.constant 26 : index
// CHECK-NEXT:     %[[LU_KEY27:.*]] = arith.constant 27 : index
// CHECK-NEXT:     %[[LU_KEY28:.*]] = arith.constant 28 : index
// CHECK-NEXT:     %[[LU_KEY29:.*]] = arith.constant 29 : index
// CHECK-NEXT:     %[[LU_KEY30:.*]] = arith.constant 30 : index
// CHECK-NEXT:     %[[LU_KEY31:.*]] = arith.constant 31 : index
// CHECK-NEXT:     %[[LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU_KEY0]] -> %[[LU0]]], {{\[}}%[[LU_KEY1]] -> %[[LU1]]], {{\[}}%[[LU_KEY2]] -> %[[LU2]]], {{\[}}%[[LU_KEY3]] -> %[[LU3]]], {{\[}}%[[LU_KEY4]] -> %[[LU4]]], {{\[}}%[[LU_KEY5]] -> %[[LU5]]], {{\[}}%[[LU_KEY6]] -> %[[LU6]]], {{\[}}%[[LU_KEY7]] -> %[[LU7]]], {{\[}}%[[LU_KEY8]] -> %[[LU8]]], {{\[}}%[[LU_KEY9]] -> %[[LU9]]], {{\[}}%[[LU_KEY10]] -> %[[LU10]]], {{\[}}%[[LU_KEY11]] -> %[[LU11]]], {{\[}}%[[LU_KEY12]] -> %[[LU12]]], {{\[}}%[[LU_KEY13]] -> %[[LU13]]], {{\[}}%[[LU_KEY14]] -> %[[LU14]]], {{\[}}%[[LU_KEY15]] -> %[[LU15]]], {{\[}}%[[LU_KEY16]] -> %[[LU16]]], {{\[}}%[[LU_KEY17]] -> %[[LU17]]], {{\[}}%[[LU_KEY18]] -> %[[LU18]]], {{\[}}%[[LU_KEY19]] -> %[[LU19]]], {{\[}}%[[LU_KEY20]] -> %[[LU20]]], {{\[}}%[[LU_KEY21]] -> %[[LU21]]], {{\[}}%[[LU_KEY22]] -> %[[LU22]]], {{\[}}%[[LU_KEY23]] -> %[[LU23]]], {{\[}}%[[LU_KEY24]] -> %[[LU24]]], {{\[}}%[[LU_KEY25]] -> %[[LU25]]], {{\[}}%[[LU_KEY26]] -> %[[LU26]]], {{\[}}%[[LU_KEY27]] -> %[[LU27]]], {{\[}}%[[LU_KEY28]] -> %[[LU28]]], {{\[}}%[[LU_KEY29]] -> %[[LU29]]], {{\[}}%[[LU_KEY30]] -> %[[LU30]]], {{\[}}%[[LU_KEY31]] -> %[[LU31]]]):index
// CHECK-NEXT:     %[[LU:.*]] = uniform.query_map(map:%[[LU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[SRC_VIEW:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[SRC:.*]] = memref.memory_space_cast %[[SRC_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// CHECK-NEXT:     %[[DST_VIEW:.*]] = ktdp.construct_memory_view %[[C128]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[DST:.*]] = memref.memory_space_cast %[[DST_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// CHECK-NEXT:     scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:       ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// CHECK-NEXT:         %[[CH:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.peer = #[[$PEER]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         %[[TOKEN:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf_lowering.execute_on %[[SU]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[SRC]]{{\[}}%[[I]], %[[C0]]] size [1, 64] to %[[CH]] size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf_lowering.signal %[[SU]], %[[LU]]
// CHECK-NEXT:         ktdf_lowering.execute_on %[[LU]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[CH]] size [64] to %[[DST]]{{\[}}%[[I]], %[[C0]]] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#peer = affine_map<(c) -> (c floordiv 4)>
#producers = affine_set<(c) : (c >= 0, 7 - c >= 0)>
#consumers = affine_set<(c) : (c >= 0, 31 - c >= 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @ring_broadcast() attributes {grid = [32]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c128, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    scf.for %i = %c0 to %c64 step %c1 {
      ktdf.pipeline {
        %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
          %ch = ktdf.fifo.allocate() {dataflow_scheduler.peer = #peer} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#1) {
          ktdf.data_transfer from %src[%i, %c0] size [1, 64] to %p#0 size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        } {dataflow_scheduler.domain = #producers}
        ktdf.stage depends_in(%p#1) depends_out(%p#2) {
          ktdf.data_transfer from %p#0 size [64] to %dst[%i, %c0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
        } {dataflow_scheduler.domain = #consumers}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
