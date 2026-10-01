// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering %s | FileCheck %s

// The broadcast relayout as the frontend writes it, on the 32-core ring
// device, through the scheduling steps it needs so far: path expansion maps
// the channel's stages to MNISU and MNILU, handle-cross-core-stages realizes
// the self-delivery of core 0 as a local copy L1LU -> SFU -> L1SU, and
// ktdf-to-ktdflowering materializes the units on the tiles of the leaf
// stages of each kind.
//
// The channel has the groups g = 0..7: producer core g sends its slice to the
// consumer cores 4g..4g+3. The ring carries C(g) without P(g), so MNISU is on
// cores 0..7 and MNILU on cores 1..31; the local copy carries core 0's slice
// to core 0, so L1LU, SFU and L1SU are on core 0 only. Each kind's map from
// tile to unit lists exactly those tiles. The stages that wrap the channel
// and the local copy are not counted for the tiles; they execute on the units
// of the stages they wrap.
//
// The ring stages still pair their units core by core (the signal and, in
// DFIR, the send and receive of the fifo); ring lowering will pair each
// consumer with the producer of its group, so the test stops before DFIR.

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$SET:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$GROUP_DOMAIN:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
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
// CHECK-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-NEXT:     %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// CHECK-NEXT:     %[[L1SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
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
// CHECK-NEXT:     %[[LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU_KEY1]] -> %[[LU1]]], {{\[}}%[[LU_KEY2]] -> %[[LU2]]], {{\[}}%[[LU_KEY3]] -> %[[LU3]]], {{\[}}%[[LU_KEY4]] -> %[[LU4]]], {{\[}}%[[LU_KEY5]] -> %[[LU5]]], {{\[}}%[[LU_KEY6]] -> %[[LU6]]], {{\[}}%[[LU_KEY7]] -> %[[LU7]]], {{\[}}%[[LU_KEY8]] -> %[[LU8]]], {{\[}}%[[LU_KEY9]] -> %[[LU9]]], {{\[}}%[[LU_KEY10]] -> %[[LU10]]], {{\[}}%[[LU_KEY11]] -> %[[LU11]]], {{\[}}%[[LU_KEY12]] -> %[[LU12]]], {{\[}}%[[LU_KEY13]] -> %[[LU13]]], {{\[}}%[[LU_KEY14]] -> %[[LU14]]], {{\[}}%[[LU_KEY15]] -> %[[LU15]]], {{\[}}%[[LU_KEY16]] -> %[[LU16]]], {{\[}}%[[LU_KEY17]] -> %[[LU17]]], {{\[}}%[[LU_KEY18]] -> %[[LU18]]], {{\[}}%[[LU_KEY19]] -> %[[LU19]]], {{\[}}%[[LU_KEY20]] -> %[[LU20]]], {{\[}}%[[LU_KEY21]] -> %[[LU21]]], {{\[}}%[[LU_KEY22]] -> %[[LU22]]], {{\[}}%[[LU_KEY23]] -> %[[LU23]]], {{\[}}%[[LU_KEY24]] -> %[[LU24]]], {{\[}}%[[LU_KEY25]] -> %[[LU25]]], {{\[}}%[[LU_KEY26]] -> %[[LU26]]], {{\[}}%[[LU_KEY27]] -> %[[LU27]]], {{\[}}%[[LU_KEY28]] -> %[[LU28]]], {{\[}}%[[LU_KEY29]] -> %[[LU29]]], {{\[}}%[[LU_KEY30]] -> %[[LU30]]], {{\[}}%[[LU_KEY31]] -> %[[LU31]]]):index
// CHECK-NEXT:     %[[LU:.*]] = uniform.query_map(map:%[[LU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[L1LU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[L1LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_KEY0]] -> %[[L1LU0]]]):index
// CHECK-NEXT:     %[[L1LU:.*]] = uniform.query_map(map:%[[L1LU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[SFU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SFU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SFU_KEY0]] -> %[[SFU0]]]):index
// CHECK-NEXT:     %[[SFU:.*]] = uniform.query_map(map:%[[SFU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[L1SU_KEY0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[L1SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1SU_KEY0]] -> %[[L1SU0]]]):index
// CHECK-NEXT:     %[[L1SU:.*]] = uniform.query_map(map:%[[L1SU_MAP]], key:%[[TILE]]) : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[SRC_VIEW:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[SRC:.*]] = memref.memory_space_cast %[[SRC_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// CHECK-NEXT:     %[[DST_VIEW:.*]] = ktdp.construct_memory_view %[[C128]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[DST:.*]] = memref.memory_space_cast %[[DST_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// CHECK-NEXT:     scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:       ktdf_lowering.execute_on %[[SU]], %[[LU]], %[[L1LU]], %[[SFU]], %[[L1SU]] {
// CHECK-NEXT:         ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// CHECK-NEXT:           ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// CHECK-NEXT:             %[[CH:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$GROUP_DOMAIN]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:             %[[TOKEN:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf_lowering.execute_on %[[SU]] {
// CHECK-NEXT:               ktdf.data_transfer from %[[SRC]]{{\[}}%[[I]], %[[C0]]] size [1, 64] to %[[CH]] size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf_lowering.signal %[[SU]], %[[LU]]
// CHECK-NEXT:             ktdf_lowering.execute_on %[[LU]] {
// CHECK-NEXT:               ktdf.data_transfer from %[[CH]] size [64] to %[[DST]]{{\[}}%[[I]], %[[C0]]] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf_lowering.execute_on %[[L1LU]], %[[SFU]], %[[L1SU]] {
// CHECK-NEXT:           ktdf_lowering.execute_on %[[L1LU]], %[[SFU]], %[[L1SU]] {
// CHECK-NEXT:             %[[TO_SFU:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[FROM_SFU:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             %[[LOADED:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[COMPUTED:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf_lowering.execute_on %[[L1LU]] {
// CHECK-NEXT:               ktdf.data_transfer from %[[SRC]]{{\[}}%[[I]], %[[C0]]] size [1, 64] to %[[TO_SFU]] size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf_lowering.execute_on %[[SFU]] {
// CHECK-NEXT:               %[[IN:.*]] = ktdf.read_from_fifo %[[TO_SFU]] : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:               %[[INIT:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:               %[[OUT:.*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]} ins(%[[IN]] : tensor<64xf16>) outs(%[[INIT]] : tensor<64xf16>) {
// CHECK-NEXT:               ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:                 linalg.yield %[[X]] : f16
// CHECK-NEXT:               } -> tensor<64xf16>
// CHECK-NEXT:               ktdf.write_to_fifo %[[OUT]], %[[FROM_SFU]] : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf_lowering.execute_on %[[L1SU]] {
// CHECK-NEXT:               ktdf.data_transfer from %[[FROM_SFU]] size [64] to %[[DST]]{{\[}}%[[I]], %[[C0]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(g) : (g >= 0, 7 - g >= 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
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
          %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
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
