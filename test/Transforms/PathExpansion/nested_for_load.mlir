// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// Test: PathExpansion handles data_transfer nested inside scf.for in a load stage
// and read_from_fifo/write_to_fifo nested inside scf.for in the compute stage.
// This reproduces the reduction-loop pattern where Stage 1 iterates over the
// reduction dimension and streams one row per iteration.
//
// Key invariants:
//   - Abstract "HBM"->"SFP" / "SFP"->"HBM" FIFOs are gone; replaced by LX
//     buffers and physical LXLU/LXSU FIFOs.
//   - Stage 1 (L3LU): scf.for body carries data_transfer to the LX buffer.
//   - Synthetic intermediate stage (LXLU): L3LU->LXLU hop wraps data_transfer
//     in a new scf.for with loop_type = reduction_loop, matching the L3LU stage.
//   - Stage 3 (SFP): read_from_fifo uses "LXLU"->"SFP", write_to_fifo uses
//     "SFP"->"LXSU".

// CHECK-LABEL:   ktdf_arch.device @spyre_1core

// CHECK-LABEL:   func.func @local_schedule_0() {
// CHECK:           ktdf.pipeline {
// CHECK:             %[[PRIV:.*]]:8 = ktdf.private -> (memref<1x64xf16, "LX">, memref<64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>, !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK:               memref.alloc() : memref<1x64xf16, "LX">
// CHECK:               memref.alloc() : memref<64xf16, "LX">
// CHECK:               ktdf.fifo.allocate() -> !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK:               ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>
// CHECK:             }

// Stage 1 (L3LU): load from HBM into LX buffer, wrapped in reduction scf.for.
// CHECK:             ktdf.stage depends_in(none) depends_out(%[[PRIV]]#4) {
// CHECK:               scf.for {{.*}} {
// CHECK:                 ktdf.data_transfer from {{.*}} to %[[PRIV]]#0[0, 0] size [1, 64] : memref<16x64xf16, strided<[64, 1], offset: ?>, "HBM">, memref<1x64xf16, "LX">
// CHECK:               } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:             } {applicable_units = ["L3LU"]}

// Synthetic LXLU stage: LX buffer -> LXLU FIFO, wrapped in the same
// reduction scf.for so the FIFO sees one row per loop iteration.
// CHECK:             ktdf.stage depends_in(%[[PRIV]]#4) depends_out(%[[PRIV]]#5) {
// CHECK:               scf.for {{.*}} {
// CHECK:                 ktdf.data_transfer from %[[PRIV]]#0[0, 0] size [1, 64] to %[[PRIV]]#2 size [64] : memref<1x64xf16, "LX">, !ktdf.fifo.slot<"LXLU" -> "SFP", 64xf16>
// CHECK:               } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:             } {applicable_units = ["LXLU"]}

// SFP compute stage: reads from LXLU FIFO, accumulates, writes to LXSU FIFO.
// CHECK:             ktdf.stage depends_in(%[[PRIV]]#5) depends_out(%[[PRIV]]#6) {
// CHECK:               scf.for {{.*}} {
// CHECK:                 ktdf.read_from_fifo %[[PRIV]]#2 : <"LXLU" -> "SFP", 64xf16>
// CHECK:               } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:               ktdf.write_to_fifo {{.*}}, %[[PRIV]]#3 : memref<64xf16, "LX">, <"SFP" -> "LXSU", 64xf16>
// CHECK:             } {applicable_units = ["SFP"]}

// LXSU store-to-LX stage.
// CHECK:             ktdf.stage depends_in(%[[PRIV]]#6) depends_out(%[[PRIV]]#7) {
// CHECK:               ktdf.data_transfer from %[[PRIV]]#3 size [64] to %[[PRIV]]#1[0] size [64] : !ktdf.fifo.slot<"SFP" -> "LXSU", 64xf16>, memref<64xf16, "LX">
// CHECK:             } {applicable_units = ["LXSU"]}

// L3SU stage: store from LX buffer to HBM.
// CHECK:             ktdf.stage depends_in(%[[PRIV]]#7) depends_out(none) {
// CHECK:               ktdf.data_transfer from %[[PRIV]]#1[0] size [64] to {{.*}}[0] size [64] : memref<64xf16, "LX">, memref<64xf16, strided<[1], offset: ?>, "HBM">
// CHECK:             } {applicable_units = ["L3SU"]}
// CHECK:           }
// CHECK:           return
// CHECK:         }

#lrfreg = {kind = "lrfreg", size = 65536 : i64, ktdf_arch.features = {ktdf_arch.feature.local_to_compute}}
#SFP = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
#LXLU = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
#LXSU = {kind = "LXSU", load_store}
#core = {kind = "core"}
#HBM = {kind = "HBM", size = 1073741824 : i64}
#LX = {kind = "LX", size = 1048576 : i64}
#L3LU = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
#L3SU = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @spyre_1core attributes {
    mem_space_mapping = #ktdf_arch.map<
      #ktdp.spyre_memory_space<HBM> = "HBM",
      #ktdp.spyre_memory_space<LX>  = "LX"
    >
  } {
    %memory = memory #HBM
    group #core share(%memory) {
      %memory_0 = memory #LX
      %exec_unit = exec_unit #L3LU
      %exec_unit_2 = exec_unit #L3SU
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory to %exec_unit : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit to %memory_0 : exec_unit, memory
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory_0 to %exec_unit_2 : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit_2 to %memory : exec_unit, memory
      group share(%memory_0) {
        %lrfreg_0 = memory #lrfreg
        %exec_unit_3 = exec_unit #SFP
        %exec_unit_4 = exec_unit #LXLU
        %exec_unit_5 = exec_unit #LXSU
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %memory_0 to %exec_unit_4 : memory, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_4 to %exec_unit_3 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_3 to %exec_unit_5 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %exec_unit_5 to %memory_0 : exec_unit, memory
      }
    }
  }

  func.func @local_schedule_0() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    %a_view = ktdp.construct_memory_view %c1024, sizes: [16, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x64xf16>
    %a_hbm = memref.memory_space_cast %a_view : memref<16x64xf16> to memref<16x64xf16, "HBM">
    %a = memref.reinterpret_cast %a_hbm to offset: [%c0], sizes: [16, 64], strides: [64, 1] : memref<16x64xf16, "HBM"> to memref<16x64xf16, strided<[64, 1], offset: ?>, "HBM">

    %c_view = ktdp.construct_memory_view %c2048, sizes: [64], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64xf16>
    %c_hbm = memref.memory_space_cast %c_view : memref<64xf16> to memref<64xf16, "HBM">
    %c = memref.reinterpret_cast %c_hbm to offset: [%c0], sizes: [64], strides: [1] : memref<64xf16, "HBM"> to memref<64xf16, strided<[1], offset: ?>, "HBM">

    ktdf.pipeline {
      %priv:4 = ktdf.private -> (!ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>, !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, !ktdf.token, !ktdf.token) {
        %fa = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>
        %fb = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %fa, %fb, %t0, %t1 : !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>, !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, !ktdf.token, !ktdf.token
      }

      // Stage 1 (load): data_transfer is nested inside scf.for over reduction dim.
      ktdf.stage depends_in(none) depends_out(%priv#2) {
        scf.for %m = %c0 to %c16 step %c1 {
          ktdf.data_transfer from %a[%m, %c0] size [1, 64] to %priv#0 size [64] : memref<16x64xf16, strided<[64, 1], offset: ?>, "HBM">, !ktdf.fifo.slot<"HBM" -> "SFP", 64xf16>
        } {loop_type = #ktdf.loop_type<reduction_loop>}
      }

      // Stage 2 (compute): read_from_fifo inside scf.for, write_to_fifo after.
      ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
        %alloc = memref.alloc() : memref<64xf16, "LX">
        %cst = arith.constant 0.000000e+00 : f16
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64xf16, "LX">)
        scf.for %m = %c0 to %c16 step %c1 {
          %row = ktdf.read_from_fifo %priv#0 : <"HBM" -> "SFP", 64xf16> -> memref<1x64xf16>
          linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%row : memref<1x64xf16>) outs(%alloc : memref<64xf16, "LX">) {
          ^bb0(%in: f16, %out: f16):
            %sum = arith.addf %in, %out : f16
            linalg.yield %sum : f16
          }
        } {loop_type = #ktdf.loop_type<reduction_loop>}
        ktdf.write_to_fifo %alloc, %priv#1 : memref<64xf16, "LX">, <"SFP" -> "HBM", 64xf16>
      } {applicable_units = ["SFP"]}

      // Stage 3 (store): direct data_transfer (not nested).
      ktdf.stage depends_in(%priv#3) depends_out(none) {
        ktdf.data_transfer from %priv#1 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFP" -> "HBM", 64xf16>, memref<64xf16, strided<[1], offset: ?>, "HBM">
      }
    }
    return
  }
}
