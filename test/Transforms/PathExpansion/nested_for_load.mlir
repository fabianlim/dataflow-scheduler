// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>



// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @local_schedule_0() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_7:.*]] = arith.constant 2048 : index
// CHECK-NEXT:     %[[VAL_8:.*]] = ktdp.construct_memory_view %[[VAL_6]], sizes: [16, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = {{.*}}<HBM>} : memref<16x64xf16>
// CHECK-NEXT:     %[[VAL_9:.*]] = memref.memory_space_cast %[[VAL_8]] : memref<16x64xf16> to memref<16x64xf16, "DDR">
// CHECK-NEXT:     %[[VAL_10:.*]] = memref.reinterpret_cast %[[VAL_9]] to offset: {{\[}}%[[VAL_2]]], sizes: [16, 64], strides: [64, 1] : memref<16x64xf16, "DDR"> to memref<16x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[VAL_11:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [64], strides: [1] {coordinate_set = #[[$ATTR_2]], memory_space = {{.*}}<HBM>} : memref<64xf16>
// CHECK-NEXT:     %[[VAL_12:.*]] = memref.memory_space_cast %[[VAL_11]] : memref<64xf16> to memref<64xf16, "DDR">
// CHECK-NEXT:     %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_12]] to offset: {{\[}}%[[VAL_2]]], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[VAL_14:.*]]:8 = ktdf.private -> (memref<1x64xf16, "L1">, memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[VAL_15:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:         %[[VAL_16:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:         %[[VAL_17:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:         %[[VAL_18:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:         %[[VAL_19:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[VAL_20:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[VAL_21:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[VAL_22:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] : memref<1x64xf16, "L1">, memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_23:.*]]#4) {
// CHECK-NEXT:         scf.for %[[VAL_24:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_10]]{{\[}}%[[VAL_24]], 0] size [1, 64] to %[[VAL_23]]#0[0, 0] size [1, 64] : memref<16x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:         } {loop_type = {{.*}}<reduction_loop>}
// CHECK-NEXT:       } {applicable_units = ["MNILU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_25:.*]]#4) depends_out(%[[VAL_25]]#5) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_25]]#0[0, 0] size [1, 64] to %[[VAL_25]]#2 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_26:.*]]#5) depends_out(%[[VAL_26]]#6) {
// CHECK-NEXT:         %[[VAL_27:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:         %[[VAL_28:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:         linalg.fill ins(%[[VAL_28]] : f16) outs(%[[VAL_27]] : memref<64xf16, "L1">)
// CHECK-NEXT:         scf.for %[[VAL_29:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK-NEXT:           %[[VAL_30:.*]] = ktdf.read_from_fifo %[[VAL_26]]#2 : <"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
// CHECK-NEXT:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel"]} ins(%[[VAL_30]] : memref<1x64xf16>) outs(%[[VAL_27]] : memref<64xf16, "L1">) {
// CHECK-NEXT:           ^bb0(%[[VAL_31:.*]]: f16, %[[VAL_32:.*]]: f16):
// CHECK-NEXT:             %[[VAL_33:.*]] = arith.addf %[[VAL_31]], %[[VAL_32]] : f16
// CHECK-NEXT:             linalg.yield %[[VAL_33]] : f16
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = {{.*}}<reduction_loop>}
// CHECK-NEXT:         ktdf.write_to_fifo %[[VAL_27]], %[[VAL_26]]#3 : memref<64xf16, "L1">, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_34:.*]]#6) depends_out(%[[VAL_34]]#7) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_34]]#3 size [64] to %[[VAL_34]]#1[0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_35:.*]]#7) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_35]]#1[0] size [64] to %[[VAL_13]][0] size [64] : memref<64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:       } {applicable_units = ["MNISU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Test: PathExpansion handles data_transfer nested inside scf.for in a load stage
// and read_from_fifo/write_to_fifo nested inside scf.for in the compute stage.
// This reproduces the reduction-loop pattern where Stage 1 iterates over the
// reduction dimension and streams one row per iteration.
//
// Key invariants:
//   - Abstract "DDR"->"SFU" / "SFU"->"DDR" FIFOs are gone; replaced by L1
//     buffers and physical L1LU/L1SU FIFOs.
//   - Stage 1 (MNILU): scf.for body carries data_transfer to the L1 buffer.
//   - Stage 3 (SFU): read_from_fifo uses "L1LU"->"SFU", write_to_fifo uses
//     "SFU"->"L1SU".




#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  func.func @local_schedule_0() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    %a_view = ktdp.construct_memory_view %c1024, sizes: [16, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x64xf16>
    %a_hbm = memref.memory_space_cast %a_view : memref<16x64xf16> to memref<16x64xf16, "DDR">
    %a = memref.reinterpret_cast %a_hbm to offset: [%c0], sizes: [16, 64], strides: [64, 1] : memref<16x64xf16, "DDR"> to memref<16x64xf16, strided<[64, 1], offset: ?>, "DDR">

    %c_view = ktdp.construct_memory_view %c2048, sizes: [64], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64xf16>
    %c_hbm = memref.memory_space_cast %c_view : memref<64xf16> to memref<64xf16, "DDR">
    %c = memref.reinterpret_cast %c_hbm to offset: [%c0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">

    ktdf.pipeline {
      %priv:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
        %fa = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %fb = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %fa, %fb, %t0, %t1 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
      }

      // Stage 1 (load): data_transfer is nested inside scf.for over reduction dim.
      ktdf.stage depends_in(none) depends_out(%priv#2) {
        scf.for %m = %c0 to %c16 step %c1 {
          ktdf.data_transfer from %a[%m, %c0] size [1, 64] to %priv#0 size [64] : memref<16x64xf16, strided<[64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        } {loop_type = #ktdf.loop_type<reduction_loop>}
      }

      // Stage 2 (compute): read_from_fifo inside scf.for, write_to_fifo after.
      ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
        %alloc = memref.alloc() : memref<64xf16, "L1">
        %cst = arith.constant 0.000000e+00 : f16
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64xf16, "L1">)
        scf.for %m = %c0 to %c16 step %c1 {
          %row = ktdf.read_from_fifo %priv#0 : <"DDR" -> "SFU", 64xf16> -> memref<1x64xf16>
          linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%row : memref<1x64xf16>) outs(%alloc : memref<64xf16, "L1">) {
          ^bb0(%in: f16, %out: f16):
            %sum = arith.addf %in, %out : f16
            linalg.yield %sum : f16
          }
        } {loop_type = #ktdf.loop_type<reduction_loop>}
        ktdf.write_to_fifo %alloc, %priv#1 : memref<64xf16, "L1">, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}

      // Stage 3 (store): direct data_transfer (not nested).
      ktdf.stage depends_in(%priv#3) depends_out(none) {
        ktdf.data_transfer from %priv#1 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, strided<[1], offset: ?>, "DDR">
      }
    }
    return
  }
}
