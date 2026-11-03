// Buffer expansion must see a buffer written by a ktdf.ind_data_transfer.
//
// A gather reads each table row as 4 sticks, one per iteration of the inner
// loop %arg4, into an L1 staging buffer (memref<64xf16>) that the L1LU stage
// then forwards. Sinking %arg4 into the stages makes that buffer shared across
// the 4 iterations, so it has to gain a leading dimension of 4, indexed by
// %arg4 on both sides. Before ind_data_transfer was handled, only the output
// staging buffer (written by an ordinary data_transfer) was expanded, every
// iteration overwrote dir_dst[0], and only the last stick survived.
//
// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s

// CHECK-LABEL: func.func @local_schedule_0
// CHECK:         ktdf.private -> (memref<4x64xf16, "L1">, memref<4x1x1x64xf16, "L1">
// CHECK:         scf.for %[[G:.*]] = %{{.*}} to %{{.*}} step
// CHECK:           %[[OFF:.*]] = arith.subi %[[G]], %{{.*}} : index
// CHECK:           %[[SLOT:.*]] = arith.divsi %[[OFF]], %{{.*}} : index
// CHECK:           ktdf.ind_data_transfer
// CHECK:            dir_src = %{{.*}}[0, %[[G]], 0] size [1, 1, 64]
// CHECK:            dir_dst = %{{.*}}[%[[SLOT]], %{{.*}}] size [1, 64]
// CHECK-NEXT:       memref<4x64xf16, "L1">
// CHECK:         ktdf.data_transfer from %{{.*}}[%{{.*}}, 0] size [1, 64] to %{{.*}} size [64] : memref<4x64xf16, "L1">

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")
    func.func @local_schedule_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %0 = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1] {coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>, memory_space = "IAB"} : memref<32xi32, "IAB">
      %1 = ktdp.construct_memory_view %c0, sizes: [32], strides: [1] {coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<32xi32, #ktdp.memory_space<global>>
      %2 = ktdp.construct_memory_view %c0, sizes: [64, 4, 64], strides: [64, 4096, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<64x4x64xf16>
      %memspacecast = memref.memory_space_cast %2 : memref<64x4x64xf16> to memref<64x4x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [64, 4, 64], strides: [64, 4096, 1] : memref<64x4x64xf16, "DDR"> to memref<64x4x64xf16, strided<[64, 4096, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<64x4x64xf16, strided<[64, 4096, 1]>, "DDR"> to memref<64x4x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">
      %reinterpret_cast_0 = memref.reinterpret_cast %0 to offset: [0], sizes: [32], strides: [1] : memref<32xi32, "IAB"> to memref<32xi32, strided<[1]>, "IAB">
      %cast_1 = memref.cast %reinterpret_cast_0 : memref<32xi32, strided<[1]>, "IAB"> to memref<32xi32, strided<[1], offset: ?>, "IAB">
      %3 = ktdp.construct_memory_view %arg2, sizes: [32, 4, 64], strides: [64, 2048, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<32x4x64xf16>
      %memspacecast_2 = memref.memory_space_cast %3 : memref<32x4x64xf16> to memref<32x4x64xf16, "DDR">
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %4 = arith.cmpi eq, %arg3, %c0 : index
        scf.for %arg4 = %c0 to %c4 step %c1 {
          ktdf.pipeline {
            %5:8 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
              %alloc = memref.alloc() : memref<64xf16, "L1">
              %6 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
              %alloc_3 = memref.alloc() : memref<1x1x64xf16, "L1">
              %8 = ktdf.create_token : !ktdf.token
              %9 = ktdf.create_token : !ktdf.token
              %10 = ktdf.create_token : !ktdf.token
              %11 = ktdf.create_token : !ktdf.token
              ktdf.private_yield %alloc, %6, %7, %alloc_3, %8, %9, %10, %11 : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
            }
            ktdf.stage depends_in(none) depends_out(%5#4) {
              scf.if %4 {
                %memspacecast_3 = memref.memory_space_cast %1 : memref<32xi32, #ktdp.memory_space<global>> to memref<32xi32, "DDR">
                ktdf.data_transfer from %memspacecast_3[0] size [32] to %0[0] size [32] {dataflow_scheduler.throttle = 1 : i64} : memref<32xi32, "DDR">, memref<32xi32, "IAB">
              }
              ktdf.ind_data_transfer
               ind_src = %cast_1[%arg3]
               dir_src = %cast[0, %arg4, 0] size [1, 1, 64]
               ind_dst = none
               dir_dst = %5#0[%c0] size [64]
               {dataflow_scheduler.throttle = 64 : i64} : memref<32xi32, strided<[1], offset: ?>, "IAB">, memref<64x4x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, memref<64xf16, "L1">
            } {applicable_units = ["MNILU"]}
            ktdf.stage depends_in(%5#4) depends_out(%5#5) {
              ktdf.data_transfer from %5#0[0] size [64] to %5#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
            } {applicable_units = ["L1LU"]}
            ktdf.stage depends_in(%5#5) depends_out(%5#6) {
              %6 = ktdf.read_from_fifo %5#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
              ktdf.write_to_fifo %6, %5#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
            } {applicable_units = ["SFU"]}
            ktdf.stage depends_in(%5#6) depends_out(%5#7) {
              ktdf.data_transfer from %5#2 size [64] to %5#3[0, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x64xf16, "L1">
            } {applicable_units = ["L1SU"]}
            ktdf.stage depends_in(%5#7) depends_out(none) {
              ktdf.data_transfer from %5#3[0, 0, 0] size [1, 1, 64] to %memspacecast_2[%arg3, %arg4, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x1x64xf16, "L1">, memref<32x4x64xf16, "DDR">
            } {applicable_units = ["MNISU"]}
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
      return
    }
}
