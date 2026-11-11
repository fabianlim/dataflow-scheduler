// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// x is 1 x 64 x 64 with strides [4096, 1, 64]: its stride-1 dim is p = 1. Every
// transfer moves one stick along q = 2, where x's stride is 64. Only the load
// and the hop are rewritten.

// CHECK-LABEL: func.func @local_schedule_0()

// A and B keep their shapes.
// CHECK:      %[[PRIV:.*]]:4 = ktdf.private -> (memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {

// MNILU: the loop picks stick I of x along p, which is contiguous.
// CHECK:      ktdf.stage
// CHECK-NEXT:   scf.for %[[I:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:     ktdf.data_transfer from %{{.*}}[%c0, 0, %c0 * 64 + %[[I]]] size [1, 64, 1] to %[[PRIV]]#0[%[[I]], 0, 0, 0] size [1, 1, 1, 64]

// Hop: a loop C over q; splat element I of A[C], extract into lane C of B[I].
// CHECK:      ktdf.stage
// CHECK-NEXT:   scf.for %[[I:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:     scf.for %[[C:.*]] = %c0 to %c64 step %c1 {
// CHECK:            ktdf.data_transfer from %[[PRIV]]#0[%[[C]], 0, 0, %[[I]]] size [1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:            ktdf.data_transfer from %{{.*}} size [64] to %[[PRIV]]#1[%[[I]], 0, 0, %[[C]]] size [1, 1, 1, 1] {transfer_mode = "extract"}

// MNISU: untouched.
// CHECK:      ktdf.stage
// CHECK-NEXT:   scf.for %[[I:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:     ktdf.data_transfer from %[[PRIV]]#1[%[[I]], 0, 0, 0] size [1, 1, 1, 64] to %{{.*}}[%c0, %[[I]], %c0 * 64] size [1, 1, 64]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  module {
    func.func @identity_0() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c162304 = arith.constant 162304 : index
      %c64000 = arith.constant 64000 : index

      %0 = ktdp.construct_memory_view %c64000, sizes: [1, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16, strided<[4096, 1, 64]>>
      %1 = ktdp.construct_memory_view %c162304, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<1x64x64xf16, strided<[4096, 1, 64]>> to memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">
      %cast = memref.cast %memspacecast : memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR"> to memref<1x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64x64xf16> to memref<1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 64, 64], strides: [4096, 64, 1] : memref<1x64x64xf16, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">

      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<64x1x1x64xf16, "L1">
          %alloc_2 = memref.alloc() : memref<64x1x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_2, %3, %4 : memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%c0, %arg0, %c0 * 64] size [1, 1, 64] to %2#0[%arg0, 0, 0, 0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">, memref<64x1x1x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<64x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                ktdf.write_to_fifo %4, %3#1 : tensor<1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                ktdf.data_transfer from %3#1 size [64] to %2#1[%arg0, 0, 0, 0] size [1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x1x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0, 0, 0] size [1, 1, 1, 64] to %cast_1[%c0, %arg0, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x1x64xf16, "L1">, memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
