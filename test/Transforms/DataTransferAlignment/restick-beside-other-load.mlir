// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// align-transfers-q-not-last.mlir plus an unrelated contiguous load from w in
// the strided load's loop over the stick dim. No loop is folded, so the other
// load keeps all of its trips and is left as it is.

// CHECK-LABEL: func.func @local_schedule_0()
// CHECK:      ktdf.stage
// CHECK-NEXT:   scf.for %[[I:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:     ktdf.data_transfer from %{{.*}}[%c0, %c0 * 64, %[[I]]] size [1, 64, 1] to %{{.*}}#4[%[[I]], 0, 0, 0] size [1, 1, 64, 1]
// CHECK-NEXT:     ktdf.data_transfer from %{{.*}}[%c0, %c0 * 64 + %[[I]], 0] size [1, 1, 64] to %{{.*}}#0[%[[I]], 0, 0, 0] size [1, 1, 64, 1]
// CHECK:        {transfer_mode = "splat"}
// CHECK:        {transfer_mode = "extract"}

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

      %0 = ktdp.construct_memory_view %c64000, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16, strided<[4096, 64, 1]>>
      %1 = ktdp.construct_memory_view %c162304, sizes: [1, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16, strided<[4096, 1, 64]>>
      %w = ktdp.construct_memory_view %c0, sizes: [1, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16, strided<[4096, 1, 64]>>
      %wc = memref.memory_space_cast %w : memref<1x64x64xf16, strided<[4096, 1, 64]>> to memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">
      %memspacecast = memref.memory_space_cast %0 : memref<1x64x64xf16, strided<[4096, 64, 1]>> to memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64x64xf16, strided<[4096, 1, 64]>> to memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">

      ktdf.pipeline {
        %2:5 = ktdf.private -> (memref<64x1x64x1xf16, "L1">, memref<64x1x64x1xf16, "L1">, !ktdf.token, !ktdf.token, memref<64x1x64x1xf16, "L1">) {
          %alloc_w = memref.alloc() : memref<64x1x64x1xf16, "L1">
          %alloc = memref.alloc() : memref<64x1x64x1xf16, "L1">
          %alloc_2 = memref.alloc() : memref<64x1x64x1xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_2, %3, %4, %alloc_w : memref<64x1x64x1xf16, "L1">, memref<64x1x64x1xf16, "L1">, !ktdf.token, !ktdf.token, memref<64x1x64x1xf16, "L1">
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %wc[%c0, %c0 * 64, %arg0] size [1, 64, 1] to %2#4[%arg0, 0, 0, 0] size [1, 1, 64, 1] : memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">, memref<64x1x64x1xf16, "L1">
            ktdf.data_transfer from %memspacecast[%c0, %c0 * 64, %arg0] size [1, 64, 1] to %2#0[%arg0, 0, 0, 0] size [1, 1, 64, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">, memref<64x1x64x1xf16, "L1">
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
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 1, 64, 1] to %3#0 size [64] : memref<64x1x64x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64x1xf16>
                ktdf.write_to_fifo %4, %3#1 : tensor<1x64x1xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                ktdf.data_transfer from %3#1 size [64] to %2#1[%arg0, 0, 0, 0] size [1, 1, 64, 1] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x1x64x1xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0, 0, 0] size [1, 1, 64, 1] to %memspacecast_0[%c0, %c0 * 64, %arg0] size [1, 64, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64x1xf16, "L1">, memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
