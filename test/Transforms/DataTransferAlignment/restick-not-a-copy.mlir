// RUN: dataflow-scheduler-opt --data-transfer-alignment -split-input-file -verify-diagnostics %s

// A strided load is re-sticked in place only when it feeds a plain copy to a
// global view and nothing else shares the copy's point loop over the stick
// dim. Each case below is align-transfers-q-not-last.mlir plus a second
// operand w laid out like y.

// x + y^T: the strided load comes first.

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
            // expected-error @+1 {{the load does not feed a plain copy}}
            ktdf.data_transfer from %memspacecast[%c0, %c0 * 64, %arg0] size [1, 64, 1] to %2#0[%arg0, 0, 0, 0] size [1, 1, 64, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">, memref<64x1x64x1xf16, "L1">
            ktdf.data_transfer from %wc[%c0, %c0 * 64, %arg0] size [1, 64, 1] to %2#4[%arg0, 0, 0, 0] size [1, 1, 64, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">, memref<64x1x64x1xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.pipeline {
              %3:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
                %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7, %f : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 1, 64, 1] to %3#0 size [64] : memref<64x1x64x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                ktdf.data_transfer from %2#4[%arg0, 0, 0, 0] size [1, 1, 64, 1] to %3#4 size [64] : memref<64x1x64x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64x1xf16>
                %w4 = ktdf.read_from_fifo %3#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64x1xf16>
                %s = arith.addf %4, %w4 : tensor<1x64x1xf16>
                ktdf.write_to_fifo %s, %3#1 : tensor<1x64x1xf16>, <"SFU" -> "L1SU", 64xf16>
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

// -----

// x + y^T with the contiguous load first: every load is checked, not just
// the first.

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
            ktdf.data_transfer from %wc[%c0, %c0 * 64, %arg0] size [1, 64, 1] to %2#4[%arg0, 0, 0, 0] size [1, 1, 64, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">, memref<64x1x64x1xf16, "L1">
            // expected-error @+1 {{the load does not feed a plain copy}}
            ktdf.data_transfer from %memspacecast[%c0, %c0 * 64, %arg0] size [1, 64, 1] to %2#0[%arg0, 0, 0, 0] size [1, 1, 64, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">, memref<64x1x64x1xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.pipeline {
              %3:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
                %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7, %f : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 1, 64, 1] to %3#0 size [64] : memref<64x1x64x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                ktdf.data_transfer from %2#4[%arg0, 0, 0, 0] size [1, 1, 64, 1] to %3#4 size [64] : memref<64x1x64x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64x1xf16>
                %w4 = ktdf.read_from_fifo %3#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64x1xf16>
                %s = arith.addf %4, %w4 : tensor<1x64x1xf16>
                ktdf.write_to_fifo %s, %3#1 : tensor<1x64x1xf16>, <"SFU" -> "L1SU", 64xf16>
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

// -----

// A pure copy, plus an unrelated load in the copy's point loop over the
// stick dim: removing the loop would drop the load's other 63 trips.

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
            // expected-error @+1 {{shares a point loop over the stick dim}}
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
