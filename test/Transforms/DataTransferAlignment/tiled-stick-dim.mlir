// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// A tiled loop over the stick dim. x is 128 x 64 with strides [1, 128], so its
// stride-1 dim p = 0 holds two sticks, and loop tiling gave the loop over it a
// ktdf.tiling.reserve_size tile size.
//
// A re-stick tile is one stick along p, so there is no tile size to choose:
// the placeholder becomes 64, and the tile loop walks the two sticks. The
// point loops over p, which would run once, are removed: each stage moves the
// whole stick at point 0.

// CHECK-LABEL: func.func @tiled
// CHECK-NOT:     ktdf.tiling.reserve_size
// CHECK:         %[[NT:.*]] = arith.ceildivui %c128, %c64 : index
// CHECK:         scf.for %[[T:.*]] = %c0 to %[[NT]] step %c1 {
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[ROW:.*]] = ktdf.tiling.linearize_index [%[[T]] : %c64], [%c0 : %c1] : index
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[%[[ROW]], %c0 * 64] size [64, 64] to %{{.*}}[%c0, 0, 0] size [1, 64, 64]
// CHECK:           ktdf.stage
// CHECK-NEXT:        scf.for %[[R:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:          scf.for %[[C:.*]] = %c0 to %c64 step %c1 {
// CHECK:                 {transfer_mode = "splat"}
// CHECK:                 {transfer_mode = "extract"}
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[ROW:.*]] = ktdf.tiling.linearize_index [%[[T]] : %c64], [%c0 : %c1] : index
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[%c0, 0, 0] size [1, 64, 64] to %{{.*}}[%[[ROW]], %c0 * 64] size [64, 64]

#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @tiled(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [128, 64], strides: [1, 128] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16, strided<[1, 128]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [128, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16>
  %xd = memref.memory_space_cast %x : memref<128x64xf16, strided<[1, 128]>> to memref<128x64xf16, strided<[1, 128]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<128x64xf16> to memref<128x64xf16, "DDR">
  %tile = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
  %ntiles = arith.ceildivui %c128, %tile : index
  scf.for %t = %c0 to %ntiles step %c1 {
    %n = ktdf.tiling.derive_size [%t : %tile], total_size = %c128 : index
    ktdf.pipeline {
      %buf:2 = ktdf.private -> (memref<?x1x64xf16, "L1">, memref<?x1x64xf16, "L1">) {
        %a = memref.alloc(%tile) : memref<?x1x64xf16, "L1">
        %b = memref.alloc(%tile) : memref<?x1x64xf16, "L1">
        ktdf.private_yield %a, %b : memref<?x1x64xf16, "L1">, memref<?x1x64xf16, "L1">
      }
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %n step %c1 {
          %row = ktdf.tiling.linearize_index [%t : %tile], [%i : %c1] : index
          ktdf.data_transfer from %xd[%row, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<128x64xf16, strided<[1, 128]>, "DDR">, memref<?x1x64xf16, "L1">
        }
      }
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %n step %c1 {
          ktdf.pipeline {
            %f = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
              %s = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
              ktdf.private_yield %s : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            }
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %buf#0[%i, 0, 0] size [1, 1, 64] to %f size [64] : memref<?x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            }
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %f size [64] to %buf#1[%i, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<?x1x64xf16, "L1">
            }
          }
        }
      }
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %n step %c1 {
          %row = ktdf.tiling.linearize_index [%t : %tile], [%i : %c1] : index
          ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%row, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x1x64xf16, "L1">, memref<128x64xf16, "DDR">
        }
      }
    }
  }
  return
}
