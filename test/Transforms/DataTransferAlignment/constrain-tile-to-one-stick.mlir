// RUN: dataflow-scheduler-opt --data-transfer-alignment --tile-size-selection %s | FileCheck %s

// Pins the assumption that a tiled stick dim gets exactly one stick per tile.
//
// data-transfer-alignment leaves the tile size of dim p (x's stride-1 dim) as
// a ktdf.tiling.reserve_size placeholder constrained to a multiple of one
// stick (64), and widens the copy assuming the tile is exactly one stick: the
// point loops over dim p run once and A and B hold one stick along p.
// tile-size-selection must then pick 64. It does today because it tries no
// tile size above max(min_value, its largest candidate).
//
// If tile-size-selection can pick more than one stick, this test fails:
// widenRestick must then walk the sticks of a tile with the point loops
// (upper bound tile / 64, stride 64 in slot p's linearize_index) and give A
// and B a dim of tile / 64.
//
// x is 128 x 64 with strides [1, 128]: dim p = 0 holds two sticks.

// CHECK-LABEL: func.func @tiled
// CHECK-NOT:     ktdf.tiling.reserve_size
// CHECK:         ktdp.construct_memory_view
// The tile size of dim p: 64, one stick.
// CHECK:         %[[TILE:.*]] = arith.constant 64 : index
// CHECK-NEXT:    %[[NT:.*]] = arith.ceildivui %c128, %[[TILE]] : index
// CHECK:         scf.for %[[T:.*]] = %c0 to %[[NT]] step %c1 {
// CHECK:           scf.for %[[I:.*]] = %c0 to %c1 step %c1 {
// CHECK-NEXT:        %[[ROW:.*]] = ktdf.tiling.linearize_index [%[[T]] : %[[TILE]]], [%[[I]] : %c1] : index
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[%[[ROW]], %c0 * 64] size [64, 64]


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
