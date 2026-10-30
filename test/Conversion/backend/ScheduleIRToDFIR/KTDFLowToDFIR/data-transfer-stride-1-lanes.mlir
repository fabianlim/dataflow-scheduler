// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A transfer wider than one vector takes its lanes from each side's stride-1
// dimension, so each vector is contiguous in memory. The source x has strides
// [4096, 1, 64]: its lanes run along dim 1 and its time steps along dim 2,
// stick t of x (addresses 64 * t + 0..63) into row t of the row-major A.

// CHECK-DAG: #[[$LOAD_TIME:.+]] = affine_map<(d0) -> (0, 0, d0)>
// CHECK-DAG: #[[$STORE_TIME:.+]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK-DAG: #[[$LOAD_SET:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0, d2 == 0)>
// CHECK-DAG: #[[$STORE_SET:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-DAG: #[[$TIME_64:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL: func.func @lanes_along_stride_1
// CHECK-NOT:     scf.for
// CHECK:         agen.composite_load_and_store
// CHECK-NEXT:      time_symbols(), load_iv(%{{.*}}:vector<64xf16>)
// CHECK-NEXT:      load_set = #[[$LOAD_SET]], load_time_addr_map = #[[$LOAD_TIME]], {{.*}}store_set = #[[$STORE_SET]], store_time_addr_map = #[[$STORE_TIME]], {{.*}}time_set = #[[$TIME_64]]}
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @lanes_along_stride_1() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %u = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %u]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %raw = memref.alloc() : memref<4096xf16, "DDR">
    %cast = memref.reinterpret_cast %raw to offset: [0], sizes: [1, 64, 64], strides: [4096, 1, 64] : memref<4096xf16, "DDR"> to memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">
    %a = memref.alloc() : memref<1x1x64x64xf16, "L1">
    ktdf_lowering.execute_on %unit {
      ktdf_lowering.execute_on %unit {
        scf.for %arg0 = %c0 to %c1 step %c1 {
          ktdf.data_transfer from %cast[%c0, %arg0, %c0 * 64] size [1, 64, 64] to %a[%arg0, 0, 0, 0] size [1, 1, 64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">, memref<1x1x64x64xf16, "L1">
        }
      }
    }
    return
  }
}
