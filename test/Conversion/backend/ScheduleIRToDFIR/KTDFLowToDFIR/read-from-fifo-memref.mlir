// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// Verify that ktdf.read_from_fifo returning a memref (PR #16) lowers to
// dataflow.receive without crashing.

// CHECK-LABEL: func.func @read_from_fifo_memref_test
// CHECK: dataflow.program_unit
// CHECK: dataflow.receive {{.*}} : vector<64xf16>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @read_from_fifo_memref_test() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %l1lu_map = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]):index
    %l1lu_q   = uniform.query_map(map:%l1lu_map, key:%tile_id) : index
    %sfu_map  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1]):index
    %sfu_q    = uniform.query_map(map:%sfu_map,  key:%tile_id) : index

    ktdf_lowering.execute_on %l1lu_q, %sfu_q {
      // Allocate a FIFO slot from L1LU to SFU carrying 64xf16 elements.
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

      // SFU side: read_from_fifo returns memref<1x64xf16> (PR #16 form).
      ktdf_lowering.execute_on %sfu_q {
        %buf = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
      }
    }

    return
  }
}
