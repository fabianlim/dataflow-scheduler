// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(address-assignment,ktdflowering-to-dfir)" %s | FileCheck %s

// An address-assigned SFU_REG buffer reaching DFIR.
//
// In sample_device, #SFU_REG is declared inside `group { kind = "SFU" }
// share()`. Because that group shares nothing, the memory-tree walk keeps the
// enclosing sub-core's L1 as its parent, so SFU_REG lands at depth 2: neither
// global (a memory-tree root, e.g. DDR) nor per-core scratchpad (a depth-1
// child of a root, e.g. L1). Depth >= 2 means compute-unit-local, so there is
// no memory unit to select and the view builder builds no logical memory view.
// The unrealized_conversion_cast that address assignment emitted survives into
// the operation lowerings, which resolve it against the compute unit owning the
// register file.
//
// Two buffers are filled so that the second one pins the *assigned* offset
// (128 bytes past the first) rather than a hardcoded zero.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   func.func @fill_sfu_reg() attributes {grid = [2]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFP", type = "SFP"} : index
// CHECK-NEXT:     %[[VAL_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-SFP", type = "SFP"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_4:.*]] -> (%[[VAL_2]], %[[VAL_3]]) : {
// CHECK-NEXT:       %[[VAL_5:.*]] = dataflow.get_local_unit %[[VAL_4]] {name = "lrfreg"} : index
// CHECK-NEXT:       %[[VAL_6:.*]] = dataflow.get_logical_memory_view %[[VAL_5]], %[[VAL_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:       %[[VAL_7:.*]] = vectorchain.constant_bitstream {value = [0x0]} : vector<1xf16>
// CHECK-NEXT:       %[[VAL_8:.*]] = vectorchain.shuffle %[[VAL_7]] {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[VAL_8]], %[[VAL_6]]{{\[}}%[[VAL_1]], %[[VAL_1]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_2]]} : memref<1x64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[VAL_9:.*]] = dataflow.get_local_unit %[[VAL_4]] {name = "lrfreg"} : index
// CHECK-NEXT:       %[[VAL_10:.*]] = dataflow.get_logical_memory_view %[[VAL_9]], %[[VAL_0]] {layout_map = #[[$ATTR_0]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:       %[[VAL_11:.*]] = vectorchain.constant_bitstream {value = [0x0]} : vector<1xf16>
// CHECK-NEXT:       %[[VAL_12:.*]] = vectorchain.shuffle %[[VAL_11]] {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[VAL_12]], %[[VAL_10]]{{\[}}%[[VAL_1]], %[[VAL_1]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_2]]} : memref<1x64xf16>, vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @fill_sfu_reg() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFP", type = "SFP"} : index
    %1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-SFP", type = "SFP"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    ktdf_lowering.execute_on %unit {
      %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %alloc2 = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      linalg.fill ins(%zero : f16) outs(%alloc2 : memref<1x64xf16, "SFU_REG">)
    }
    return
  }
}
