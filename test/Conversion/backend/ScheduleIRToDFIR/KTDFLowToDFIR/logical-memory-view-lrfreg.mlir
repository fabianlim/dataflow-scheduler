// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Test: Source B — unrealized_conversion_cast from index to lrfreg memref is
// lowered to dataflow.get_logical_memory_view. lrfreg is at depth-2 in the
// device tree (below the LX scratchpad, inside a corelet group), so
// isBelowScratchPad returns true for it and it must take the per-core uniform
// map path, just like L1 scratchpad.
//
// Checks:
//   - lrfreg get_unit emitted at func level (one per core)
//   - uniform.def_immutable_mapping + uniform.query_map emitted inside program_unit
//   - index operand of cast becomes start_address of get_logical_memory_view
//   - layout_map is row-major linearization of shape <1x64> -> (d0*64+d1)
//   - result type is plain memref (no memory space attr)
//   - linalg.fill uses the resolved plain-memref view
//   - linalg.generic uses the resolved plain-memref view (ins and outs)
//   - original unrealized_conversion_cast is gone

// CHECK: #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK: #[[$ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func private @"local-schedule-0"() attributes {grid = [1]} {
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[ADDR:.*]] = arith.constant 512 : index
// CHECK:         %[[SFP:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFP", type = "SFP"} : index
// CHECK:         %[[LRFREG:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-lrfreg", type = "lrfreg"} : index
// CHECK:         dataflow.program_unit iter_arg : {{.*}} -> (%[[SFP]]) : {
// CHECK:           dataflow.program_unit iter_arg : %[[KEY:.*]] -> (%[[SFP]]) : {
// CHECK:             %[[MAP:.*]] = uniform.def_immutable_mapping([%[[SFP]] -> %[[LRFREG]]]):index
// CHECK:             %[[UNIT:.*]] = uniform.query_map(map:%[[MAP]], key:%[[KEY]]) : index
// CHECK:             %[[VIEW:.*]] = dataflow.get_logical_memory_view %[[UNIT]], %[[ADDR]] {layout_map = #[[$LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK:             linalg.fill ins(%[[CST]] : f16) outs(%[[VIEW]] : memref<1x64xf16>)
// CHECK:             linalg.generic {indexing_maps = [#[[$ID]], #[[$ID]]], iterator_types = ["parallel", "parallel"]} ins(%[[VIEW]] : memref<1x64xf16>) outs(%[[VIEW]] : memref<1x64xf16>)
// CHECK-NOT:         unrealized_conversion_cast

// Verify the original cast is completely gone.
// CHECK-NOT: unrealized_conversion_cast

#lrfreg = {kind = "lrfreg", size = 65536 : i64, ktdf_arch.features = {ktdf_arch.feature.local_to_compute}}
#SFP = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
#LXLU = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
#LXSU = {kind = "LXSU", load_store}
#core = {kind = "core"}
#HBM = {kind = "HBM", size = 1073741824 : i64}
#LX = {kind = "LX", size = 1048576 : i64}
#L3LU = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
#L3SU = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}

#map_2d = affine_map<(d0, d1) -> (d0, d1)>

module {
  // Device with lrfreg at depth-2: HBM (root) -> LX (scratchpad) -> lrfreg
  ktdf_arch.device @spyre_1core attributes {
    mem_space_mapping = #ktdf_arch.map<
      #ktdp.spyre_memory_space<HBM> = "HBM",
      #ktdp.spyre_memory_space<LX>  = "LX"
    >
  } {
    %memory = memory #HBM
    group #core share(%memory) {
      %memory_0 = memory #LX
      %exec_unit = exec_unit #L3LU
      %exec_unit_2 = exec_unit #L3SU
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory to %exec_unit : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit to %memory_0 : exec_unit, memory
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %memory_0 to %exec_unit_2 : memory, exec_unit
      datapath {ktdf_arch.transfer_granularity = array<i64: 64>} %exec_unit_2 to %memory : exec_unit, memory
      // Sub-core group: lrfreg lives here (depth-2 relative to HBM root)
      group share(%memory_0) {
        %lrfreg_0 = memory #lrfreg
        %exec_unit_3 = exec_unit #SFP
        %exec_unit_4 = exec_unit #LXLU
        %exec_unit_5 = exec_unit #LXSU
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %memory_0 to %exec_unit_4 : memory, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_4 to %exec_unit_3 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}, ktdf_arch.transfer_granularity = array<i64: 64, 2>} %exec_unit_3 to %exec_unit_5 : exec_unit, exec_unit
        datapath {ktdf_arch.features = {ktdf_arch.feature.queue = {depth = 16 : i64, ordered}}} %exec_unit_5 to %memory_0 : exec_unit, memory
      }
    }
  }
  module {
    func.func @test() attributes {grid = [1]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.0 : f16
      %addr = arith.constant 512 : index
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-SFP", type = "SFP"} : index
      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
        // Source B: unrealized_conversion_cast(index -> memref<1x64xf16, "lrfreg">)
        // produced by pass 18 AddressAssignment for a memref.alloc with lrfreg space.
        %lrf_view = builtin.unrealized_conversion_cast %addr : index to memref<1x64xf16, "lrfreg">
        // linalg.fill zeros the accumulator register file buffer.
        linalg.fill ins(%cst : f16) outs(%lrf_view : memref<1x64xf16, "lrfreg">)
        // linalg.generic reads from and writes to the same lrfreg buffer.
        linalg.generic {
          indexing_maps = [#map_2d, #map_2d],
          iterator_types = ["parallel", "parallel"]
        } ins(%lrf_view : memref<1x64xf16, "lrfreg">)
          outs(%lrf_view : memref<1x64xf16, "lrfreg">) {
        ^bb0(%in: f16, %out: f16):
          %sum = arith.addf %in, %out : f16
          linalg.yield %sum : f16
        }
        memref.dealloc %lrf_view : memref<1x64xf16, "lrfreg">
      }
      return
    }
  }
}
