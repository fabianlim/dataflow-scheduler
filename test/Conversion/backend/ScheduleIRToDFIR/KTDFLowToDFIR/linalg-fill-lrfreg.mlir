// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that linalg.fill with buffer semantics on an lrfreg-backed memref
// is lowered to agen.vector_store of a dense-zero vector.
//
// Checks:
//   - arith.constant dense<0.0> : vector<64xf16> is emitted
//   - agen.vector_store with that constant is emitted
//   - linalg.fill is gone

// CHECK: #[[$ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$SET:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL: func.func private @"local-schedule-0"
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// CHECK:         dataflow.program_unit
// CHECK:           dataflow.program_unit
// CHECK:             %[[VIEW:.*]] = dataflow.get_logical_memory_view
// CHECK:             agen.vector_store %[[ZERO]], %[[VIEW]][%[[C0]], %[[C0]]]
// CHECK-SAME:          {store_order = #[[$ID]], store_set = #[[$SET]]}
// CHECK-NOT:         linalg.fill

#lrfreg = {kind = "lrfreg", size = 65536 : i64, ktdf_arch.features = {ktdf_arch.feature.local_to_compute}}
#SFP = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
#LXLU = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
#LXSU = {kind = "LXSU", load_store}
#core = {kind = "core"}
#HBM = {kind = "HBM", size = 1073741824 : i64}
#LX = {kind = "LX", size = 1048576 : i64}
#L3LU = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
#L3SU = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}

module {
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
      %cst = arith.constant 0.0 : f16
      %addr = arith.constant 512 : index
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-SFP", type = "SFP"} : index
      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
        %lrf_view = builtin.unrealized_conversion_cast %addr : index to memref<1x64xf16, "lrfreg">
        linalg.fill ins(%cst : f16) outs(%lrf_view : memref<1x64xf16, "lrfreg">)
      }
      return
    }
  }
}
