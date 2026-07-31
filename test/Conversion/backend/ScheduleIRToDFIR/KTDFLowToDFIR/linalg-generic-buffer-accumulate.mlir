// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// Verify that a buffer-semantics linalg.generic with a reduction iterator and
// arith.addf body is lowered to:
//   agen.vector_load  (input row)
//   agen.vector_load  (accumulator)
//   vectorchain.binary {binary_op = add}
//   agen.vector_store (accumulator)
// and that the linalg.generic itself is gone.

// CHECK: #[[$SET_3D:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET_2D:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL: func.func private @"local-schedule-0"
// CHECK:         dataflow.program_unit
// CHECK:           dataflow.program_unit
// CHECK:             dataflow.program_unit
// CHECK:               %[[VIEW:.*]] = dataflow.get_logical_memory_view
// CHECK:               %[[RECV:.*]] = dataflow.receive
// CHECK:               %[[ROW:.*]] = builtin.unrealized_conversion_cast %[[RECV]]
// CHECK:               %[[IN_VEC:.*]] = agen.vector_load %[[ROW]]
// CHECK-SAME:            {load_order = {{.*}}, load_set = #[[$SET_3D]]}
// CHECK-SAME:            : memref<1x1x64xf16>, vector<64xf16>
// CHECK:               %[[ACC_VEC:.*]] = agen.vector_load %[[VIEW]]
// CHECK-SAME:            {load_order = {{.*}}, load_set = #[[$SET_2D]]}
// CHECK-SAME:            : memref<1x64xf16>, vector<64xf16>
// CHECK:               %[[SUM:.*]] = vectorchain.binary %[[IN_VEC]], %[[ACC_VEC]]
// CHECK-SAME:            {binary_op = #vectorchain<binary_operator add>
// CHECK:               agen.vector_store %[[SUM]], %[[VIEW]]
// CHECK-SAME:            {store_order = {{.*}}, store_set = #[[$SET_2D]]}
// CHECK-NOT:           linalg.generic

#lrfreg = {kind = "lrfreg", size = 65536 : i64, ktdf_arch.features = {ktdf_arch.feature.local_to_compute}}
#SFP = {kind = "SFP", ktdf_arch.features = {ktdf_arch.feature.compute, ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f16 = 64 : i64>}}}
#LXLU = {kind = "LXLU", ktdf_arch.features = {ktdf_arch.feature.simd = {splat, zero_pad}}, load_store}
#LXSU = {kind = "LXSU", load_store}
#core = {kind = "core"}
#HBM = {kind = "HBM", size = 1073741824 : i64}
#LX = {kind = "LX", size = 1048576 : i64}
#L3LU = {dataflow_scheduler.double_buffer_last, kind = "L3LU", load_store}
#L3SU = {dataflow_scheduler.double_buffer_last, kind = "L3SU", load_store}

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

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
      %addr = arith.constant 512 : index
      %sfp0 = dataflow.get_unit {core = 0 : i32, name = "C0-SFP", type = "SFP"} : index
      %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-LXLU", type = "LXLU"} : index
      dataflow.program_unit iter_arg : %arg0 -> (%sfp0) : {
        dataflow.program_unit iter_arg : %arg1 -> (%sfp0) : {
          %lrfreg_unit = dataflow.get_unit {core = 0 : i32, name = "C0-lrfreg", type = "lrfreg"} : index
          %map_lrfreg = uniform.def_immutable_mapping([%sfp0 -> %lrfreg_unit]):index
          %unit_lrfreg = uniform.query_map(map:%map_lrfreg, key:%arg1) : index
          %acc_view = dataflow.get_logical_memory_view %unit_lrfreg, %addr {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          %row_vec = dataflow.receive %l1lu0 : vector<64xf16>
          %row_memref = builtin.unrealized_conversion_cast %row_vec : vector<64xf16> to memref<1x1x64xf16>
          linalg.generic {
            indexing_maps = [#map_in, #map_out],
            iterator_types = ["parallel", "reduction", "parallel"]
          } ins(%row_memref : memref<1x1x64xf16>)
            outs(%acc_view : memref<1x64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %r = arith.addf %in, %out : f16
            linalg.yield %r : f16
          }
        }
      }
      return
    }
  }
}
