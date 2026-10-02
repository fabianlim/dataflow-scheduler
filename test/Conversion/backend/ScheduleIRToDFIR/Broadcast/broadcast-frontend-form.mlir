// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering %s | FileCheck %s --check-prefix=LOWERING
// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering --ktdflowering-to-dfir %s | FileCheck %s
// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering --ktdflowering-to-dfir %s | FileCheck %s --check-prefix=NOSYNC

// The broadcast relayout as the frontend writes it, on the 4-core ring
// device, through the scheduling steps it needs so far: path expansion maps
// the channel's stages to MNISU and MNILU, handle-cross-core-stages realizes
// the self-delivery of core 0 as a local copy L1LU -> SFU -> L1SU, and
// ktdf-to-ktdflowering materializes the units on the tiles of the leaf
// stages of each kind.
//
// The channel has the groups g = 0..1: producer core g sends its 64x64 tile
// to the consumer cores 2g and 2g+1, in messages of one 64-element stick,
// the throttle of its transfers. Core 1 is both the producer of group 1 and
// a consumer of group 0. The ring carries C(g) without P(g), so MNISU is on
// cores 0..1 and MNILU on cores 1..3; the local copy carries core 0's tile
// to core 0, so L1LU, SFU and L1SU are on core 0 only. Each kind's map from
// tile to unit lists exactly those tiles. The stages that wrap the channel
// and the local copy are not counted for the tiles; they execute on the
// units of the stages they wrap. The corelet path moves one vector per
// transfer, so the local copy is a loop over the 64 sticks around its
// pipeline, one stick per iteration.
//
// The token between the ring stages gets no signal: the producer stage
// writes no memory, only the fifo, so the stages have no memory conflict.
//
// The ring stages use the fifo through their ends of its multicast groups
// (checked under LOWERING): the producer names its own MNISU, each consumer
// the MNISU of its group's producer, from %src_map, which maps the ring
// consumer tiles 1..3 to it. Both carry the attributes of the producers
// 0..1: group g, 1 ring consumer for group 0 and 2 for group 1, and the
// direction each producer sends in.
//
// In DFIR the attributes of a group differ per unit, so the MNISU and MNILU
// program units have one region per unit; each creates the group of its
// producer's entry and moves the tile with one composite load that sends to
// the group or one composite store that receives from it, one stick per time
// step over 64 time steps, with no loop around them.

// The DFIR, in full: every unit declaration; the ring producer program unit
// over the MNISU of cores 0..1, with one region per unit, each creating the
// group of its core and sending to it in its direction; the ring consumer
// program unit over the MNILU of cores 1..3, with one region per unit, each
// joining the group of the producer of its core; and the local copy on core
// 0, as in local-identity-chain.mlir. No sync is emitted and no ktdf or
// ktdf_lowering op survives.

// NOSYNC-NOT:    dataflow.sync
// NOSYNC-NOT:    ktdf.
// NOSYNC-NOT:    ktdf_lowering.

// CHECK-DAG: #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG: #[[$ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$NEXT_ROW:.+]] = affine_map<(d0) -> (d0, 0)>
// CHECK-DAG: #[[$TIME_ORDER:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$STICK:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$ROWS:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL:   func.func @ring_broadcast() attributes {grid = [4]} {
// CHECK-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-NEXT:     %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// CHECK-NEXT:     %[[L1SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// CHECK-NEXT:     %[[MEM0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM2:.*]] = dataflow.get_unit {core = 2 : i32, name = "C2-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM3:.*]] = dataflow.get_unit {core = 3 : i32, name = "C3-l1", type = "l1"} : index

// The ring producer: MNISU on cores 0..1, one region per unit.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_SU:.*]] -> (%[[SU0]], %[[SU1]]) : {
// CHECK-NEXT:       %[[SU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU0]] -> %[[MEM0]]], {{\[}}%[[SU1]] -> %[[MEM1]]]):index
// CHECK-NEXT:       %[[SU_MEM:.*]] = uniform.query_map(map:%[[SU_MEM_MAP]], key:%[[PU_SU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[SU0]]){
// CHECK-NEXT:           %[[GROUP0:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           %[[SRC0:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_load %[[SRC0]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()(%[[STICK0:.*]]:vector<64xf16>)
// CHECK-NEXT:            {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             dataflow.send %[[GROUP0]], %[[STICK0]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU1]]){
// CHECK-NEXT:           %[[GROUP1:.*]] = dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 2 : i32} : index
// CHECK-NEXT:           %[[SRC1:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_load %[[SRC1]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()(%[[STICK1:.*]]:vector<64xf16>)
// CHECK-NEXT:            {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             dataflow.send %[[GROUP1]], %[[STICK1]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The ring consumer: MNILU on cores 1..3, one region per unit, each joining
// the group of the producer of its core, c floordiv 2.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_LU:.*]] -> (%[[LU1]], %[[LU2]], %[[LU3]]) : {
// CHECK-NEXT:       %[[LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU1]] -> %[[MEM1]]], {{\[}}%[[LU2]] -> %[[MEM2]]], {{\[}}%[[LU3]] -> %[[MEM3]]]):index
// CHECK-NEXT:       %[[LU_MEM:.*]] = uniform.query_map(map:%[[LU_MEM_MAP]], key:%[[PU_LU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[LU1]]){
// CHECK-NEXT:           %[[JOIN1:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 1 : i32} : index
// CHECK-NEXT:           %[[DST1:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_store %[[DST1]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()
// CHECK-NEXT:            {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             %[[RECEIVED1:.*]] = dataflow.receive %[[JOIN1]] : vector<64xf16>
// CHECK-NEXT:             agen.yield %[[RECEIVED1]] : vector<64xf16>
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU2]]){
// CHECK-NEXT:           %[[JOIN2:.*]] = dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 2 : i32} : index
// CHECK-NEXT:           %[[DST2:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_store %[[DST2]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()
// CHECK-NEXT:            {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             %[[RECEIVED2:.*]] = dataflow.receive %[[JOIN2]] : vector<64xf16>
// CHECK-NEXT:             agen.yield %[[RECEIVED2]] : vector<64xf16>
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU3]]){
// CHECK-NEXT:           %[[JOIN3:.*]] = dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 2 : i32} : index
// CHECK-NEXT:           %[[DST3:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           agen.composite_store %[[DST3]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:            time_symbols()
// CHECK-NEXT:            {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NEXT_ROW]], time_order = #[[$TIME_ORDER]], time_set = #[[$ROWS]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             %[[RECEIVED3:.*]] = dataflow.receive %[[JOIN3]] : vector<64xf16>
// CHECK-NEXT:             agen.yield %[[RECEIVED3]] : vector<64xf16>
// CHECK-NEXT:           } : memref<64x64xf16>
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The local copy on core 0, in a loop over the 64 sticks: L1LU loads a stick
// from the view at offset 0 and sends it to the SFU, which forwards it to
// L1SU, which stores it to the view at offset 128.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_L1LU:.*]] -> (%[[L1LU0]]) : {
// CHECK-NEXT:       %[[L1LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU0]] -> %[[MEM0]]]):index
// CHECK-NEXT:       %[[L1LU_MEM:.*]] = uniform.query_map(map:%[[L1LU_MEM_MAP]], key:%[[PU_L1LU]]) : index
// CHECK-NEXT:       %[[LOCAL_SRC:.*]] = dataflow.get_logical_memory_view %[[L1LU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       scf.for %[[ROW0:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:         %[[LOADED:.*]] = agen.vector_load %[[LOCAL_SRC]]{{\[}}%[[C0]] + %[[ROW0]], %[[C0]]] {load_order = #[[$ORDER]], load_set = #[[$STICK]]} : memref<64x64xf16>, vector<64xf16>
// CHECK-NEXT:         %[[TO_SFU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU0]] -> %[[SFU0]]]):index
// CHECK-NEXT:         %[[TO_SFU:.*]] = uniform.query_map(map:%[[TO_SFU_MAP]], key:%[[PU_L1LU]]) : index
// CHECK-NEXT:         dataflow.send %[[TO_SFU]], %[[LOADED]] : vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_SFU:.*]] -> (%[[SFU0]]) : {
// CHECK-NEXT:       scf.for %{{.*}} = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:         %[[FROM_L1LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SFU0]] -> %[[L1LU0]]]):index
// CHECK-NEXT:         %[[FROM_L1LU:.*]] = uniform.query_map(map:%[[FROM_L1LU_MAP]], key:%[[PU_SFU]]) : index
// CHECK-NEXT:         %[[FORWARDED:.*]] = dataflow.receive %[[FROM_L1LU]] : vector<64xf16>
// CHECK-NEXT:         %[[TO_L1SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SFU0]] -> %[[L1SU0]]]):index
// CHECK-NEXT:         %[[TO_L1SU:.*]] = uniform.query_map(map:%[[TO_L1SU_MAP]], key:%[[PU_SFU]]) : index
// CHECK-NEXT:         dataflow.send %[[TO_L1SU]], %[[FORWARDED]] : vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_L1SU:.*]] -> (%[[L1SU0]]) : {
// CHECK-NEXT:       %[[L1SU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1SU0]] -> %[[MEM0]]]):index
// CHECK-NEXT:       %[[L1SU_MEM:.*]] = uniform.query_map(map:%[[L1SU_MEM_MAP]], key:%[[PU_L1SU]]) : index
// CHECK-NEXT:       %[[LOCAL_DST:.*]] = dataflow.get_logical_memory_view %[[L1SU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       scf.for %[[ROW2:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:         %[[FROM_SFU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1SU0]] -> %[[SFU0]]]):index
// CHECK-NEXT:         %[[FROM_SFU:.*]] = uniform.query_map(map:%[[FROM_SFU_MAP]], key:%[[PU_L1SU]]) : index
// CHECK-NEXT:         %[[STORED:.*]] = dataflow.receive %[[FROM_SFU]] : vector<64xf16>
// CHECK-NEXT:         agen.vector_store %[[STORED]], %[[LOCAL_DST]]{{\[}}%[[C0]] + %[[ROW2]], %[[C0]]] {store_order = #[[$ORDER]], store_set = #[[$STICK]]} : memref<64x64xf16>, vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// LOWERING-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// LOWERING-DAG: #[[$SET:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// LOWERING-DAG: #[[$GROUP_DOMAIN:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// LOWERING-LABEL:   ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")

// LOWERING-LABEL:   func.func @ring_broadcast() attributes {grid = [4]} {
// LOWERING-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// LOWERING-NEXT:     %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// LOWERING-NEXT:     %[[L1SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// LOWERING-NEXT:     %[[TILE:.*]] = ktdp.get_compute_tile_id : index
// LOWERING-NEXT:     %[[SU_KEY0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[SU_KEY1:.*]] = arith.constant 1 : index
// LOWERING-NEXT:     %[[SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU_KEY0]] -> %[[SU0]]], {{\[}}%[[SU_KEY1]] -> %[[SU1]]]):index
// LOWERING-NEXT:     %[[SU:.*]] = uniform.query_map(map:%[[SU_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[LU_KEY1:.*]] = arith.constant 1 : index
// LOWERING-NEXT:     %[[LU_KEY2:.*]] = arith.constant 2 : index
// LOWERING-NEXT:     %[[LU_KEY3:.*]] = arith.constant 3 : index
// LOWERING-NEXT:     %[[LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU_KEY1]] -> %[[LU1]]], {{\[}}%[[LU_KEY2]] -> %[[LU2]]], {{\[}}%[[LU_KEY3]] -> %[[LU3]]]):index
// LOWERING-NEXT:     %[[LU:.*]] = uniform.query_map(map:%[[LU_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[L1LU_KEY0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[L1LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_KEY0]] -> %[[L1LU0]]]):index
// LOWERING-NEXT:     %[[L1LU:.*]] = uniform.query_map(map:%[[L1LU_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[SFU_KEY0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[SFU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SFU_KEY0]] -> %[[SFU0]]]):index
// LOWERING-NEXT:     %[[SFU:.*]] = uniform.query_map(map:%[[SFU_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[L1SU_KEY0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[L1SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1SU_KEY0]] -> %[[L1SU0]]]):index
// LOWERING-NEXT:     %[[L1SU:.*]] = uniform.query_map(map:%[[L1SU_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[SRC_KEY1:.*]] = arith.constant 1 : index
// LOWERING-NEXT:     %[[SRC_KEY2:.*]] = arith.constant 2 : index
// LOWERING-NEXT:     %[[SRC_KEY3:.*]] = arith.constant 3 : index
// LOWERING-NEXT:     %[[SRC_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SRC_KEY1]] -> %[[SU0]]], {{\[}}%[[SRC_KEY2]] -> %[[SU1]]], {{\[}}%[[SRC_KEY3]] -> %[[SU1]]]):index
// LOWERING-NEXT:     %[[SRC_SU:.*]] = uniform.query_map(map:%[[SRC_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// LOWERING-NEXT:     %[[SRC_VIEW:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// LOWERING-NEXT:     %[[SRC:.*]] = memref.memory_space_cast %[[SRC_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// LOWERING-NEXT:     %[[DST_VIEW:.*]] = ktdp.construct_memory_view %[[C128]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// LOWERING-NEXT:     %[[DST:.*]] = memref.memory_space_cast %[[DST_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// LOWERING-NEXT:     ktdf_lowering.execute_on %[[SU]], %[[LU]], %[[L1LU]], %[[SFU]], %[[L1SU]] {
// LOWERING-NEXT:       ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// LOWERING-NEXT:         ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// LOWERING-NEXT:           %[[CH:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$GROUP_DOMAIN]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:           %[[TOKEN:.*]] = ktdf.create_token : !ktdf.token
// LOWERING-NEXT:           ktdf_lowering.execute_on %[[SU]] {
// LOWERING-NEXT:             %[[TO_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, producers = array<i64: 0, 1>} : <"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:             ktdf.data_transfer from %[[SRC]]{{\[}}%[[C0]], %[[C0]]] size [64, 64] to %[[TO_GROUP]] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:           }
// LOWERING-NEXT:           ktdf_lowering.execute_on %[[LU]] {
// LOWERING-NEXT:             %[[FROM_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SRC_SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, producers = array<i64: 0, 1>} : <"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:             ktdf.data_transfer from %[[FROM_GROUP]] size [64, 64] to %[[DST]]{{\[}}%[[C0]], %[[C0]]] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
// LOWERING-NEXT:           }
// LOWERING-NEXT:         }
// LOWERING-NEXT:       }
// LOWERING-NEXT:       ktdf_lowering.execute_on %[[L1LU]], %[[SFU]], %[[L1SU]] {
// LOWERING-NEXT:         %[[ROW_LB:.*]] = arith.constant 0 : index
// LOWERING-NEXT:         %[[ROW_STEP:.*]] = arith.constant 1 : index
// LOWERING-NEXT:         %[[ROW_UB:.*]] = arith.constant 64 : index
// LOWERING-NEXT:         scf.for %[[ROW:.*]] = %[[ROW_LB]] to %[[ROW_UB]] step %[[ROW_STEP]] {
// LOWERING-NEXT:           ktdf_lowering.execute_on %[[L1LU]], %[[SFU]], %[[L1SU]] {
// LOWERING-NEXT:             %[[TO_SFU:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// LOWERING-NEXT:             %[[FROM_SFU:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// LOWERING-NEXT:             %[[LOADED:.*]] = ktdf.create_token : !ktdf.token
// LOWERING-NEXT:             %[[COMPUTED:.*]] = ktdf.create_token : !ktdf.token
// LOWERING-NEXT:             ktdf_lowering.execute_on %[[L1LU]] {
// LOWERING-NEXT:               ktdf.data_transfer from %[[SRC]]{{\[}}%[[C0]] + %[[ROW]], %[[C0]]] size [1, 64] to %[[TO_SFU]] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// LOWERING-NEXT:             }
// LOWERING-NEXT:             ktdf_lowering.execute_on %[[SFU]] {
// LOWERING-NEXT:               %[[IN:.*]] = ktdf.read_from_fifo %[[TO_SFU]] : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// LOWERING-NEXT:               %[[INIT:.*]] = tensor.empty() : tensor<64xf16>
// LOWERING-NEXT:               %[[OUT:.*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]} ins(%[[IN]] : tensor<64xf16>) outs(%[[INIT]] : tensor<64xf16>) {
// LOWERING-NEXT:               ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// LOWERING-NEXT:                 linalg.yield %[[X]] : f16
// LOWERING-NEXT:               } -> tensor<64xf16>
// LOWERING-NEXT:               ktdf.write_to_fifo %[[OUT]], %[[FROM_SFU]] : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// LOWERING-NEXT:             }
// LOWERING-NEXT:             ktdf_lowering.execute_on %[[L1SU]] {
// LOWERING-NEXT:               ktdf.data_transfer from %[[FROM_SFU]] size [1, 64] to %[[DST]]{{\[}}%[[C0]] + %[[ROW]], %[[C0]]] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x64xf16, "L1">
// LOWERING-NEXT:             }
// LOWERING-NEXT:           }
// LOWERING-NEXT:         }
// LOWERING-NEXT:       }
// LOWERING-NEXT:     }
// LOWERING-NEXT:     return
// LOWERING-NEXT:   }

#groups = affine_set<(g) : (g >= 0, 1 - g >= 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 2 * g >= 0, 2 * g + 1 - c >= 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring_device import("../../../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_broadcast() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c128, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[%c0, %c0] size [64, 64] to %p#0 size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [64, 64] to %dst[%c0, %c0] size [64, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
      } {dataflow_scheduler.domain = #consumers}
    }
    return
  }
}
