// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering %s | FileCheck %s --check-prefix=LOWERING
// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering --ktdflowering-to-dfir %s | FileCheck %s
// RUN: dataflow-scheduler-opt --path-expansion --handle-cross-core-stages --ktdf-to-ktdflowering --ktdflowering-to-dfir %s | FileCheck %s --check-prefix=NOSYNC

// The broadcast relayout as the frontend writes it, on the 32-core ring
// device, through the scheduling steps it needs so far: path expansion maps
// the channel's stages to MNISU and MNILU, handle-cross-core-stages realizes
// the self-delivery of core 0 as a local copy L1LU -> SFU -> L1SU, and
// ktdf-to-ktdflowering materializes the units on the tiles of the leaf
// stages of each kind.
//
// The channel has the groups g = 0..7: producer core g sends its slice to the
// consumer cores 4g..4g+3. The ring carries C(g) without P(g), so MNISU is on
// cores 0..7 and MNILU on cores 1..31; the local copy carries core 0's slice
// to core 0, so L1LU, SFU and L1SU are on core 0 only. Each kind's map from
// tile to unit lists exactly those tiles. The stages that wrap the channel
// and the local copy are not counted for the tiles; they execute on the units
// of the stages they wrap.
//
// The token between the ring stages gets no signal: the producer stage
// writes no memory, only the fifo, so the stages have no memory conflict.
//
// The ring stages use the fifo through their ends of its multicast groups
// (checked under LOWERING): the producer names its own MNISU, each consumer
// the MNISU of its group's producer, from %src_map, which maps the ring
// consumer tiles 1..31 to it. Both carry the attributes of the producers
// 0..7: group g, 3 ring consumers for group 0 and 4 for the others, and the
// direction each producer sends in.
//
// In DFIR the attributes of a group differ per unit, so the MNISU and MNILU
// program units have one region per unit; each creates the group of its
// producer's entry and moves one stick per iteration with a composite load
// that sends to the group or a composite store that receives from it.

// The DFIR. Its output is large, so the checks below are explicit about what
// they cover: every unit declaration; the ring producer program unit over
// the MNISU of cores 0..7, with one region per unit, each creating the group
// of its core and sending to it in its direction (core 0's region in full);
// the ring consumer program unit over the MNILU of cores 1..31, with one
// region per unit, each joining the group of the producer of its core (the
// regions of core 1, fed by core 0, and core 9, fed by core 2, in full); and
// the local copy on core 0, as in local-identity-chain.mlir. No sync is
// emitted and no ktdf or ktdf_lowering op survives.

// NOSYNC-NOT:    dataflow.sync
// NOSYNC-NOT:    ktdf.
// NOSYNC-NOT:    ktdf_lowering.

// CHECK-DAG: #[[$LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG: #[[$ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$NO_OFFSET:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK-DAG: #[[$TIME_ORDER:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$STICK:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$ONE_STEP:.+]] = affine_set<(d0) : (d0 == 0)>

// CHECK-LABEL:   func.func @ring_broadcast() attributes {grid = [32]} {
// CHECK-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU4:.*]] = dataflow.get_unit {core = 4 : i32, corelet = 0 : i32, name = "C4-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU5:.*]] = dataflow.get_unit {core = 5 : i32, corelet = 0 : i32, name = "C5-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU6:.*]] = dataflow.get_unit {core = 6 : i32, corelet = 0 : i32, name = "C6-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[SU7:.*]] = dataflow.get_unit {core = 7 : i32, corelet = 0 : i32, name = "C7-mnisu", type = "mnisu"} : index
// CHECK-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU4:.*]] = dataflow.get_unit {core = 4 : i32, corelet = 0 : i32, name = "C4-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU5:.*]] = dataflow.get_unit {core = 5 : i32, corelet = 0 : i32, name = "C5-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU6:.*]] = dataflow.get_unit {core = 6 : i32, corelet = 0 : i32, name = "C6-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU7:.*]] = dataflow.get_unit {core = 7 : i32, corelet = 0 : i32, name = "C7-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU8:.*]] = dataflow.get_unit {core = 8 : i32, corelet = 0 : i32, name = "C8-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU9:.*]] = dataflow.get_unit {core = 9 : i32, corelet = 0 : i32, name = "C9-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU10:.*]] = dataflow.get_unit {core = 10 : i32, corelet = 0 : i32, name = "C10-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU11:.*]] = dataflow.get_unit {core = 11 : i32, corelet = 0 : i32, name = "C11-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU12:.*]] = dataflow.get_unit {core = 12 : i32, corelet = 0 : i32, name = "C12-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU13:.*]] = dataflow.get_unit {core = 13 : i32, corelet = 0 : i32, name = "C13-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU14:.*]] = dataflow.get_unit {core = 14 : i32, corelet = 0 : i32, name = "C14-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU15:.*]] = dataflow.get_unit {core = 15 : i32, corelet = 0 : i32, name = "C15-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU16:.*]] = dataflow.get_unit {core = 16 : i32, corelet = 0 : i32, name = "C16-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU17:.*]] = dataflow.get_unit {core = 17 : i32, corelet = 0 : i32, name = "C17-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU18:.*]] = dataflow.get_unit {core = 18 : i32, corelet = 0 : i32, name = "C18-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU19:.*]] = dataflow.get_unit {core = 19 : i32, corelet = 0 : i32, name = "C19-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU20:.*]] = dataflow.get_unit {core = 20 : i32, corelet = 0 : i32, name = "C20-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU21:.*]] = dataflow.get_unit {core = 21 : i32, corelet = 0 : i32, name = "C21-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU22:.*]] = dataflow.get_unit {core = 22 : i32, corelet = 0 : i32, name = "C22-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU23:.*]] = dataflow.get_unit {core = 23 : i32, corelet = 0 : i32, name = "C23-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU24:.*]] = dataflow.get_unit {core = 24 : i32, corelet = 0 : i32, name = "C24-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU25:.*]] = dataflow.get_unit {core = 25 : i32, corelet = 0 : i32, name = "C25-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU26:.*]] = dataflow.get_unit {core = 26 : i32, corelet = 0 : i32, name = "C26-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU27:.*]] = dataflow.get_unit {core = 27 : i32, corelet = 0 : i32, name = "C27-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU28:.*]] = dataflow.get_unit {core = 28 : i32, corelet = 0 : i32, name = "C28-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU29:.*]] = dataflow.get_unit {core = 29 : i32, corelet = 0 : i32, name = "C29-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU30:.*]] = dataflow.get_unit {core = 30 : i32, corelet = 0 : i32, name = "C30-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[LU31:.*]] = dataflow.get_unit {core = 31 : i32, corelet = 0 : i32, name = "C31-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-NEXT:     %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// CHECK-NEXT:     %[[L1SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// CHECK-NEXT:     %[[MEM0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM2:.*]] = dataflow.get_unit {core = 2 : i32, name = "C2-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM3:.*]] = dataflow.get_unit {core = 3 : i32, name = "C3-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM4:.*]] = dataflow.get_unit {core = 4 : i32, name = "C4-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM5:.*]] = dataflow.get_unit {core = 5 : i32, name = "C5-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM6:.*]] = dataflow.get_unit {core = 6 : i32, name = "C6-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM7:.*]] = dataflow.get_unit {core = 7 : i32, name = "C7-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM8:.*]] = dataflow.get_unit {core = 8 : i32, name = "C8-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM9:.*]] = dataflow.get_unit {core = 9 : i32, name = "C9-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM10:.*]] = dataflow.get_unit {core = 10 : i32, name = "C10-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM11:.*]] = dataflow.get_unit {core = 11 : i32, name = "C11-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM12:.*]] = dataflow.get_unit {core = 12 : i32, name = "C12-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM13:.*]] = dataflow.get_unit {core = 13 : i32, name = "C13-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM14:.*]] = dataflow.get_unit {core = 14 : i32, name = "C14-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM15:.*]] = dataflow.get_unit {core = 15 : i32, name = "C15-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM16:.*]] = dataflow.get_unit {core = 16 : i32, name = "C16-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM17:.*]] = dataflow.get_unit {core = 17 : i32, name = "C17-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM18:.*]] = dataflow.get_unit {core = 18 : i32, name = "C18-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM19:.*]] = dataflow.get_unit {core = 19 : i32, name = "C19-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM20:.*]] = dataflow.get_unit {core = 20 : i32, name = "C20-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM21:.*]] = dataflow.get_unit {core = 21 : i32, name = "C21-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM22:.*]] = dataflow.get_unit {core = 22 : i32, name = "C22-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM23:.*]] = dataflow.get_unit {core = 23 : i32, name = "C23-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM24:.*]] = dataflow.get_unit {core = 24 : i32, name = "C24-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM25:.*]] = dataflow.get_unit {core = 25 : i32, name = "C25-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM26:.*]] = dataflow.get_unit {core = 26 : i32, name = "C26-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM27:.*]] = dataflow.get_unit {core = 27 : i32, name = "C27-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM28:.*]] = dataflow.get_unit {core = 28 : i32, name = "C28-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM29:.*]] = dataflow.get_unit {core = 29 : i32, name = "C29-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM30:.*]] = dataflow.get_unit {core = 30 : i32, name = "C30-l1", type = "l1"} : index
// CHECK-NEXT:     %[[MEM31:.*]] = dataflow.get_unit {core = 31 : i32, name = "C31-l1", type = "l1"} : index

// The ring producer: MNISU on cores 0..7, one region per unit.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_SU:.*]] -> (%[[SU0]], %[[SU1]], %[[SU2]], %[[SU3]], %[[SU4]], %[[SU5]], %[[SU6]], %[[SU7]]) : {
// CHECK-NEXT:       %[[SU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU0]] -> %[[MEM0]]], {{\[}}%[[SU1]] -> %[[MEM1]]], {{\[}}%[[SU2]] -> %[[MEM2]]], {{\[}}%[[SU3]] -> %[[MEM3]]], {{\[}}%[[SU4]] -> %[[MEM4]]], {{\[}}%[[SU5]] -> %[[MEM5]]], {{\[}}%[[SU6]] -> %[[MEM6]]], {{\[}}%[[SU7]] -> %[[MEM7]]]):index
// CHECK-NEXT:       %[[SU_MEM:.*]] = uniform.query_map(map:%[[SU_MEM_MAP]], key:%[[PU_SU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[SU0]]){
// CHECK-NEXT:           %[[SRC0:.*]] = dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           scf.for %[[I0:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:             %[[GROUP0:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 3 : i32} : index
// CHECK-NEXT:             agen.composite_load %[[SRC0]]{{\[}}%[[I0]], %[[C0]]]
// CHECK-NEXT:              time_symbols()(%[[STICK0:.*]]:vector<64xf16>)
// CHECK-NEXT:              {load_order = #[[$ORDER]], load_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               dataflow.send %[[GROUP0]], %[[STICK0]] {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK-NEXT:               agen.yield
// CHECK-NEXT:             } : memref<64x64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU1]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP1:.*]] = dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP1]], %{{.*}} {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU2]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP2:.*]] = dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 2 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP2]], %{{.*}} {dir = #dataflow<direction Clockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU3]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP3:.*]] = dataflow.create_multicast_group(%[[SU3]] -> ()) {count = 0 : i32, group_id = 3 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP3]], %{{.*}} {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU4]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP4:.*]] = dataflow.create_multicast_group(%[[SU4]] -> ()) {count = 0 : i32, group_id = 4 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP4]], %{{.*}} {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU5]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP5:.*]] = dataflow.create_multicast_group(%[[SU5]] -> ()) {count = 0 : i32, group_id = 5 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP5]], %{{.*}} {dir = #dataflow<direction Clockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU6]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP6:.*]] = dataflow.create_multicast_group(%[[SU6]] -> ()) {count = 0 : i32, group_id = 6 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP6]], %{{.*}} {dir = #dataflow<direction Clockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[SU7]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[SU_MEM]], %[[C0]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             %[[GROUP7:.*]] = dataflow.create_multicast_group(%[[SU7]] -> ()) {count = 0 : i32, group_id = 7 : i32, num_consumers = 4 : i32} : index
// CHECK:                  dataflow.send %[[GROUP7]], %{{.*}} {dir = #dataflow<direction CounterClockwise>} : vector<64xf16>
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The ring consumer: MNILU on cores 1..31, one region per unit, each joining
// the group of the producer of its core, c floordiv 4.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_LU:.*]] -> (%[[LU1]], %[[LU2]], %[[LU3]], %[[LU4]], %[[LU5]], %[[LU6]], %[[LU7]], %[[LU8]], %[[LU9]], %[[LU10]], %[[LU11]], %[[LU12]], %[[LU13]], %[[LU14]], %[[LU15]], %[[LU16]], %[[LU17]], %[[LU18]], %[[LU19]], %[[LU20]], %[[LU21]], %[[LU22]], %[[LU23]], %[[LU24]], %[[LU25]], %[[LU26]], %[[LU27]], %[[LU28]], %[[LU29]], %[[LU30]], %[[LU31]]) : {
// CHECK-NEXT:       %[[LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU1]] -> %[[MEM1]]], {{\[}}%[[LU2]] -> %[[MEM2]]], {{\[}}%[[LU3]] -> %[[MEM3]]], {{\[}}%[[LU4]] -> %[[MEM4]]], {{\[}}%[[LU5]] -> %[[MEM5]]], {{\[}}%[[LU6]] -> %[[MEM6]]], {{\[}}%[[LU7]] -> %[[MEM7]]], {{\[}}%[[LU8]] -> %[[MEM8]]], {{\[}}%[[LU9]] -> %[[MEM9]]], {{\[}}%[[LU10]] -> %[[MEM10]]], {{\[}}%[[LU11]] -> %[[MEM11]]], {{\[}}%[[LU12]] -> %[[MEM12]]], {{\[}}%[[LU13]] -> %[[MEM13]]], {{\[}}%[[LU14]] -> %[[MEM14]]], {{\[}}%[[LU15]] -> %[[MEM15]]], {{\[}}%[[LU16]] -> %[[MEM16]]], {{\[}}%[[LU17]] -> %[[MEM17]]], {{\[}}%[[LU18]] -> %[[MEM18]]], {{\[}}%[[LU19]] -> %[[MEM19]]], {{\[}}%[[LU20]] -> %[[MEM20]]], {{\[}}%[[LU21]] -> %[[MEM21]]], {{\[}}%[[LU22]] -> %[[MEM22]]], {{\[}}%[[LU23]] -> %[[MEM23]]], {{\[}}%[[LU24]] -> %[[MEM24]]], {{\[}}%[[LU25]] -> %[[MEM25]]], {{\[}}%[[LU26]] -> %[[MEM26]]], {{\[}}%[[LU27]] -> %[[MEM27]]], {{\[}}%[[LU28]] -> %[[MEM28]]], {{\[}}%[[LU29]] -> %[[MEM29]]], {{\[}}%[[LU30]] -> %[[MEM30]]], {{\[}}%[[LU31]] -> %[[MEM31]]]):index
// CHECK-NEXT:       %[[LU_MEM:.*]] = uniform.query_map(map:%[[LU_MEM_MAP]], key:%[[PU_LU]]) : index
// CHECK-NEXT:       uniform.uniformize_regions -> () {
// CHECK-NEXT:         (%{{.*}} -> %[[LU1]]){
// CHECK-NEXT:           %[[DST1:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           scf.for %[[I1:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:             %[[JOIN1:.*]] = dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 3 : i32} : index
// CHECK-NEXT:             agen.composite_store %[[DST1]]{{\[}}%[[I1]], %[[C0]]]
// CHECK-NEXT:              time_symbols()
// CHECK-NEXT:              {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               %[[RECEIVED1:.*]] = dataflow.receive %[[JOIN1]] : vector<64xf16>
// CHECK-NEXT:               agen.yield %[[RECEIVED1]] : vector<64xf16>
// CHECK-NEXT:             } : memref<64x64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU2]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 3 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU3]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU0]] -> ()) {count = 0 : i32, group_id = 0 : i32, num_consumers = 3 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU4]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU5]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU6]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU7]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU1]] -> ()) {count = 0 : i32, group_id = 1 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU8]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 2 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU9]]){
// CHECK-NEXT:           %[[DST9:.*]] = dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:           scf.for %[[I9:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:             %[[JOIN9:.*]] = dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 2 : i32, num_consumers = 4 : i32} : index
// CHECK-NEXT:             agen.composite_store %[[DST9]]{{\[}}%[[I9]], %[[C0]]]
// CHECK-NEXT:              time_symbols()
// CHECK-NEXT:              {store_order = #[[$ORDER]], store_set = #[[$STICK]], time_addr_map = #[[$NO_OFFSET]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
// CHECK-NEXT:             {
// CHECK-NEXT:               %[[RECEIVED9:.*]] = dataflow.receive %[[JOIN9]] : vector<64xf16>
// CHECK-NEXT:               agen.yield %[[RECEIVED9]] : vector<64xf16>
// CHECK-NEXT:             } : memref<64x64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU10]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 2 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU11]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU2]] -> ()) {count = 0 : i32, group_id = 2 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU12]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU3]] -> ()) {count = 0 : i32, group_id = 3 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU13]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU3]] -> ()) {count = 0 : i32, group_id = 3 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU14]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU3]] -> ()) {count = 0 : i32, group_id = 3 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU15]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU3]] -> ()) {count = 0 : i32, group_id = 3 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU16]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU4]] -> ()) {count = 0 : i32, group_id = 4 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU17]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU4]] -> ()) {count = 0 : i32, group_id = 4 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU18]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU4]] -> ()) {count = 0 : i32, group_id = 4 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU19]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU4]] -> ()) {count = 0 : i32, group_id = 4 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU20]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU5]] -> ()) {count = 0 : i32, group_id = 5 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU21]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU5]] -> ()) {count = 0 : i32, group_id = 5 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU22]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU5]] -> ()) {count = 0 : i32, group_id = 5 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU23]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU5]] -> ()) {count = 0 : i32, group_id = 5 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU24]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU6]] -> ()) {count = 0 : i32, group_id = 6 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU25]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU6]] -> ()) {count = 0 : i32, group_id = 6 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU26]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU6]] -> ()) {count = 0 : i32, group_id = 6 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU27]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU6]] -> ()) {count = 0 : i32, group_id = 6 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU28]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU7]] -> ()) {count = 0 : i32, group_id = 7 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU29]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU7]] -> ()) {count = 0 : i32, group_id = 7 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU30]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU7]] -> ()) {count = 0 : i32, group_id = 7 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         (%{{.*}} -> %[[LU31]]){
// CHECK-NEXT:           dataflow.get_logical_memory_view %[[LU_MEM]], %[[C128]]
// CHECK-NEXT:           scf.for
// CHECK-NEXT:             dataflow.create_multicast_group(%[[SU7]] -> ()) {count = 0 : i32, group_id = 7 : i32, num_consumers = 4 : i32} : index
// CHECK:                uniform.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// The local copy on core 0: L1LU loads a stick from the view at offset 0 and
// sends it to the SFU, which forwards it to L1SU, which stores it to the view
// at offset 128.
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[PU_L1LU:.*]] -> (%[[L1LU0]]) : {
// CHECK-NEXT:       %[[L1LU_MEM_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU0]] -> %[[MEM0]]]):index
// CHECK-NEXT:       %[[L1LU_MEM:.*]] = uniform.query_map(map:%[[L1LU_MEM_MAP]], key:%[[PU_L1LU]]) : index
// CHECK-NEXT:       %[[LOCAL_SRC:.*]] = dataflow.get_logical_memory_view %[[L1LU_MEM]], %[[C0]] {layout_map = #[[$LAYOUT]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       scf.for %[[ROW0:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:         %[[LOADED:.*]] = agen.vector_load %[[LOCAL_SRC]]{{\[}}%[[ROW0]], %[[C0]]] {load_order = #[[$ORDER]], load_set = #[[$STICK]]} : memref<64x64xf16>, vector<64xf16>
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
// CHECK-NEXT:         agen.vector_store %[[STORED]], %[[LOCAL_DST]]{{\[}}%[[ROW2]], %[[C0]]] {store_order = #[[$ORDER]], store_set = #[[$STICK]]} : memref<64x64xf16>, vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// LOWERING-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// LOWERING-DAG: #[[$SET:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// LOWERING-DAG: #[[$GROUP_DOMAIN:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
// LOWERING-LABEL:   ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")

// LOWERING-LABEL:   func.func @ring_broadcast() attributes {grid = [32]} {
// LOWERING-NEXT:     %[[SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU4:.*]] = dataflow.get_unit {core = 4 : i32, corelet = 0 : i32, name = "C4-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU5:.*]] = dataflow.get_unit {core = 5 : i32, corelet = 0 : i32, name = "C5-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU6:.*]] = dataflow.get_unit {core = 6 : i32, corelet = 0 : i32, name = "C6-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[SU7:.*]] = dataflow.get_unit {core = 7 : i32, corelet = 0 : i32, name = "C7-mnisu", type = "mnisu"} : index
// LOWERING-NEXT:     %[[LU1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU2:.*]] = dataflow.get_unit {core = 2 : i32, corelet = 0 : i32, name = "C2-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU3:.*]] = dataflow.get_unit {core = 3 : i32, corelet = 0 : i32, name = "C3-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU4:.*]] = dataflow.get_unit {core = 4 : i32, corelet = 0 : i32, name = "C4-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU5:.*]] = dataflow.get_unit {core = 5 : i32, corelet = 0 : i32, name = "C5-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU6:.*]] = dataflow.get_unit {core = 6 : i32, corelet = 0 : i32, name = "C6-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU7:.*]] = dataflow.get_unit {core = 7 : i32, corelet = 0 : i32, name = "C7-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU8:.*]] = dataflow.get_unit {core = 8 : i32, corelet = 0 : i32, name = "C8-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU9:.*]] = dataflow.get_unit {core = 9 : i32, corelet = 0 : i32, name = "C9-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU10:.*]] = dataflow.get_unit {core = 10 : i32, corelet = 0 : i32, name = "C10-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU11:.*]] = dataflow.get_unit {core = 11 : i32, corelet = 0 : i32, name = "C11-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU12:.*]] = dataflow.get_unit {core = 12 : i32, corelet = 0 : i32, name = "C12-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU13:.*]] = dataflow.get_unit {core = 13 : i32, corelet = 0 : i32, name = "C13-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU14:.*]] = dataflow.get_unit {core = 14 : i32, corelet = 0 : i32, name = "C14-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU15:.*]] = dataflow.get_unit {core = 15 : i32, corelet = 0 : i32, name = "C15-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU16:.*]] = dataflow.get_unit {core = 16 : i32, corelet = 0 : i32, name = "C16-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU17:.*]] = dataflow.get_unit {core = 17 : i32, corelet = 0 : i32, name = "C17-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU18:.*]] = dataflow.get_unit {core = 18 : i32, corelet = 0 : i32, name = "C18-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU19:.*]] = dataflow.get_unit {core = 19 : i32, corelet = 0 : i32, name = "C19-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU20:.*]] = dataflow.get_unit {core = 20 : i32, corelet = 0 : i32, name = "C20-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU21:.*]] = dataflow.get_unit {core = 21 : i32, corelet = 0 : i32, name = "C21-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU22:.*]] = dataflow.get_unit {core = 22 : i32, corelet = 0 : i32, name = "C22-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU23:.*]] = dataflow.get_unit {core = 23 : i32, corelet = 0 : i32, name = "C23-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU24:.*]] = dataflow.get_unit {core = 24 : i32, corelet = 0 : i32, name = "C24-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU25:.*]] = dataflow.get_unit {core = 25 : i32, corelet = 0 : i32, name = "C25-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU26:.*]] = dataflow.get_unit {core = 26 : i32, corelet = 0 : i32, name = "C26-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU27:.*]] = dataflow.get_unit {core = 27 : i32, corelet = 0 : i32, name = "C27-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU28:.*]] = dataflow.get_unit {core = 28 : i32, corelet = 0 : i32, name = "C28-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU29:.*]] = dataflow.get_unit {core = 29 : i32, corelet = 0 : i32, name = "C29-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU30:.*]] = dataflow.get_unit {core = 30 : i32, corelet = 0 : i32, name = "C30-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[LU31:.*]] = dataflow.get_unit {core = 31 : i32, corelet = 0 : i32, name = "C31-mnilu", type = "mnilu"} : index
// LOWERING-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// LOWERING-NEXT:     %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// LOWERING-NEXT:     %[[L1SU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// LOWERING-NEXT:     %[[TILE:.*]] = ktdp.get_compute_tile_id : index
// LOWERING-NEXT:     %[[SU_KEY0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[SU_KEY1:.*]] = arith.constant 1 : index
// LOWERING-NEXT:     %[[SU_KEY2:.*]] = arith.constant 2 : index
// LOWERING-NEXT:     %[[SU_KEY3:.*]] = arith.constant 3 : index
// LOWERING-NEXT:     %[[SU_KEY4:.*]] = arith.constant 4 : index
// LOWERING-NEXT:     %[[SU_KEY5:.*]] = arith.constant 5 : index
// LOWERING-NEXT:     %[[SU_KEY6:.*]] = arith.constant 6 : index
// LOWERING-NEXT:     %[[SU_KEY7:.*]] = arith.constant 7 : index
// LOWERING-NEXT:     %[[SU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SU_KEY0]] -> %[[SU0]]], {{\[}}%[[SU_KEY1]] -> %[[SU1]]], {{\[}}%[[SU_KEY2]] -> %[[SU2]]], {{\[}}%[[SU_KEY3]] -> %[[SU3]]], {{\[}}%[[SU_KEY4]] -> %[[SU4]]], {{\[}}%[[SU_KEY5]] -> %[[SU5]]], {{\[}}%[[SU_KEY6]] -> %[[SU6]]], {{\[}}%[[SU_KEY7]] -> %[[SU7]]]):index
// LOWERING-NEXT:     %[[SU:.*]] = uniform.query_map(map:%[[SU_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[LU_KEY1:.*]] = arith.constant 1 : index
// LOWERING-NEXT:     %[[LU_KEY2:.*]] = arith.constant 2 : index
// LOWERING-NEXT:     %[[LU_KEY3:.*]] = arith.constant 3 : index
// LOWERING-NEXT:     %[[LU_KEY4:.*]] = arith.constant 4 : index
// LOWERING-NEXT:     %[[LU_KEY5:.*]] = arith.constant 5 : index
// LOWERING-NEXT:     %[[LU_KEY6:.*]] = arith.constant 6 : index
// LOWERING-NEXT:     %[[LU_KEY7:.*]] = arith.constant 7 : index
// LOWERING-NEXT:     %[[LU_KEY8:.*]] = arith.constant 8 : index
// LOWERING-NEXT:     %[[LU_KEY9:.*]] = arith.constant 9 : index
// LOWERING-NEXT:     %[[LU_KEY10:.*]] = arith.constant 10 : index
// LOWERING-NEXT:     %[[LU_KEY11:.*]] = arith.constant 11 : index
// LOWERING-NEXT:     %[[LU_KEY12:.*]] = arith.constant 12 : index
// LOWERING-NEXT:     %[[LU_KEY13:.*]] = arith.constant 13 : index
// LOWERING-NEXT:     %[[LU_KEY14:.*]] = arith.constant 14 : index
// LOWERING-NEXT:     %[[LU_KEY15:.*]] = arith.constant 15 : index
// LOWERING-NEXT:     %[[LU_KEY16:.*]] = arith.constant 16 : index
// LOWERING-NEXT:     %[[LU_KEY17:.*]] = arith.constant 17 : index
// LOWERING-NEXT:     %[[LU_KEY18:.*]] = arith.constant 18 : index
// LOWERING-NEXT:     %[[LU_KEY19:.*]] = arith.constant 19 : index
// LOWERING-NEXT:     %[[LU_KEY20:.*]] = arith.constant 20 : index
// LOWERING-NEXT:     %[[LU_KEY21:.*]] = arith.constant 21 : index
// LOWERING-NEXT:     %[[LU_KEY22:.*]] = arith.constant 22 : index
// LOWERING-NEXT:     %[[LU_KEY23:.*]] = arith.constant 23 : index
// LOWERING-NEXT:     %[[LU_KEY24:.*]] = arith.constant 24 : index
// LOWERING-NEXT:     %[[LU_KEY25:.*]] = arith.constant 25 : index
// LOWERING-NEXT:     %[[LU_KEY26:.*]] = arith.constant 26 : index
// LOWERING-NEXT:     %[[LU_KEY27:.*]] = arith.constant 27 : index
// LOWERING-NEXT:     %[[LU_KEY28:.*]] = arith.constant 28 : index
// LOWERING-NEXT:     %[[LU_KEY29:.*]] = arith.constant 29 : index
// LOWERING-NEXT:     %[[LU_KEY30:.*]] = arith.constant 30 : index
// LOWERING-NEXT:     %[[LU_KEY31:.*]] = arith.constant 31 : index
// LOWERING-NEXT:     %[[LU_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[LU_KEY1]] -> %[[LU1]]], {{\[}}%[[LU_KEY2]] -> %[[LU2]]], {{\[}}%[[LU_KEY3]] -> %[[LU3]]], {{\[}}%[[LU_KEY4]] -> %[[LU4]]], {{\[}}%[[LU_KEY5]] -> %[[LU5]]], {{\[}}%[[LU_KEY6]] -> %[[LU6]]], {{\[}}%[[LU_KEY7]] -> %[[LU7]]], {{\[}}%[[LU_KEY8]] -> %[[LU8]]], {{\[}}%[[LU_KEY9]] -> %[[LU9]]], {{\[}}%[[LU_KEY10]] -> %[[LU10]]], {{\[}}%[[LU_KEY11]] -> %[[LU11]]], {{\[}}%[[LU_KEY12]] -> %[[LU12]]], {{\[}}%[[LU_KEY13]] -> %[[LU13]]], {{\[}}%[[LU_KEY14]] -> %[[LU14]]], {{\[}}%[[LU_KEY15]] -> %[[LU15]]], {{\[}}%[[LU_KEY16]] -> %[[LU16]]], {{\[}}%[[LU_KEY17]] -> %[[LU17]]], {{\[}}%[[LU_KEY18]] -> %[[LU18]]], {{\[}}%[[LU_KEY19]] -> %[[LU19]]], {{\[}}%[[LU_KEY20]] -> %[[LU20]]], {{\[}}%[[LU_KEY21]] -> %[[LU21]]], {{\[}}%[[LU_KEY22]] -> %[[LU22]]], {{\[}}%[[LU_KEY23]] -> %[[LU23]]], {{\[}}%[[LU_KEY24]] -> %[[LU24]]], {{\[}}%[[LU_KEY25]] -> %[[LU25]]], {{\[}}%[[LU_KEY26]] -> %[[LU26]]], {{\[}}%[[LU_KEY27]] -> %[[LU27]]], {{\[}}%[[LU_KEY28]] -> %[[LU28]]], {{\[}}%[[LU_KEY29]] -> %[[LU29]]], {{\[}}%[[LU_KEY30]] -> %[[LU30]]], {{\[}}%[[LU_KEY31]] -> %[[LU31]]]):index
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
// LOWERING-NEXT:     %[[SRC_KEY4:.*]] = arith.constant 4 : index
// LOWERING-NEXT:     %[[SRC_KEY5:.*]] = arith.constant 5 : index
// LOWERING-NEXT:     %[[SRC_KEY6:.*]] = arith.constant 6 : index
// LOWERING-NEXT:     %[[SRC_KEY7:.*]] = arith.constant 7 : index
// LOWERING-NEXT:     %[[SRC_KEY8:.*]] = arith.constant 8 : index
// LOWERING-NEXT:     %[[SRC_KEY9:.*]] = arith.constant 9 : index
// LOWERING-NEXT:     %[[SRC_KEY10:.*]] = arith.constant 10 : index
// LOWERING-NEXT:     %[[SRC_KEY11:.*]] = arith.constant 11 : index
// LOWERING-NEXT:     %[[SRC_KEY12:.*]] = arith.constant 12 : index
// LOWERING-NEXT:     %[[SRC_KEY13:.*]] = arith.constant 13 : index
// LOWERING-NEXT:     %[[SRC_KEY14:.*]] = arith.constant 14 : index
// LOWERING-NEXT:     %[[SRC_KEY15:.*]] = arith.constant 15 : index
// LOWERING-NEXT:     %[[SRC_KEY16:.*]] = arith.constant 16 : index
// LOWERING-NEXT:     %[[SRC_KEY17:.*]] = arith.constant 17 : index
// LOWERING-NEXT:     %[[SRC_KEY18:.*]] = arith.constant 18 : index
// LOWERING-NEXT:     %[[SRC_KEY19:.*]] = arith.constant 19 : index
// LOWERING-NEXT:     %[[SRC_KEY20:.*]] = arith.constant 20 : index
// LOWERING-NEXT:     %[[SRC_KEY21:.*]] = arith.constant 21 : index
// LOWERING-NEXT:     %[[SRC_KEY22:.*]] = arith.constant 22 : index
// LOWERING-NEXT:     %[[SRC_KEY23:.*]] = arith.constant 23 : index
// LOWERING-NEXT:     %[[SRC_KEY24:.*]] = arith.constant 24 : index
// LOWERING-NEXT:     %[[SRC_KEY25:.*]] = arith.constant 25 : index
// LOWERING-NEXT:     %[[SRC_KEY26:.*]] = arith.constant 26 : index
// LOWERING-NEXT:     %[[SRC_KEY27:.*]] = arith.constant 27 : index
// LOWERING-NEXT:     %[[SRC_KEY28:.*]] = arith.constant 28 : index
// LOWERING-NEXT:     %[[SRC_KEY29:.*]] = arith.constant 29 : index
// LOWERING-NEXT:     %[[SRC_KEY30:.*]] = arith.constant 30 : index
// LOWERING-NEXT:     %[[SRC_KEY31:.*]] = arith.constant 31 : index
// LOWERING-NEXT:     %[[SRC_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[SRC_KEY1]] -> %[[SU0]]], {{\[}}%[[SRC_KEY2]] -> %[[SU0]]], {{\[}}%[[SRC_KEY3]] -> %[[SU0]]], {{\[}}%[[SRC_KEY4]] -> %[[SU1]]], {{\[}}%[[SRC_KEY5]] -> %[[SU1]]], {{\[}}%[[SRC_KEY6]] -> %[[SU1]]], {{\[}}%[[SRC_KEY7]] -> %[[SU1]]], {{\[}}%[[SRC_KEY8]] -> %[[SU2]]], {{\[}}%[[SRC_KEY9]] -> %[[SU2]]], {{\[}}%[[SRC_KEY10]] -> %[[SU2]]], {{\[}}%[[SRC_KEY11]] -> %[[SU2]]], {{\[}}%[[SRC_KEY12]] -> %[[SU3]]], {{\[}}%[[SRC_KEY13]] -> %[[SU3]]], {{\[}}%[[SRC_KEY14]] -> %[[SU3]]], {{\[}}%[[SRC_KEY15]] -> %[[SU3]]], {{\[}}%[[SRC_KEY16]] -> %[[SU4]]], {{\[}}%[[SRC_KEY17]] -> %[[SU4]]], {{\[}}%[[SRC_KEY18]] -> %[[SU4]]], {{\[}}%[[SRC_KEY19]] -> %[[SU4]]], {{\[}}%[[SRC_KEY20]] -> %[[SU5]]], {{\[}}%[[SRC_KEY21]] -> %[[SU5]]], {{\[}}%[[SRC_KEY22]] -> %[[SU5]]], {{\[}}%[[SRC_KEY23]] -> %[[SU5]]], {{\[}}%[[SRC_KEY24]] -> %[[SU6]]], {{\[}}%[[SRC_KEY25]] -> %[[SU6]]], {{\[}}%[[SRC_KEY26]] -> %[[SU6]]], {{\[}}%[[SRC_KEY27]] -> %[[SU6]]], {{\[}}%[[SRC_KEY28]] -> %[[SU7]]], {{\[}}%[[SRC_KEY29]] -> %[[SU7]]], {{\[}}%[[SRC_KEY30]] -> %[[SU7]]], {{\[}}%[[SRC_KEY31]] -> %[[SU7]]]):index
// LOWERING-NEXT:     %[[SRC_SU:.*]] = uniform.query_map(map:%[[SRC_MAP]], key:%[[TILE]]) : index
// LOWERING-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// LOWERING-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// LOWERING-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// LOWERING-NEXT:     %[[C128:.*]] = arith.constant 128 : index
// LOWERING-NEXT:     %[[SRC_VIEW:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// LOWERING-NEXT:     %[[SRC:.*]] = memref.memory_space_cast %[[SRC_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// LOWERING-NEXT:     %[[DST_VIEW:.*]] = ktdp.construct_memory_view %[[C128]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET]], memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
// LOWERING-NEXT:     %[[DST:.*]] = memref.memory_space_cast %[[DST_VIEW]] : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
// LOWERING-NEXT:     scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// LOWERING-NEXT:       ktdf_lowering.execute_on %[[SU]], %[[LU]], %[[L1LU]], %[[SFU]], %[[L1SU]] {
// LOWERING-NEXT:         ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// LOWERING-NEXT:           ktdf_lowering.execute_on %[[SU]], %[[LU]] {
// LOWERING-NEXT:             %[[CH:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$GROUP_DOMAIN]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:             %[[TOKEN:.*]] = ktdf.create_token : !ktdf.token
// LOWERING-NEXT:             ktdf_lowering.execute_on %[[SU]] {
// LOWERING-NEXT:               %[[TO_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>, #dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>, #dataflow<direction Clockwise>, #dataflow<direction CounterClockwise>], group_ids = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>, num_consumers = array<i32: 3, 4, 4, 4, 4, 4, 4, 4>, producers = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>} : <"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:               ktdf.data_transfer from %[[SRC]]{{\[}}%[[I]], %[[C0]]] size [1, 64] to %[[TO_GROUP]] size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:             }
// LOWERING-NEXT:             ktdf_lowering.execute_on %[[LU]] {
// LOWERING-NEXT:               %[[FROM_GROUP:.*]] = ktdf_lowering.multicast_group %[[CH]] producer(%[[SRC_SU]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>, #dataflow<direction CounterClockwise>, #dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>, #dataflow<direction Clockwise>, #dataflow<direction CounterClockwise>], group_ids = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>, num_consumers = array<i32: 3, 4, 4, 4, 4, 4, 4, 4>, producers = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>} : <"MNISU" -> "MNILU", 64xf16>
// LOWERING-NEXT:               ktdf.data_transfer from %[[FROM_GROUP]] size [64] to %[[DST]]{{\[}}%[[I]], %[[C0]]] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
// LOWERING-NEXT:             }
// LOWERING-NEXT:           }
// LOWERING-NEXT:         }
// LOWERING-NEXT:         ktdf_lowering.execute_on %[[L1LU]], %[[SFU]], %[[L1SU]] {
// LOWERING-NEXT:           ktdf_lowering.execute_on %[[L1LU]], %[[SFU]], %[[L1SU]] {
// LOWERING-NEXT:             %[[TO_SFU:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// LOWERING-NEXT:             %[[FROM_SFU:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// LOWERING-NEXT:             %[[LOADED:.*]] = ktdf.create_token : !ktdf.token
// LOWERING-NEXT:             %[[COMPUTED:.*]] = ktdf.create_token : !ktdf.token
// LOWERING-NEXT:             ktdf_lowering.execute_on %[[L1LU]] {
// LOWERING-NEXT:               ktdf.data_transfer from %[[SRC]]{{\[}}%[[I]], %[[C0]]] size [1, 64] to %[[TO_SFU]] size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
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
// LOWERING-NEXT:               ktdf.data_transfer from %[[FROM_SFU]] size [64] to %[[DST]]{{\[}}%[[I]], %[[C0]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x64xf16, "L1">
// LOWERING-NEXT:             }
// LOWERING-NEXT:           }
// LOWERING-NEXT:         }
// LOWERING-NEXT:       }
// LOWERING-NEXT:     }
// LOWERING-NEXT:     return
// LOWERING-NEXT:   }

#groups = affine_set<(g) : (g >= 0, 7 - g >= 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @ring32_device import("../../../../Dialect/KTDFArch/ring32_device.mlir")
  func.func @ring_broadcast() attributes {grid = [32]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %src = memref.memory_space_cast %0 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    %1 = ktdp.construct_memory_view %c128, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<64x64xf16, #ktdp.memory_space<ct_local>>
    %dst = memref.memory_space_cast %1 : memref<64x64xf16, #ktdp.memory_space<ct_local>> to memref<64x64xf16, "L1">
    scf.for %i = %c0 to %c64 step %c1 {
      ktdf.pipeline {
        %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token) {
          %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#1) {
          ktdf.data_transfer from %src[%i, %c0] size [1, 64] to %p#0 size [64] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        } {dataflow_scheduler.domain = #producers}
        ktdf.stage depends_in(%p#1) depends_out(%p#2) {
          ktdf.data_transfer from %p#0 size [64] to %dst[%i, %c0] size [1, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64x64xf16, "L1">
        } {dataflow_scheduler.domain = #consumers}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
