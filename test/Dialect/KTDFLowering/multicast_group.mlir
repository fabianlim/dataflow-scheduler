// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s

// One end of the multicast groups of a cross-core fifo with two producers,
// tiles 0 and 1, whose groups have one and two ring consumers.

// CHECK-LABEL:   func.func @multicast_group(
// CHECK-SAME:      %[[UNIT:.*]]: index) {
// CHECK-NEXT:     %[[FIFO:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:     %{{.*}} = ktdf_lowering.multicast_group %[[FIFO]] producer(%[[UNIT]]) {directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>], group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, producers = array<i64: 0, 1>} : <"MNISU" -> "MNILU", 64xf16>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @multicast_group(%unit: index) {
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
    %group = ktdf_lowering.multicast_group %fifo producer(%unit) {producers = array<i64: 0, 1>, group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>]} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
    return
  }
}
