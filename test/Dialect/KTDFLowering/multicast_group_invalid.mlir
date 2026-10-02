// RUN: dataflow-scheduler-opt %s -split-input-file -verify-diagnostics

func.func @no_producer(%unit: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  // expected-error @+1 {{'ktdf_lowering.multicast_group' op must describe at least one producer}}
  %group = ktdf_lowering.multicast_group %fifo producer(%unit) {producers = array<i64>, group_ids = array<i32>, num_consumers = array<i32>, directions = []} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  return
}

// -----

func.func @fewer_group_ids(%unit: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  // expected-error @+1 {{'ktdf_lowering.multicast_group' op 'producers', 'group_ids', 'num_consumers' and 'directions' must have one entry per producer, but they have 2, 1, 2 and 2 entries}}
  %group = ktdf_lowering.multicast_group %fifo producer(%unit) {producers = array<i64: 0, 1>, group_ids = array<i32: 0>, num_consumers = array<i32: 1, 2>, directions = [#dataflow<direction CounterClockwise>, #dataflow<direction Clockwise>]} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  return
}

// -----

func.func @fewer_directions(%unit: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  // expected-error @+1 {{'ktdf_lowering.multicast_group' op 'producers', 'group_ids', 'num_consumers' and 'directions' must have one entry per producer, but they have 2, 2, 2 and 1 entries}}
  %group = ktdf_lowering.multicast_group %fifo producer(%unit) {producers = array<i64: 0, 1>, group_ids = array<i32: 0, 1>, num_consumers = array<i32: 1, 2>, directions = [#dataflow<direction CounterClockwise>]} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  return
}

// -----

func.func @result_type_differs(%unit: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  // expected-error @+1 {{'ktdf_lowering.multicast_group' op failed to verify that all of {fifo, result} have same type}}
  %group = "ktdf_lowering.multicast_group"(%fifo, %unit) {producers = array<i64: 0>, group_ids = array<i32: 0>, num_consumers = array<i32: 1>, directions = [#dataflow<direction Clockwise>]} : (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, index) -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 128xf16>
  return
}

// -----

func.func @not_a_direction(%unit: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  // expected-error @+1 {{'ktdf_lowering.multicast_group' op attribute 'directions' failed to satisfy constraint: array of routing directions}}
  %group = ktdf_lowering.multicast_group %fifo producer(%unit) {producers = array<i64: 0>, group_ids = array<i32: 0>, num_consumers = array<i32: 1>, directions = [0 : i32]} : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
  return
}
