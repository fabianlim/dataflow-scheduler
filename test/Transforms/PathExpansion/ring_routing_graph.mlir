// Verify how the routing graph folds switches into kind-level nodes, using the
// graph that path expansion dumps under --debug-only.

// RUN: dataflow-scheduler-opt --split-input-file --path-expansion --debug-only=path-expansion %s 2>&1 >/dev/null | FileCheck %s

// All four ring stops fold into one RING_STOP node, so an L1 -> L1 transfer
// between cores is the path L1 -> MNISU -> RING_STOP -> MNILU -> L1. The ring
// links between stops collapse into a self edge on RING_STOP.

// CHECK-LABEL: RoutingGraph {
// CHECK:         - [[L1:[0-9]+]]: "L1" [Memory]
// CHECK-NEXT:    - [[MNILU:[0-9]+]]: "MNILU" [LoadStoreUnit]
// CHECK-NEXT:    - [[MNISU:[0-9]+]]: "MNISU" [LoadStoreUnit]
// CHECK:         - [[STOP:[0-9]+]]: "RING_STOP" [Switch]
// CHECK-NOT:     "RING_STOP" [Switch]
// CHECK:       Edges:
// CHECK-DAG:     - [[L1]]: "L1" -> [[MNISU]]: "MNISU" [cost=1]
// CHECK-DAG:     - [[MNISU]]: "MNISU" -> [[STOP]]: "RING_STOP" [cost=1]
// CHECK-DAG:     - [[STOP]]: "RING_STOP" -> [[MNILU]]: "MNILU" [cost=1]
// CHECK-DAG:     - [[MNILU]]: "MNILU" -> [[L1]]: "L1" [cost=1]
// CHECK-DAG:     - [[STOP]]: "RING_STOP" -> [[STOP]]: "RING_STOP" [cost=1]
// CHECK:       }
ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")

// -----

// A switch's connectivity decides which of its datapaths become edges: port 0
// only reaches port 1, so nothing is routed out of port 2 or into port 3.

// CHECK-LABEL: RoutingGraph {
// CHECK:         - [[IN:[0-9]+]]: "IN" [Memory]
// CHECK-NEXT:    - [[OUT:[0-9]+]]: "OUT" [Memory]
// CHECK-NEXT:    - [[DEAD:[0-9]+]]: "DEAD" [Memory]
// CHECK-NEXT:    - [[SW:[0-9]+]]: "SW" [Switch]
// CHECK:       Edges:
// CHECK-NEXT:    - [[IN]]: "IN" -> [[SW]]: "SW" [cost=1]
// CHECK-NEXT:    - [[SW]]: "SW" -> [[OUT]]: "OUT" [cost=1]
// CHECK-NEXT:  }
ktdf_arch.device @switch_device {
  %in = memory { kind = "IN" }
  %out = memory { kind = "OUT" }
  %dead = memory { kind = "DEAD" }
  %sw:4 = switch[4] {
    kind = "SW",
    connectivity = sparse<[[0, 1]], true> : tensor<4x4xi1>
  }
  datapath %in to %sw#0 : memory, port
  datapath %sw#1 to %out : port, memory
  datapath %sw#2 to %dead : port, memory
  datapath %dead to %sw#3 : memory, port
}
