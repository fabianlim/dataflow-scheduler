// Verify how the routing graph folds switches into kind-level nodes, using the
// graph that path expansion dumps under --debug-only.

// RUN: dataflow-scheduler-opt --split-input-file --path-expansion --debug-only=path-expansion %s 2>&1 >/dev/null | FileCheck %s

// The four cores, their corelets and the four ring stops each fold into one
// node per kind: the MNILU, MNISU and corelet units of cores 1..3 resolve to
// the nodes of core 0's. The L1 -> L1 paths are the local corelet chain
// L1 -> L1LU -> SFU -> L1SU -> L1 and the ring path between cores
// L1 -> MNISU -> RING_STOP -> MNILU -> L1; the ring links between stops
// collapse into a self edge on RING_STOP.

// CHECK-LABEL: RoutingGraph {
// CHECK-NEXT:  Nodes:
// CHECK-NEXT:    - [[DDR:[0-9]+]]: "DDR" [Memory]
// CHECK-NEXT:    - [[L1:[0-9]+]]: "L1" [Memory]
// CHECK-NEXT:    - [[MNILU:[0-9]+]]: "MNILU" [LoadStoreUnit]
// CHECK-NEXT:    - [[MNISU:[0-9]+]]: "MNISU" [LoadStoreUnit]
// CHECK-NEXT:    - {{[0-9]+}}: "SFU_REG" [Memory]
// CHECK-NEXT:    - [[SFU:[0-9]+]]: "SFU" [Compute]
// CHECK-NEXT:    - [[L1LU:[0-9]+]]: "L1LU" [LoadStoreUnit]
// CHECK-NEXT:    - [[L1SU:[0-9]+]]: "L1SU" [LoadStoreUnit]
// CHECK-NEXT:    - [[STOP:[0-9]+]]: "RING_STOP" [Switch]
// CHECK-NEXT:  Edges:
// CHECK-NEXT:    - [[DDR]]: "DDR" -> [[MNILU]]: "MNILU" [cost=1]
// CHECK-NEXT:    - [[L1]]: "L1" -> [[MNISU]]: "MNISU" [cost=1]
// CHECK-NEXT:    - [[L1]]: "L1" -> [[L1LU]]: "L1LU" [cost=1]
// CHECK-NEXT:    - [[MNILU]]: "MNILU" -> [[L1]]: "L1" [cost=1]
// CHECK-NEXT:    - [[MNISU]]: "MNISU" -> [[DDR]]: "DDR" [cost=1]
// CHECK-NEXT:    - [[MNISU]]: "MNISU" -> [[STOP]]: "RING_STOP" [cost=1]
// CHECK-NEXT:    - [[SFU]]: "SFU" -> [[L1SU]]: "L1SU" [cost=1]
// CHECK-NEXT:    - [[L1LU]]: "L1LU" -> [[SFU]]: "SFU" [cost=1]
// CHECK-NEXT:    - [[L1SU]]: "L1SU" -> [[L1]]: "L1" [cost=1]
// CHECK-NEXT:    - [[STOP]]: "RING_STOP" -> [[MNILU]]: "MNILU" [cost=1]
// CHECK-NEXT:    - [[STOP]]: "RING_STOP" -> [[STOP]]: "RING_STOP" [cost=1]
// CHECK-NEXT:  }
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
