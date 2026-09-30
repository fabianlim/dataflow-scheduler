// Verify the routing graph of the 32-core ring device, using the graph that
// path expansion dumps under --debug-only.

// RUN: dataflow-scheduler-opt --path-expansion --debug-only=path-expansion %s 2>&1 >/dev/null | FileCheck %s

// The 32 cores, their corelets and the 32 ring stops each fold into one node
// per kind. The L1 -> L1 paths are the local corelet chain
// L1 -> L1LU -> SFU -> L1SU -> L1 and the ring path
// L1 -> MNISU -> RING_STOP -> MNILU -> L1; the links between stops collapse
// into a self edge on RING_STOP.

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
ktdf_arch.device @ring32_device import("../../Dialect/KTDFArch/ring32_device.mlir")
