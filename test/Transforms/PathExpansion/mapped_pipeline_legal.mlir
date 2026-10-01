// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s
// RUN: dataflow-scheduler-opt --path-expansion --path-expansion %s | FileCheck %s

// A pipeline whose stages all have a unit has nothing left for path expansion
// to assign. It is legal if the data two consecutive stages exchange can
// travel between their units: a cross-core fifo, one with a group domain,
// through switches only; any other fifo directly; a memory buffer through
// that memory. A legal pipeline comes out unchanged; the illegal cases are in
// mapped_pipeline_illegal.mlir.
//
// @ring_fifo: a cross-core fifo from MNISU to MNILU over the ring; the units
// are joined through the RING_STOP switch. @local_chain: the local identity
// chain L1LU -> SFU -> L1SU with every stage mapped; each fifo is a direct
// link.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0)[s0] : (d0 - s0 == 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 >= 0, -d0 + s0 * 4 + 3 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")

// CHECK-LABEL:   func.func @ring_fifo(
// CHECK-SAME:      %[[ARG0:.*]]: memref<64x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<64x64xf16, "L1">) {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() {dataflow_scheduler.groups = #[[$ATTR_1]]} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#1) {
// CHECK-NEXT:         ktdf.data_transfer from %[[ARG0]][0, 0] size [64, 64] to %[[PRIVATE_0]]#0 size [4096] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
// CHECK-NEXT:       } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #[[$ATTR_2]]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[PRIVATE_0]]#1) depends_out(%[[PRIVATE_0]]#2) {
// CHECK-NEXT:         ktdf.data_transfer from %[[PRIVATE_0]]#0 size [4096] to %[[ARG1]][0, 0] size [64, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, memref<64x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #[[$ATTR_3]]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @local_chain(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<1x64xf16, "L1">) {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#2) {
// CHECK-NEXT:         ktdf.data_transfer from %[[ARG0]][0, 0] size [1, 64] to %[[PRIVATE_0]]#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[PRIVATE_0]]#2) depends_out(%[[PRIVATE_0]]#3) {
// CHECK-NEXT:         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:         ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16):
// CHECK-NEXT:           linalg.yield %[[VAL_2]] : f16
// CHECK-NEXT:         } -> tensor<64xf16>
// CHECK-NEXT:         ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_0]]#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[PRIVATE_0]]#3) depends_out(%[[PRIVATE_0]]#4) {
// CHECK-NEXT:         ktdf.data_transfer from %[[PRIVATE_0]]#1 size [64] to %[[ARG1]][0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#groups = affine_set<(g) : (g == 0)>
#producers = affine_set<(c)[g] : (c - g == 0)>
#consumers = affine_set<(c)[g] : (c - 4 * g >= 0, 4 * g + 3 - c >= 0)>
#id = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @ring_device import("../../Dialect/KTDFArch/ring_device.mlir")
  func.func @ring_fifo(%src: memref<64x64xf16, "L1">, %dst: memref<64x64xf16, "L1">) {
    ktdf.pipeline {
      %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token) {
        %ch = ktdf.fifo.allocate() {dataflow_scheduler.groups = #groups} -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %ch, %t0, %t1 : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0, 0] size [64, 64] to %p#0 size [4096] : memref<64x64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>
      } {applicable_units = ["MNISU"], dataflow_scheduler.domain = #producers}
      ktdf.stage depends_in(%p#1) depends_out(%p#2) {
        ktdf.data_transfer from %p#0 size [4096] to %dst[0, 0] size [64, 64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 4096xf16>, memref<64x64xf16, "L1">
      } {applicable_units = ["MNILU"], dataflow_scheduler.domain = #consumers}
    }
    return
  }

  func.func @local_chain(%src: memref<1x64xf16, "L1">, %dst: memref<1x64xf16, "L1">) {
    ktdf.pipeline {
      %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1, %t2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %v = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %o = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%v : tensor<64xf16>) outs(%e : tensor<64xf16>) {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %o, %p#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        ktdf.data_transfer from %p#1 size [64] to %dst[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
