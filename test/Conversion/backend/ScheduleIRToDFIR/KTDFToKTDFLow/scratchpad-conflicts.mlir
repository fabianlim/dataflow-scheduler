// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering %s | FileCheck %s

// A token edge gets a signal only if the producer stage writes a memory that
// the consumer stage reads, as decided from the memrefs the stages access, not
// from the memories their units could reach. Fifos are not memory.

module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // Stages joined only by a fifo: the MNISU stage writes no memory, so there
  // is no signal, although MNISU can write DDR and MNILU can read it.

  // CHECK-LABEL: func.func @fifo_only(
  // CHECK:         ktdf_lowering.execute_on %[[SU:[[:alnum:]_]+]] {
  // CHECK-NEXT:      ktdf.data_transfer from %{{.*}} to %[[FIFO:.*]] size [64]
  // CHECK-NEXT:    }
  // CHECK-NEXT:    ktdf_lowering.execute_on %[[LU:[[:alnum:]_]+]] {
  // CHECK-NEXT:      ktdf.data_transfer from %[[FIFO]] size [64] to
  // CHECK-NOT:     ktdf_lowering.signal
  // CHECK:         return
  func.func @fifo_only(%src: memref<64xf16, "L1">, %dst: memref<64xf16, "L1">) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %fifo, %t = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token) {
        %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
        %k = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f, %k : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t) {
        ktdf.data_transfer from %src[%c0] size [64] to %fifo size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>
      } {applicable_units = ["MNISU"]}
      ktdf.stage depends_in(%t) depends_out(none) {
        ktdf.data_transfer from %fifo size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"MNISU" -> "MNILU", 64xf16>, memref<64xf16, "L1">
      } {applicable_units = ["MNILU"]}
    }
    return
  }

  // Stages joined by an L1 buffer: the MNILU stage writes it and the MNISU
  // stage reads it, so there is a signal.

  // CHECK-LABEL: func.func @l1_buffer(
  // CHECK:         ktdf_lowering.execute_on %[[LU:[[:alnum:]_]+]] {
  // CHECK-NEXT:      ktdf.data_transfer from %{{.*}} to %[[BUF:.*]]{{\[}}%{{.*}}] size [64]
  // CHECK-NEXT:    }
  // CHECK-NEXT:    ktdf_lowering.signal %[[LU]], %[[SU:[[:alnum:]_]+]]
  // CHECK-NEXT:    ktdf_lowering.execute_on %[[SU]] {
  // CHECK-NEXT:      ktdf.data_transfer from %[[BUF]]
  func.func @l1_buffer(%A: memref<64xf16, "DDR">, %C: memref<64xf16, "DDR">) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %buf, %t = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, "L1">
        %k = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t) {
        ktdf.data_transfer from %A[%c0] size [64] to %buf[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
      } {applicable_units = ["MNILU"]}
      ktdf.stage depends_in(%t) depends_out(none) {
        ktdf.data_transfer from %buf[%c0] size [64] to %C[%c0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
      } {applicable_units = ["MNISU"]}
    }
    return
  }
}
