// RUN: dataflow-scheduler-opt -lower-init-register %s | FileCheck %s

// At function scope, the register is allocated where it was, and a pipeline of
// its own fills it right after. Its users read the allocation.

// CHECK-LABEL:   func.func @at_function_scope(
// CHECK-SAME:        %[[BUF:.*]]: memref<64xf16, "L1">) {
// CHECK-NEXT:      %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:          linalg.fill ins(%[[CST]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:        } {applicable_units = ["SFU"]}
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:          %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:          linalg.generic {{.*}} outs(%[[BUF]] : memref<64xf16, "L1">) {
// CHECK-NEXT:          ^bb0(%[[OUT:.*]]: f16):
// CHECK-NEXT:            %[[ZERO:.*]] = memref.load %[[REG]][%[[C0]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:            %[[SUM:.*]] = arith.addf %[[OUT]], %[[ZERO]] : f16
// CHECK-NEXT:            linalg.yield %[[SUM]] : f16
// CHECK-NEXT:          }
// CHECK-NEXT:        } {applicable_units = ["SFU"]}
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// In a stage, the register is allocated and filled where it was, by that
// stage. No pipeline is added.

// CHECK-LABEL:   func.func @in_a_stage(
// CHECK-SAME:        %[[BUF:.*]]: memref<64xf16, "L1">) {
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:          %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[CST:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:          %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:          linalg.fill ins(%[[CST]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:          linalg.generic {{.*}} outs(%[[BUF]] : memref<64xf16, "L1">) {
// CHECK-NEXT:          ^bb0(%[[OUT:.*]]: f16):
// CHECK-NEXT:            %[[ONE:.*]] = memref.load %[[REG]][%[[C0]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:            %[[SUM:.*]] = arith.addf %[[OUT]], %[[ONE]] : f16
// CHECK-NEXT:            linalg.yield %[[SUM]] : f16
// CHECK-NEXT:          }
// CHECK-NEXT:        } {applicable_units = ["SFU"]}
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

#map = affine_map<(d0) -> (d0)>

module {
  func.func @at_function_scope(%buf: memref<64xf16, "L1">) {
    %reg = ktdf.init_register 0.0 : f16 on "SFU" -> memref<64xf16, "SFU_REG">
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        %c0 = arith.constant 0 : index
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%buf : memref<64xf16, "L1">) {
        ^bb0(%out: f16):
          %zero = memref.load %reg[%c0] : memref<64xf16, "SFU_REG">
          %sum = arith.addf %out, %zero : f16
          linalg.yield %sum : f16
        }
      } {applicable_units = ["SFU"]}
    }
    return
  }

  func.func @in_a_stage(%buf: memref<64xf16, "L1">) {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        %c0 = arith.constant 0 : index
        %reg = ktdf.init_register 1.0 : f16 on "SFU" -> memref<64xf16, "SFU_REG">
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%buf : memref<64xf16, "L1">) {
        ^bb0(%out: f16):
          %one = memref.load %reg[%c0] : memref<64xf16, "SFU_REG">
          %sum = arith.addf %out, %one : f16
          linalg.yield %sum : f16
        }
      } {applicable_units = ["SFU"]}
    }
    return
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
