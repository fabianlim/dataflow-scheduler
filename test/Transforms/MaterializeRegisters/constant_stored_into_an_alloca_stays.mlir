// A constant stored into an alloca of the body fills that alloca in the stage,
// inside the loop, and hoist-constant-storage leaves it there. A constant the
// body's arithmetic reads gets an alloc of its own, filled outside the loop.

// RUN: dataflow-scheduler-opt %s -materialize-registers -hoist-constant-storage | FileCheck %s

// CHECK-LABEL: func.func @constant_stored_into_an_alloca_stays(
// CHECK:         %[[STORED:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:    %[[READ:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 2.000000e+00 : f16
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    %[[READ_REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:    ktdf.pipeline {
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        linalg.fill ins(%[[READ]] : f16) outs(%[[READ_REG]] : memref<64xf16, "SFU_REG">)
// CHECK:         scf.for
// CHECK-NEXT:      ktdf.pipeline {
// CHECK:             ktdf.stage
// CHECK-NEXT:          %[[LATCH:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:          linalg.fill ins(%[[STORED]] : f16) outs(%[[LATCH]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:          linalg.generic

#id = affine_map<(d0) -> (d0)>
module {
  func.func @constant_stored_into_an_alloca_stays(%in: tensor<64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %stored = arith.constant 1.000000e+00 : f16
    %read = arith.constant 2.000000e+00 : f16
    scf.for %i = %c0 to %c4 step %c1 {
      ktdf.pipeline {
        %token = ktdf.private -> (!ktdf.token) {
          %t = ktdf.create_token : !ktdf.token
          ktdf.private_yield %t : !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%token) {
          %init = tensor.empty() : tensor<64xf16>
          %result = linalg.generic {indexing_maps = [#id, #id],
                                    iterator_types = ["parallel"]}
              ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
          ^bb0(%x: f16, %out: f16):
            %latch = memref.alloca() : memref<f16, "SFU_REG">
            memref.store %stored, %latch[] : memref<f16, "SFU_REG">
            %latched = memref.load %latch[] : memref<f16, "SFU_REG">
            %sum = arith.addf %x, %read : f16
            %max = arith.maximumf %sum, %latched : f16
            linalg.yield %max : f16
          } -> tensor<64xf16>
        } {applicable_units = ["SFU"]}
      }
    }
    return
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
