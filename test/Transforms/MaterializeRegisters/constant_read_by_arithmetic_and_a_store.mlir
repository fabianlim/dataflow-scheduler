// A constant read by arithmetic and stored into a register of the body: the
// arithmetic reads a lane of a register of its own, and the store keeps the
// scalar, which then fills the register it was stored into.

// RUN: dataflow-scheduler-opt %s -materialize-registers | FileCheck %s

// CHECK-LABEL: func.func @constant_read_by_arithmetic_and_a_store(
// CHECK:        %[[C:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:   %[[C_REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   linalg.fill ins(%[[C]] : f16) outs(%[[C_REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:   %[[REG:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:   linalg.fill ins(%[[C]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:     %[[STORED:.*]] = memref.load %[[REG]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:     %[[LANE:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[C_REG]][%[[LANE]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:     %[[SUM:.*]] = arith.addf %[[X]], %[[V]] : f16
// CHECK-NEXT:     %[[MAX:.*]] = arith.maximumf %[[SUM]], %[[STORED]] : f16
// CHECK-NEXT:     linalg.yield %[[MAX]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @constant_read_by_arithmetic_and_a_store(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant 0.000000e+00 : f16
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %reg = memref.alloca() : memref<f16, "SFU_REG">
      memref.store %c, %reg[] : memref<f16, "SFU_REG">
      %v = memref.load %reg[] : memref<f16, "SFU_REG">
      %sum = arith.addf %x, %c : f16
      %max = arith.maximumf %sum, %v : f16
      linalg.yield %max : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
