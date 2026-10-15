// A constant the kernel writes and arithmetic reads goes in a register in the
// memory local to the compute, and the add reads a lane out of it.

// RUN: dataflow-scheduler-opt %s -materialize-registers | FileCheck %s

// CHECK-LABEL: func.func @untagged_constant_read_by_arithmetic(
// CHECK:        %[[C:.*]] = arith.constant 1.000000e+01 : f16
// CHECK-NEXT:   %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   linalg.fill ins(%[[C]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[REG]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:     %[[SUM:.*]] = arith.addf %[[X]], %[[V]] : f16
// CHECK-NEXT:     linalg.yield %[[SUM]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @untagged_constant_read_by_arithmetic(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant 1.000000e+01 : f16
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %sum = arith.addf %x, %c : f16
      linalg.yield %sum : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
