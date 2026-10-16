// A constant the mapping marks is handed to a template call as its register.

// RUN: dataflow-scheduler-opt %s -materialize-registers | FileCheck %s

// CHECK-LABEL: func.func @tagged_constant_read_by_a_template(
// CHECK:        %[[C:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 2.000000e+00 : f16
// CHECK-NEXT:   %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   linalg.fill ins(%[[C]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:   %[[OUT:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%{{.*}}: f16, %{{.*}}: f16):
// CHECK-NEXT:     ktdf.opaque "fake_template"
// CHECK-NEXT:       ins(%[[REG]]: memref<64xf16, "SFU_REG">)
// CHECK-NEXT:       outs(%[[OUT]]: memref<64xf16, "SFU_REG">)
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[OUT]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:     linalg.yield %[[V]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @tagged_constant_read_by_a_template(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 2.000000e+00 : f16
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %reg = memref.alloca() : memref<f16, "SFU_REG">
      ktdf.opaque "fake_template" ins(%c : f16) outs(%reg : memref<f16, "SFU_REG">)
      %v = memref.load %reg[] : memref<f16, "SFU_REG">
      linalg.yield %v : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
