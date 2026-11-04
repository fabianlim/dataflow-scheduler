// An index constant that arithmetic reads is left where it is.

// RUN: dataflow-scheduler-opt %s -materialize-registers | FileCheck %s

// CHECK-LABEL: func.func @index_constant_left_alone(
// CHECK-SAME:      %{{.*}}: tensor<64xf16>, %[[TABLE:.*]]: memref<65xf16, "L1">)
// CHECK-NEXT:   %{{.*}} = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%{{.*}}: f16, %{{.*}}: f16):
// CHECK-NEXT:     %[[I:.*]] = linalg.index 0 : index
// CHECK-NEXT:     %[[NEXT:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[TABLE]][%[[NEXT]]] : memref<65xf16, "L1">
// CHECK-NEXT:     linalg.yield %[[V]] : f16
// CHECK-NOT:    memref.alloc

#id = affine_map<(d0) -> (d0)>
module {
  func.func @index_constant_left_alone(%in: tensor<64xf16>, %table: memref<65xf16, "L1">) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c1 = arith.constant 1 : index
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %i = linalg.index 0 : index
      %next = arith.addi %i, %c1 : index
      %v = memref.load %table[%next] : memref<65xf16, "L1">
      linalg.yield %v : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
