// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @reduction_iter_arg() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf16>
// CHECK-NEXT:     %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:     %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_3]]) -> (tensor<64xf16>) {
// CHECK-NEXT:       %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[EMPTY_0]] : tensor<64xf16>) outs(%[[VAL_1]] : tensor<64xf16>) {
// CHECK-NEXT:       ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16):
// CHECK-NEXT:         %[[ADDF_0:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f16
// CHECK-NEXT:         linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:       } -> tensor<64xf16>
// CHECK-NEXT:       scf.yield %[[GENERIC_0]] : tensor<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// Pass-00 KTIRLegalityCheck: a reduction scf.for whose tensor iter_arg is consumed by a
// linalg.generic reduction is ACCEPTED. Contrast reject-scf-for-iter-args.mlir, which rejects
// a general loop-carried iter_arg -- this reduction-accumulator shape is what the
// nested-pipeline reduction design relies on.
#map = affine_map<(d0) -> (d0)>
func.func @reduction_iter_arg() {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf16>
  %in = tensor.empty() : tensor<64xf16>
  %r = scf.for %i = %c0 to %c256 step %c1 iter_args(%acc = %cst) -> (tensor<64xf16>) {
    %g = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%acc : tensor<64xf16>) {
      ^bb0(%a: f16, %b: f16):
        %s = arith.addf %a, %b : f16
        linalg.yield %s : f16
    } -> tensor<64xf16>
    scf.yield %g : tensor<64xf16>
  }
  return
}
