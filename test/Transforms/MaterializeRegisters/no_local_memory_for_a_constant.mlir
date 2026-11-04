// A constant arithmetic reads on a compute with no local memory has no register
// to go in.

// RUN: dataflow-scheduler-opt %s -materialize-registers -verify-diagnostics

#id = affine_map<(d0) -> (d0)>
module {
  func.func @no_local_memory(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant 1.000000e+01 : f16
    // expected-error@+1 {{no unambiguous local memory of "MPE_ARRAY" to hold a register for a constant}}
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>)
        attrs = {ktdf_arch.maps_to = "MPE_ARRAY"} {
    ^bb0(%x: f16, %out: f16):
      %sum = arith.addf %x, %c : f16
      linalg.yield %sum : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
