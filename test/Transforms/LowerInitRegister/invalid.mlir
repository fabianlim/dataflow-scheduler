// RUN: dataflow-scheduler-opt -lower-init-register --split-input-file --verify-diagnostics %s

// Negative: the enclosing stage does not run on the unit.

module {
  func.func @unit_not_applicable() {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        // expected-error@+1 {{unit "SFU" is not an applicable unit of the enclosing stage}}
        %reg = ktdf.init_register 0.0 : f16 on "SFU" -> memref<64xf16, "SFU_REG">
      } {applicable_units = ["L1LU"]}
    }
    return
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}

// -----

// Negative: in a pipeline, but not in one of its stages.

module {
  func.func @outside_a_stage() {
    ktdf.pipeline {
      %reg = ktdf.private -> (memref<64xf16, "SFU_REG">) {
        // expected-error@+1 {{must be inside a stage of its pipeline}}
        %r = ktdf.init_register 0.0 : f16 on "SFU" -> memref<64xf16, "SFU_REG">
        ktdf.private_yield %r : memref<64xf16, "SFU_REG">
      }
    }
    return
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
