// RUN: dataflow-scheduler-opt %s -address-assignment | FileCheck %s

// SFU_REG is a compute-unit-local memory declared inside a yield-wrapper
// `group { kind = "SFU" } share()` in sample_device. This test pins that
// address assignment treats it like any other memory space: no register-file
// special case and no skip list are needed.
//
// It also pins the accounting model. sample_device declares #SFU_REG twice
// (once per sub-core), but capacity and the bump pointer are keyed by memory
// *kind*, not by instance, so both allocations below come out of one shared
// 2048-byte budget with one running offset. Each corelet therefore uses the
// same assigned offset in its own private register file; the offset itself is
// a plain constant, not a per-corelet value.

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  // CHECK-LABEL: @sfu_reg_allocations
  func.func @sfu_reg_allocations() attributes {grid = [2]} {
    // 1x64xf16 = 128 bytes; first allocation starts at offset 0.
    // CHECK: %[[ADDR0:.*]] = arith.constant 0 : index
    // CHECK: %[[MEM0:.*]] = builtin.unrealized_conversion_cast %[[ADDR0]] : index to memref<1x64xf16, "SFU_REG">
    %0 = memref.alloc() : memref<1x64xf16, "SFU_REG">

    // Second allocation bumps past the first within the same per-kind pool.
    // CHECK: %[[ADDR1:.*]] = arith.constant 128 : index
    // CHECK: %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[ADDR1]] : index to memref<1x64xf16, "SFU_REG">
    %1 = memref.alloc() : memref<1x64xf16, "SFU_REG">

    return
  }
}
