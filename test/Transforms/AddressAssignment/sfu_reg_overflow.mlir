// RUN: dataflow-scheduler-opt %s -address-assignment -verify-diagnostics

// SFU_REG declares `size = 2048` in sample_device. Pin that the declared
// capacity is actually enforced by the allocator, so an oversized register-file
// buffer produces a clean diagnostic rather than silently overflowing into
// whatever follows it.
//
// The `group { kind = "SFU" }` wrapper does not hide the memory from the
// memory-tree walk, which is why the capacity is found at all.

// expected-error @below {{AddressAssignment: failed to assign addresses for 1 allocation(s)}}
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  func.func @sfu_reg_overflow() attributes {grid = [2]} {
    // 1x2048xf16 = 4096 bytes, twice the declared 2048-byte capacity.
    // expected-error @below {{Failed to allocate "SFU_REG" memory: Allocation of 4096 bytes (aligned to 1) exceeds "SFU_REG" capacity (2048 bytes available, 0 bytes already allocated)}}
    %0 = memref.alloc() : memref<1x2048xf16, "SFU_REG">
    return
  }
}
