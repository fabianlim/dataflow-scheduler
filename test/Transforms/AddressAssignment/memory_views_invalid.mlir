// RUN: dataflow-scheduler-opt %s -address-assignment -split-input-file -verify-diagnostics

// A view outside global memory whose base is not a constant cannot be kept
// clear of.

#set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  module @program {
    func.func @program(%addr: index) {
      // expected-error @below {{memory view outside global memory needs a constant base address to be kept clear of allocations}}
      %view = ktdp.construct_memory_view %addr, sizes: [1024], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1024xf16>
      %l1 = memref.memory_space_cast %view : memref<1024xf16> to memref<1024xf16, "L1">
      %0 = memref.alloc() : memref<128xf16, "L1">
      return
    }
  }
}

// -----

// A program whose allocations no longer fit above the view fails as any
// allocation that does not fit does. The view ends at 1048064 bytes of the
// 1048576 L1 has, leaving 512 for an allocation of 1024.

#set = affine_set<(d0) : (d0 >= 0, -d0 + 524031 >= 0)>
// expected-error @below {{AddressAssignment: failed to assign addresses for}}
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  module @program {
    func.func @program() {
      %c0 = arith.constant 0 : index
      %view = ktdp.construct_memory_view %c0, sizes: [524032], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<524032xf16>
      %l1 = memref.memory_space_cast %view : memref<524032xf16> to memref<524032xf16, "L1">
      // expected-error @below {{Failed to allocate "L1" memory: Allocation of 1024 bytes (aligned to 1) exceeds "L1" capacity (1048576 bytes of capacity, 1048064 bytes already allocated)}}
      %0 = memref.alloc() : memref<512xf16, "L1">
      return
    }
  }
}

// -----

// A view that does not fit in its memory at all is reported on the view.

#set = affine_set<(d0) : (d0 >= 0, -d0 + 524288 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  module @program {
    func.func @program() {
      %c0 = arith.constant 0 : index
      // expected-error @below {{Failed to keep "L1" memory view clear of allocations: Reservation of addresses below 1048578 exceeds "L1" capacity (1048576 bytes of capacity)}}
      %view = ktdp.construct_memory_view %c0, sizes: [524289], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<524289xf16>
      %l1 = memref.memory_space_cast %view : memref<524289xf16> to memref<524289xf16, "L1">
      %0 = memref.alloc() : memref<128xf16, "L1">
      return
    }
  }
}
