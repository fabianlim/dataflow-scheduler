// RUN: dataflow-scheduler-opt %s -address-assignment -split-input-file | FileCheck %s

// A memory view outside global memory is how one program hands an intermediate
// to the next without it leaving the memory the programs otherwise have to
// themselves. It is not an allocation and its address was fixed by the kernel,
// so every program's allocations are kept above the end of the highest such
// view, the one that wrote it and the one that reads it alike.
//
// The view's base counts elements: 2x256x64 f16 at 0 ends at 65536 bytes.

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @producer {
// CHECK-NEXT:      func.func @producer() {
// CHECK-NEXT:        %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.memory_space<ct_local>} : memref<2x256x64xf16>
// CHECK-NEXT:        %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "L1">
// CHECK-NEXT:        %[[CONSTANT_1:.*]] = arith.constant 65536 : index
// CHECK-NEXT:        %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_1]] : index to memref<128xf16, "L1">
// CHECK-NEXT:        %[[CONSTANT_2:.*]] = arith.constant 65792 : index
// CHECK-NEXT:        %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_2]] : index to memref<64xf16, "L1">
// CHECK-NEXT:        %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_3]] : index to memref<512xf16, "DDR">
// CHECK-NEXT:        return
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// CHECK-LABEL:   module @consumer {
// CHECK-NEXT:      func.func @consumer() {
// CHECK-NEXT:        %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.memory_space<ct_local>} : memref<2x256x64xf16>
// CHECK-NEXT:        %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "L1">
// CHECK-NEXT:        %[[CONSTANT_1:.*]] = arith.constant 65536 : index
// CHECK-NEXT:        %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_1]] : index to memref<32xf16, "L1">
// CHECK-NEXT:        return
// CHECK-NEXT:      }
// CHECK-NEXT:    }


#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  module @producer {
    func.func @producer() {
      %c0 = arith.constant 0 : index
      %view = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<2x256x64xf16>
      %l1 = memref.memory_space_cast %view : memref<2x256x64xf16> to memref<2x256x64xf16, "L1">
      %0 = memref.alloc() : memref<128xf16, "L1">
      %1 = memref.alloc() : memref<64xf16, "L1">
      %2 = memref.alloc() : memref<512xf16, "DDR">
      return
    }
  }

  module @consumer {
    func.func @consumer() {
      %c0 = arith.constant 0 : index
      %view = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<2x256x64xf16>
      %l1 = memref.memory_space_cast %view : memref<2x256x64xf16> to memref<2x256x64xf16, "L1">
      %0 = memref.alloc() : memref<32xf16, "L1">
      return
    }
  }
}

// -----

// Views in different programs: every program starts above the highest end
// among all of them, including a program that has no view of its own. The
// highest is 512 f16 at element 4096, ending at (4096 + 512) * 2 = 9216 bytes.
// Where the views sit relative to each other is up to whoever placed them: one
// overlapping another, as program_b's second view does program_a's, is not an
// error.

#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>
#set2 = affine_set<(d0) : (d0 >= 0, -d0 + 511 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  module @program_a {
    // CHECK-LABEL: @program_a
    // CHECK: %[[A0:.*]] = arith.constant 9216 : index
    // CHECK: builtin.unrealized_conversion_cast %[[A0]] : index to memref<128xf16, "L1">
    func.func @program_a() {
      %c0 = arith.constant 0 : index
      %view = ktdp.construct_memory_view %c0, sizes: [1024], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<ct_local>} : memref<1024xf16>
      %l1 = memref.memory_space_cast %view : memref<1024xf16> to memref<1024xf16, "L1">
      %0 = memref.alloc() : memref<128xf16, "L1">
      return
    }
  }

  module @program_b {
    // CHECK-LABEL: @program_b
    // CHECK: %[[B0:.*]] = arith.constant 9216 : index
    // CHECK: builtin.unrealized_conversion_cast %[[B0]] : index to memref<64xf16, "L1">
    func.func @program_b() {
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %view = ktdp.construct_memory_view %c4096, sizes: [512], strides: [1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<512xf16>
      %l1 = memref.memory_space_cast %view : memref<512xf16> to memref<512xf16, "L1">
      %overlapping = ktdp.construct_memory_view %c256, sizes: [512], strides: [1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<512xf16>
      %l1_overlapping = memref.memory_space_cast %overlapping : memref<512xf16> to memref<512xf16, "L1">
      %0 = memref.alloc() : memref<64xf16, "L1">
      return
    }
  }

  // The floor applies to every program, since the views are surveyed once
  // per module.
  module @program_c {
    // CHECK-LABEL: @program_c
    // CHECK: %[[C0:.*]] = arith.constant 9216 : index
    // CHECK: builtin.unrealized_conversion_cast %[[C0]] : index to memref<32xf16, "L1">
    func.func @program_c() {
      %0 = memref.alloc() : memref<32xf16, "L1">
      return
    }
  }
}

// -----

// A view in global memory is left alone: global memory carries what earlier
// programs left in it already, and its base need not be a constant.

#set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  module @global_views {
    // CHECK-LABEL: @global_views
    // CHECK: %[[L0:.*]] = arith.constant 0 : index
    // CHECK: builtin.unrealized_conversion_cast %[[L0]] : index to memref<128xf16, "L1">
    // CHECK: %[[G0:.*]] = arith.constant 0 : index
    // CHECK: builtin.unrealized_conversion_cast %[[G0]] : index to memref<512xf16, "DDR">
    func.func @global_views(%addr: index) {
      %c64 = arith.constant 64 : index
      %view = ktdp.construct_memory_view %c64, sizes: [1024], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1024xf16>
      %ddr = memref.memory_space_cast %view : memref<1024xf16> to memref<1024xf16, "DDR">
      %input = ktdp.construct_memory_view %addr, sizes: [1024], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1024xf16>
      %ddr_input = memref.memory_space_cast %input : memref<1024xf16> to memref<1024xf16, "DDR">
      %0 = memref.alloc() : memref<128xf16, "L1">
      %1 = memref.alloc() : memref<512xf16, "DDR">
      return
    }
  }
}
