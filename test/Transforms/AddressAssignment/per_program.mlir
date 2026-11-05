// RUN: dataflow-scheduler-opt %s -address-assignment | FileCheck %s

// Each program has the memories it allocates from to itself, so addresses start
// over per program rather than accumulating across them. Only what sits outside
// the programs shares a scope.
//
// Global memory is the exception: it is how one program hands its results to the
// next, so what one put there is still live when that one runs and its addresses
// carry on. The root of the memory tree says which resource that is, so a device
// naming it something other than DDR is covered too.
//
// Addresses count elements of the allocation's element type, as a memory
// view's offset does; the tracker underneath still works in bytes.

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  // Outside the programs, and so a scope of its own.
  // CHECK: %[[OUTER:.*]] = arith.constant 0 : index
  // CHECK: builtin.unrealized_conversion_cast %[[OUTER]] : index to memref<256xf16, "L1">
  func.func @outside() {
    %0 = memref.alloc() : memref<256xf16, "L1">
    return
  }

  module @program_a {
    // CHECK-LABEL: @program_a
    // CHECK: %[[A0:.*]] = arith.constant 0 : index
    // CHECK: builtin.unrealized_conversion_cast %[[A0]] : index to memref<128xf16, "L1">
    // CHECK: %[[A1:.*]] = arith.constant 128 : index
    // CHECK: builtin.unrealized_conversion_cast %[[A1]] : index to memref<64xf16, "L1">
    // CHECK: %[[AG:.*]] = arith.constant 0 : index
    // CHECK: builtin.unrealized_conversion_cast %[[AG]] : index to memref<512xf16, "DDR">
    func.func @program_a() {
      %0 = memref.alloc() : memref<128xf16, "L1">
      %1 = memref.alloc() : memref<64xf16, "L1">
      %2 = memref.alloc() : memref<512xf16, "DDR">
      return
    }
  }

  // The next program starts at zero again in L1 instead of after program_a's
  // 384 bytes, while its global allocation carries on from program_a's 1024
  // bytes, which is element 512 of the f16 allocation.
  module @program_b {
    // CHECK-LABEL: @program_b
    // CHECK: %[[B0:.*]] = arith.constant 0 : index
    // CHECK: builtin.unrealized_conversion_cast %[[B0]] : index to memref<32xf16, "L1">
    // CHECK: %[[BG:.*]] = arith.constant 512 : index
    // CHECK: builtin.unrealized_conversion_cast %[[BG]] : index to memref<256xf16, "DDR">
    func.func @program_b() {
      %0 = memref.alloc() : memref<32xf16, "L1">
      %1 = memref.alloc() : memref<256xf16, "DDR">
      return
    }
  }
}
