// RUN: dataflow-scheduler-opt %s -address-assignment | FileCheck %s

// Test basic L1 memory allocation tracking
// The MemoryTracker should assign sequential addresses starting from 0.
// Addresses count elements of the allocation's element type, as a memory
// view's offset does.

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @test_sequential_allocations() {
    // CHECK-LABEL: @test_sequential_allocations
    
    // First allocation should start at address 0
    // CHECK: %[[ADDR0:.*]] = arith.constant 0 : index
    // CHECK: %[[MEM0:.*]] = builtin.unrealized_conversion_cast %[[ADDR0]] : index to memref<128xf16, "L1">
    %0 = memref.alloc() : memref<128xf16, "L1">
    
    // Second allocation should start after first (128 f16 = 256 bytes = 128 elements)
    // CHECK: %[[ADDR1:.*]] = arith.constant 128 : index
    // CHECK: %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[ADDR1]] : index to memref<64xf16, "L1">
    %1 = memref.alloc() : memref<64xf16, "L1">
    
    // Third allocation should start after second (128 + 64 = 192 elements)
    // CHECK: %[[ADDR2:.*]] = arith.constant 192 : index
    // CHECK: %[[MEM2:.*]] = builtin.unrealized_conversion_cast %[[ADDR2]] : index to memref<32xf16, "L1">
    %2 = memref.alloc() : memref<32xf16, "L1">
    
    return
  }
}