// RUN: dataflow-scheduler-opt %s -address-assignment | FileCheck %s

// Test that allocations skip the memory's reserved region
// L1 reserves its first 4096 bytes, so the first f16 allocation starts at
// element 2048. Addresses count elements, as a memory view's offset does.

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device_reserved.mlir")
  func.func @test_reserved_allocations() {
    // CHECK-LABEL: @test_reserved_allocations

    // First allocation should start at the first unreserved address
    // CHECK: %[[ADDR0:.*]] = arith.constant 2048 : index
    // CHECK: %[[MEM0:.*]] = builtin.unrealized_conversion_cast %[[ADDR0]] : index to memref<128xf16, "L1">
    %0 = memref.alloc() : memref<128xf16, "L1">

    // Second allocation should start after first (2048 + 128 = 2176 elements)
    // CHECK: %[[ADDR1:.*]] = arith.constant 2176 : index
    // CHECK: %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[ADDR1]] : index to memref<64xf16, "L1">
    %1 = memref.alloc() : memref<64xf16, "L1">

    return
  }
}
