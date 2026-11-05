// Verify that an index memory view whose base address is a function argument
// is supported: that block argument is collected the same way as any base
// address/stride dependency and becomes a forwarded argument of the
// generated idx_to_addr function — idx_view's chain is unioned with
// base/stride's chain, not treated specially.
//
// The data base (1024) and stride (64) here are compile-time constants, so
// they are cloned directly into the function body; only the index memref's
// %arg0 dependency is forwarded.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// Orchestrator checks — %arg0 forwarded verbatim as the sole argument
// ============================================================================

// CHECK-LABEL: func.func @orchestrator
// CHECK-SAME:      %[[ARG0:.*]]: index
// CHECK:         call @local_schedule_0_idx_to_addr(%[[ARG0]]) : (index) -> ()
// CHECK:         call @local_schedule_0(%[[ARG0]]) : (index) -> ()

// ============================================================================
// @local_schedule_0_idx_to_addr — %arg0 feeds the cloned index memref's base;
// data base/stride are constants cloned inside the function
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr
// CHECK-SAME:        (%[[FIDXBASE:.*]]: index)
// -- base and stride in 64-byte words: 1024 and 64 f16 elements
// CHECK:           %[[FBASE:.*]] = arith.constant 32 : index
// CHECK:           %[[FSTRIDE:.*]] = arith.constant 2 : index
// CHECK:           ktdp.construct_memory_view %[[FIDXBASE]], sizes: [2, 32], strides: [32, 1]
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr

// ============================================================================
// @local_schedule_0 — zero-offset base in the rewritten data CMV
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// CHECK:           ktdp_lowering.construct_memory_view {{.*}}, sizes: [2, 32]
// CHECK-SAME:        "IAB"
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [64, 2, 64]

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator(%arg0: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0) : (index) -> ()
      return
    }
    func.func private @local_schedule_0(index)
  }

  module @local_schedule_0 {
    // The index memref's base address is %arg0 rather than a constant.
    func.func @local_schedule_0(%arg0: index) attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c1024  = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      // Index memref: base = %arg0 (a function argument).
      %desc_0 = ktdp.construct_memory_view %arg0, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data memref: base = %c1024, indirect dimension D = 0.
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Output memref.
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      %tmp1 = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %tmp1_0, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
