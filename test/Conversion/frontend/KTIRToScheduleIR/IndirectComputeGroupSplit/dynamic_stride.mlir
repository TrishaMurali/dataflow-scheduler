// Verify the dynamic-stride code path (§3.1):
// static_strides[D] == ShapedType::kDynamic so the stride comes from the
// CMV's $strides SSA operand list at the correct dynamic index.
//
// Setup: the data tensor ($base) has a dynamic stride at dim 0 (the indirect
// dimension D=0) passed as a function argument %arg1 : index.  The pass must:
//   1. Extract %arg1 as the stride value (not emit arith.constant).
//   2. Emit func.call @..._idx_to_addr(%arg1) — base is still a compile-time
//      constant, so only the stride is forwarded.
//   3. Forward %arg1 as %stride in the new function's body.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// Orchestrator checks — %arg1 (the dynamic stride) forwarded to the new call
// ============================================================================

// CHECK-LABEL: func.func @orchestrator
// CHECK-SAME:      %[[DBASE:.*]]: index, %[[DSTRIDE:.*]]: index
// CHECK:         call @local_schedule_0_idx_to_addr(%[[DSTRIDE]]) : (index) -> ()
// CHECK:         call @local_schedule_0(%[[DBASE]], %[[DSTRIDE]]) : (index, index) -> ()

// ============================================================================
// @local_schedule_0_idx_to_addr — %stride is the function arg value, %base is
// a constant cloned inside the function
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr
// CHECK-SAME:        (%[[FSTRIDE_ELEMS:.*]]: index)
// -- base and stride in 64-byte words: a runtime f16 count is divided by 32
// CHECK:           %[[FBASE:.*]] = arith.constant 32 : index
// CHECK:           %[[C32:.*]] = arith.constant 32 : index
// CHECK:           %[[FSTRIDE:.*]] = arith.divsi %[[FSTRIDE_ELEMS]], %[[C32]] : index
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr

// ============================================================================
// @local_schedule_0 — zero-offset base in the rewritten module
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        base_ptr =

// ============================================================================
// Input IR — dim 0 stride of desc_1 is a dynamic function argument
// ============================================================================

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator receives (%base_addr: index, %row_stride: index).
    // %row_stride is the dynamic stride along the indirect dimension of desc_1.
    func.func @orchestrator(%arg0: index, %arg1: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0, %arg1) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }

  module @local_schedule_0 {
    // data tensor stride at dim 0 (the indirect dimension) is dynamic (%arg1).
    // static_strides[0] = kDynamic; the pass must read baseCMV.getStrides()[0].
    func.func @local_schedule_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      %c0     = arith.constant 0      : index
      %c100   = arith.constant 100    : index
      %c1024  = arith.constant 1024   : index
      %c10000 = arith.constant 10000  : index

      // Index memref: static strides.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data memref: stride[0] (the indirect dim D=0) is dynamic (%arg1).
      // The static_strides array has kDynamic at position 0; the other entries
      // are static (4096 at dim 1, 1 at dim 2).
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [%arg1, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      %tmp1 = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      %12 = tensor.empty() : tensor<2x32x2x64xf16>
      %13 = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%tmp1_0 : tensor<2x32x2x64xf16>) outs(%12 : tensor<2x32x2x64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %cf10 = arith.constant 10.000000e+00 : f16
          %14 = arith.addf %in, %cf10 : f16
          linalg.yield %14 : f16
      } -> tensor<2x32x2x64xf16>

      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %13, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
