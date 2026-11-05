// Verify the dynamic-base code path (§3.1): the $base CMV's offset is a
// function argument rather than a compile-time constant.
//
// When %c1024 (the data tensor base address) is replaced by a function
// argument %arg0 : index, the orchestrator must:
//   - carry that argument value forward into @local_schedule_0_idx_to_addr(%arg0)
//   - use it verbatim as the sole call operand — no constant materialisation
//
// @local_schedule_0_idx_to_addr must:
//   - accept (%base: index) as its only argument — the stride is still a
//     compile-time constant, so it is cloned inside the function, not forwarded
//   - forward %base directly into the linalg.generic body via arith.index_castui
//
// @local_schedule_0 must:
//   - replace the original %arg0-based CMV offset with %c0
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// Orchestrator checks — %arg0 is forwarded verbatim as the sole argument
// ============================================================================

// CHECK-LABEL: func.func @orchestrator
// CHECK-SAME:      %[[ARG0:.*]]: index
// CHECK:         call @local_schedule_0_idx_to_addr(%[[ARG0]]) : (index) -> ()
// CHECK:         call @local_schedule_0(%[[ARG0]]) : (index) -> ()

// ============================================================================
// @local_schedule_0_idx_to_addr — %base is the function arg, %stride is a
// constant cloned inside the function
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr
// CHECK-SAME:        (%[[FBASE_ELEMS:.*]]: index)
// -- base and stride in 64-byte words: a runtime f16 count is divided by 32
// CHECK:           %[[C32:.*]] = arith.constant 32 : index
// CHECK:           %[[FBASE:.*]] = arith.divsi %[[FBASE_ELEMS]], %[[C32]] : index
// CHECK:           %[[FSTRIDE:.*]] = arith.constant 2 : index
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr

// ============================================================================
// @local_schedule_0 — original non-zero base replaced by %c0
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// CHECK:           ktdp_lowering.construct_memory_view {{.*}}, sizes: [2, 32]
// CHECK-SAME:        "IAB"
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [64, 2, 64]

// ============================================================================
// Input IR — data tensor base address comes from function argument %arg0
// ============================================================================

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator takes %arg0 as the dynamic base address for the data tensor.
    func.func @orchestrator(%arg0: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0) : (index) -> ()
      return
    }
    func.func private @local_schedule_0(index)
  }

  module @local_schedule_0 {
    // The data tensor offset comes from a function argument (%arg0) rather than
    // a compile-time constant.  All other structure matches gather_basic.mlir.
    func.func @local_schedule_0(%arg0: index) attributes {grid = [1]} {
      %c0   = arith.constant 0   : index
      %c100 = arith.constant 100 : index
      %c10000 = arith.constant 10000 : index

      // Index memref: constant base, static strides.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data memref: base = %arg0 (dynamic), stride[0] = 64 (static indirect dim).
      %desc_1 = ktdp.construct_memory_view %arg0, sizes: [64, 2, 64], strides: [64, 4096, 1] {
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
