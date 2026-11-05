// Verify the "arbitrary computation chain" code path (§3.1):
// the $base CMV offset is an arith.addi (or similar) SSA computation rather
// than a bare constant or function argument.
//
// The pass must carry the block-argument dependency (%offset_arg) that the
// offset's SSA def-use chain reaches out into the forwarded-argument list,
// and clone the rest of the chain (the arith.addi and the constant it adds)
// directly into the generated function — none of it is reconstructed in the
// orchestrator any more.
//
// Setup: the data tensor base = arith.addi(%c_base, %offset_arg) where
// %c_base is a compile-time constant and %offset_arg is a function argument.
// The resulting addi value is the CMV offset.
//
// Expected:
//   - Orchestrator: forwards %offset_arg verbatim to @_idx_to_addr — it no
//     longer materializes arith.addi/arith.constant itself.
//   - @_idx_to_addr: receives (%offset_arg: index) and reconstructs
//     arith.addi(%c_base, %offset_arg) internally to use as %base in the
//     linalg.generic body.
//   - @local_schedule_0: zero-offset CMV replacing the original addi-based offset.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// Orchestrator checks — %offset_arg forwarded raw, no arithmetic reconstructed
// ============================================================================

// CHECK-LABEL: func.func @orchestrator
// CHECK-SAME:      %[[OFFARG:.*]]: index
// CHECK:         call @local_schedule_0_idx_to_addr(%[[OFFARG]]) : (index) -> ()
// CHECK:         call @local_schedule_0(%[[OFFARG]]) : (index) -> ()

// ============================================================================
// @local_schedule_0_idx_to_addr — rebuilds the addi chain from its own arg
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr
// CHECK-SAME:        (%[[FOFFARG:.*]]: index)
// CHECK:           %[[CBASE:.*]] = arith.constant 1024 : index
// CHECK:           %[[FBASE_ELEMS:.*]] = arith.addi %[[CBASE]], %[[FOFFARG]]
// -- base and stride in 64-byte words: a runtime f16 count is divided by 32
// CHECK:           %[[C32:.*]] = arith.constant 32 : index
// CHECK:           %[[FBASE:.*]] = arith.divsi %[[FBASE_ELEMS]], %[[C32]] : index
// CHECK:           %[[FSTRIDE:.*]] = arith.constant 2 : index
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr

// ============================================================================
// @local_schedule_0 — zero-offset base in the rewritten data CMV
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[ABUFC0:.*]] = arith.constant 0 : index
// CHECK:           %[[IABC0:.*]] = arith.constant 0 : index
// CHECK:           %[[IAB:.*]] = ktdp_lowering.construct_memory_view %[[IABC0]], sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:        "IAB"
// CHECK:           ktdp.construct_memory_view %[[ABUFC0]], sizes: [2, 32], strides: [32, 1]
// CHECK:           %[[DATAC0:.*]] = arith.constant 0 : index
// CHECK:           %[[DATA:.*]] = ktdp.construct_memory_view %[[DATAC0]], sizes: [64, 2, 64]
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        intermediate_variables(%[[A5:[^,]*]], %[[A6:[^,]*]], %[[A7:[^,]*]], %[[A8:[^)]*]])
// CHECK-SAME:        base_ptr = %[[IAB]][%[[A5]], %[[A6]]]
// CHECK-SAME:        %[[DATA]][0, %[[C0]] + %[[A7]], %[[A8]]]

// ============================================================================
// Input IR — $base CMV offset is computed via arith.addi
// ============================================================================

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator takes a byte offset to add to the base address.
    func.func @orchestrator(%arg0: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0) : (index) -> ()
      return
    }
    func.func private @local_schedule_0(index)
  }

  module @local_schedule_0 {
    // The data tensor base address is computed as (c_base + %arg0) via arith.addi.
    // materializeDependency must clone both arith.constant 1024 and arith.addi
    // into the new function and into the orchestrator call site.
    func.func @local_schedule_0(%arg0: index) attributes {grid = [1]} {
      %c0     = arith.constant 0      : index
      %c100   = arith.constant 100    : index
      %c1024  = arith.constant 1024   : index
      %c10000 = arith.constant 10000  : index

      // Computed base address: static part + runtime offset.
      %data_base = arith.addi %c1024, %arg0 : index

      // Index memref: constant base, static strides.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data memref: base = %data_base (arith.addi result).
      %desc_1 = ktdp.construct_memory_view %data_base, sizes: [64, 2, 64], strides: [64, 4096, 1] {
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
