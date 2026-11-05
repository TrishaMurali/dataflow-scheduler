// Canonical 2×32×2×64 gather: verify the full split structure produced by
// IndirectComputeGroupSplit.
//
// Input: output of ComputeGroupExtraction — a top-level module containing an
// orchestrator child module and @local_schedule_0.  @local_schedule_0 contains
// a ktdp.construct_indirect_access_tile (gather direction: the tile is consumed
// by a ktdp.load).
//
// Expected structural invariants after the pass (§7.2):
//   1. Orchestrator calls @local_schedule_0_idx_to_addr() before
//      @local_schedule_0(). Base and stride are compile-time constants here,
//      so idx_to_addr takes no arguments — see computed_base.mlir /
//      dynamic_base.mlir / dynamic_stride.mlir for the forwarded-argument case.
//   2. A new @local_schedule_0_idx_to_addr module exists before @local_schedule_0:
//        - ktdp.construct_memory_view for the global address buffer
//        - ktdp.load of the index tensor → linalg.generic with spyreop.idx32toaddr
//        - ktdp.store into the address buffer
//   3. @local_schedule_0 module is rewritten in-place:
//        - ktdp_lowering.construct_memory_view for the IAB view (memory_space = "IAB")
//        - ktdp.load of the global addr buffer → ktdp.store into IAB
//        - ktdp.construct_memory_view with %c0 base for the zero-offset data memref
//        - ktdp_lowering.construct_indirect_access_tile replaces the indirect op
//        - original ktdp.load, linalg.generic, ktdp.store unchanged
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// Orchestrator checks
// ============================================================================

// CHECK-LABEL: func.func @orchestrator
// CHECK:         call @local_schedule_0_idx_to_addr() : () -> ()
// CHECK:         call @local_schedule_0() : () -> ()
// CHECK:         func.func private @local_schedule_0()
// CHECK:         func.func private @local_schedule_0_idx_to_addr()

// ============================================================================
// @local_schedule_0_idx_to_addr module checks
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr()
// -- base and stride, cloned directly into the function body
// -- base and stride in 64-byte words: 1024 and 64 f16 elements
// CHECK:           %[[FBASE:.*]] = arith.constant 32 : index
// CHECK:           %[[FSTRIDE:.*]] = arith.constant 2 : index
// -- index memref CMV (desc_0 reconstructed)
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xsi32
// -- address buffer CMV (global, element type = i32)
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xi32
// -- load index tensor
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.load
// -- linalg.generic with spyreop.idx32toaddr
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr
// -- store address tensor into address buffer
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.store
// CHECK:           return

// ============================================================================
// @local_schedule_0 (gather/scatter module) checks
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// -- IAB memory view (memory_space = "IAB")
// CHECK:           ktdp_lowering.construct_memory_view
// CHECK-SAME:        "IAB"
// CHECK-SAME:        memref<2x32xi32, "IAB">
// -- global addr buf CMV for loading into IAB
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xi32
// -- load addr tensor from global buffer → store into IAB
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.load
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.store
// -- zero-offset data memref (reuses the arith.constant 0 from the access-tile
//    indices above — no new constant is emitted here)
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [64, 2, 64]
// -- lowered indirect op: IAB for base_ptr
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        base_ptr =
// -- original compute chain is intact
// CHECK:           ktdp.load
// CHECK:           linalg.generic
// CHECK:           ktdp.store

// ============================================================================
// Input IR (output of ComputeGroupExtraction)
// ============================================================================

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  // Inline device declaration with indirect_address_buffer feature (entry_type = si32).
  // The pass reads the IAB entry type from this node to derive computeType = i32.
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator child module (no sym_name → identified as orchestrator).
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    // Extracted child module: contains the canonical gather indirect op.
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0    = arith.constant 0     : index
      %c100  = arith.constant 100   : index
      %c1024 = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      // Index memref: 2×32 si32, base = %c100, stride[0] = 32, stride[1] = 1.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data (embedding) memref: 64×2×64 f16, base = %c1024.
      // Indirect dimension D = 0 (stride[0] = 64 — the static stride for dim 0).
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Output memref: 2×32×2×64 f16.
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      // Indirect access tile: dim 0 of desc_1 is indirect (via desc_0).
      %tmp1 = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      // Gather: load from the indirect tile.
      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      // Element-wise compute.
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

      // Direct store to output.
      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %13, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }

}
