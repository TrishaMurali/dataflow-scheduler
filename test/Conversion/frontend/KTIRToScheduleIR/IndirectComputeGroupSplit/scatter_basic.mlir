// Canonical 2×32×2×64 scatter: verify the full split structure produced by
// IndirectComputeGroupSplit on the scatter (indirect store) path.
//
// Symmetric to gather_basic.mlir: the indirect op appears as the operand of
// ktdp.store rather than ktdp.load.  The transformation is identical — the
// pass detects both directions via the shared ktdp.construct_indirect_access_tile.
//
// Input: output of ComputeGroupExtraction — @local_schedule_0 contains a
// direct ktdp.load (from the source), an element-wise linalg.generic, and an
// indirect ktdp.store (scatter to the destination).
//
// Expected structural invariants (§7.3):
//   1. Orchestrator calls @local_schedule_0_idx_to_addr() before
//      @local_schedule_0(). base (10240) and stride (64) are compile-time
//      constants here, so idx_to_addr takes no arguments.
//   2. @local_schedule_0_idx_to_addr module: index-to-address conversion for
//      the scatter destination addresses (same structure as gather).
//   3. @local_schedule_0 module: same IAB setup + zero-offset base rewrite, but
//      ktdp_lowering.construct_indirect_access_tile now drives a ktdp.store.
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
// @local_schedule_0_idx_to_addr module checks (scatter)
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr()
// -- base and stride, cloned directly into the function body
// -- base and stride in 64-byte words: 10240 and 64 f16 elements
// CHECK:           %[[FBASE:.*]] = arith.constant 320 : index
// CHECK:           %[[FSTRIDE:.*]] = arith.constant 2 : index
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xsi32
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xi32
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.load
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.store
// CHECK:           return

// ============================================================================
// @local_schedule_0 (scatter module) checks
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// CHECK:           ktdp_lowering.construct_memory_view
// CHECK-SAME:        "IAB"
// CHECK-SAME:        memref<2x32xi32, "IAB">
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xi32
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.load
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.store
// -- zero-offset data memref (reuses the arith.constant 0 from the access-tile
//    indices above — no new constant is emitted here)
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [64, 2, 64]
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        base_ptr =
// CHECK:           ktdp.store

// ============================================================================
// Input IR (output of ComputeGroupExtraction, scatter variant)
// ============================================================================

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator child module.
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    // Scatter: direct load from src, compute, indirect store to dst.
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0      : index
      %c100   = arith.constant 100    : index
      %c1000  = arith.constant 1000   : index
      %c10240 = arith.constant 10240  : index

      // Index memref: 2×32 si32.
      %desc_idx = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Source (direct) memref: 2×32×2×64 f16.
      %desc_src = ktdp.construct_memory_view %c1000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      // Destination (scatter target) memref: 64×2×64 f16, base = %c10240.
      // Indirect dimension D = 0 (stride[0] = 64 — the static stride for dim 0).
      %desc_dst = ktdp.construct_memory_view %c10240, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Direct load from source.
      %src_at = ktdp.construct_access_tile %desc_src[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      %src = ktdp.load %src_at : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      // Element-wise compute.
      %empty = tensor.empty() : tensor<2x32x2x64xf16>
      %result = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%src : tensor<2x32x2x64xf16>) outs(%empty : tensor<2x32x2x64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %cf10 = arith.constant 10.000000e+00 : f16
          %sum = arith.addf %in, %cf10 : f16
          linalg.yield %sum : f16
      } -> tensor<2x32x2x64xf16>

      // Indirect access tile for scatter destination (dim 0 is indirect).
      %dst_tile = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_dst[ind(%desc_idx[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      // Scatter: indirect store.
      ktdp.store %result, %dst_tile : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
