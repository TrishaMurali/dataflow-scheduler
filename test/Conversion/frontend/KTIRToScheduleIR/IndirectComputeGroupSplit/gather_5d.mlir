// 5-dimensional gather test: verify IndirectComputeGroupSplit with 5D tensors.
//
// Extends gather_basic.mlir to test higher rank tensors (5D index memref and 5D output).
// Tests the pass infrastructure with more dimensions while maintaining the core
// split structure.
//
// Tensor dimensions:
//   - Index memref: 2×4×3×2×8 si32 (5D, indirect dim 0)
//   - Data memref: 128×16×32 f16 (3D, indirect on dim 0)
//   - Output memref: 2×4×3×2×8×16×32 f16 (7D)
//
// Expected structural invariants:
//   1. Orchestrator calls @local_schedule_0_idx_to_addr() before @local_schedule_0()
//   2. @local_schedule_0_idx_to_addr module:
//        - 5D index memref loaded and converted to addresses
//        - linalg.generic applies spyreop.idx32toaddr element-wise
//        - addresses stored into 5D address buffer
//   3. @local_schedule_0 module rewritten:
//        - IAB view created for the 5D address buffer
//        - ktdp_lowering.construct_indirect_access_tile replaces original op
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
// -- base and stride constants
// -- base and stride in 64-byte words: 2048 and 512 f16 elements
// CHECK:           %[[FBASE:.*]] = arith.constant 64 : index
// CHECK:           %[[FSTRIDE:.*]] = arith.constant 16 : index
// -- 5D index memref CMV
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 4, 3, 2, 8], strides: [192, 48, 16, 8, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x4x3x2x8xsi32
// -- 5D address buffer CMV (global, element type = i32)
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 4, 3, 2, 8], strides: [192, 48, 16, 8, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x4x3x2x8xi32
// -- linalg.generic with spyreop.idx32toaddr
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr
// -- store address tensor into address buffer
// CHECK:           ktdp.store
// CHECK:           return

// ============================================================================
// @local_schedule_0 (gather module) checks
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// -- 5D IAB memory view
// CHECK:           ktdp_lowering.construct_memory_view
// CHECK-SAME:        "IAB"
// CHECK-SAME:        memref<2x4x3x2x8xi32, "IAB">
// -- global 5D addr buf CMV for loading into IAB
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 4, 3, 2, 8], strides: [192, 48, 16, 8, 1]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x4x3x2x8xi32
// -- load addr tensor from global buffer → store into IAB
// CHECK:           ktdp.load
// CHECK:           ktdp.store
// -- zero-offset data memref (3D)
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [128, 16, 32]
// -- lowered indirect op with IAB
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        base_ptr =
// -- original compute chain is intact (5D output)
// CHECK:           ktdp.load
// CHECK:           linalg.generic
// CHECK:           ktdp.store

// ============================================================================
// Input IR (output of ComputeGroupExtraction)
// ============================================================================

#map_5d  = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map_7d  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map_3d  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#set_5d_idx = affine_set<(d0, d1, d2, d3, d4) :
  (d0 >= 0, -d0 + 1 >= 0,
   d1 >= 0, -d1 + 3 >= 0,
   d2 >= 0, -d2 + 2 >= 0,
   d3 >= 0, -d3 + 1 >= 0,
   d4 >= 0, -d4 + 7 >= 0)>

#set_3d_data = affine_set<(d0, d1, d2) :
  (d0 >= 0, -d0 + 127 >= 0,
   d1 >= 0, -d1 + 15 >= 0,
   d2 >= 0, -d2 + 31 >= 0)>

#set_7d_out = affine_set<(d0, d1, d2, d3, d4, d5, d6) :
  (d0 >= 0, -d0 + 1 >= 0,
   d1 >= 0, -d1 + 3 >= 0,
   d2 >= 0, -d2 + 2 >= 0,
   d3 >= 0, -d3 + 1 >= 0,
   d4 >= 0, -d4 + 7 >= 0,
   d5 >= 0, -d5 + 15 >= 0,
   d6 >= 0, -d6 + 31 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator child module (no sym_name).
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    // 5D gather compute group.
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0    = arith.constant 0     : index
      %c100  = arith.constant 100   : index
      %c2048 = arith.constant 2048  : index
      %c20000 = arith.constant 20000 : index

      // Index memref: 2×4×3×2×8 si32
      // Row-major strides: [192, 48, 16, 8, 1]
      %idx_desc = ktdp.construct_memory_view %c100,
        sizes: [2, 4, 3, 2, 8], strides: [192, 48, 16, 8, 1] {
        coordinate_set = #set_5d_idx,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x4x3x2x8xsi32>

      // Data memref: 128×16×32 f16
      // Indirect dimension D = 0 (stride[0] = 512)
      // Static strides: [512, 32, 1]
      %data_desc = ktdp.construct_memory_view %c2048,
        sizes: [128, 16, 32], strides: [512, 32, 1] {
        coordinate_set = #set_3d_data,
        memory_space = #ktdp.memory_space<global>
      } : memref<128x16x32xf16>

      // Output memref: 2×4×3×2×8×16×32 f16
      // Static strides calculated for row-major: [16384, 4096, 1024, 512, 64, 32, 1]
      %out_desc = ktdp.construct_memory_view %c20000,
        sizes: [2, 4, 3, 2, 8, 16, 32], strides: [16384, 4096, 1024, 512, 64, 32, 1] {
        coordinate_set = #set_7d_out,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x4x3x2x8x16x32xf16>

      // 5D indirect access tile: dim 0 of data_desc is indirect (via idx_desc)
      // Intermediate variables: %arg0-%arg4 for 5D index tensor, %arg5-%arg6 for data dims
      %tile_indirect = ktdp.construct_indirect_access_tile
          intermediate_variables(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
          %data_desc[
            ind(%idx_desc[%c0 + %arg0, %c0 + %arg1, %c0 + %arg2, %c0 + %arg3, %c0 + %arg4]),
            (%c0 + %arg5),
            (%arg6)
          ]
          {variables_space_order = #map_7d, variables_space_set = #set_7d_out}
          : memref<128x16x32xf16>, memref<2x4x3x2x8xsi32>
            -> !ktdp.access_tile<2x4x3x2x8x16x32xindex>

      // 5D gather: load from the indirect tile
      %gathered = ktdp.load %tile_indirect :
        !ktdp.access_tile<2x4x3x2x8x16x32xindex> -> tensor<2x4x3x2x8x16x32xf16>

      // Element-wise compute over 5D output
      %out_init = tensor.empty() : tensor<2x4x3x2x8x16x32xf16>
      %computed = linalg.generic {
          indexing_maps = [#map_7d, #map_7d],
          iterator_types = ["parallel", "parallel", "parallel", "parallel",
                           "parallel", "parallel", "parallel"]
        } ins(%gathered : tensor<2x4x3x2x8x16x32xf16>)
          outs(%out_init : tensor<2x4x3x2x8x16x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %cf5 = arith.constant 5.000000e+00 : f16
          %result = arith.addf %in, %cf5 : f16
          linalg.yield %result : f16
      } -> tensor<2x4x3x2x8x16x32xf16>

      // Direct store to 7D output
      %out_tile = ktdp.construct_access_tile
        %out_desc[%c0, %c0, %c0, %c0, %c0, %c0, %c0] {
        access_tile_order = #map_7d,
        access_tile_set = #set_7d_out
      } : memref<2x4x3x2x8x16x32xf16> -> !ktdp.access_tile<2x4x3x2x8x16x32xindex>

      ktdp.store %computed, %out_tile :
        tensor<2x4x3x2x8x16x32xf16>, !ktdp.access_tile<2x4x3x2x8x16x32xindex>

      return
    }
  }
}
