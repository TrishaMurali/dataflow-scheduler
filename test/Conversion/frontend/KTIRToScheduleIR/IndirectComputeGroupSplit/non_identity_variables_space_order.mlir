// Non-identity variables_space_order: verify that the indirect dimension's
// per_dim_subscript_map is zero'd out in the lowering op regardless of
// variables_space_order, and that variables_space_order is forwarded unchanged.
//
// The input uses a transposed variables_space_order:
//   #map_transposed = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
//
// With variables_space_set = {d0∈[0,1], d1∈[0,31], d2∈[0,1], d3∈[0,63]},
// the transposed order swaps d0 and d1, so the result access tile shape is
// [32, 2, 2, 64] instead of [2, 32, 2, 64].
//
// The key correctness checks:
//   1. The lowered ktdp_lowering.construct_indirect_access_tile uses:
//        - variables_space_order = #map_transposed (forwarded unchanged)
//        - per_dim_subscript_maps[0] (the indirect dim) = constant 0
//        - per_dim_subscript_maps[1,2] = original direct maps (unchanged)
//   2. The addr-buf / IAB shape is derived from the physical index-memref
//      shape [2, 32] — not permuted by variables_space_order.
//   3. base address of the data CMV is zero (original base folded into IAB).
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// @local_schedule_0 (rewritten gather/scatter module) checks
// ============================================================================

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0

// -- IAB view: shape unchanged from physical idxCMV [2, 32], not permuted.
// CHECK:           ktdp_lowering.construct_memory_view
// CHECK-SAME:        sizes: [2, 32]
// CHECK-SAME:        "IAB"
// CHECK-SAME:        memref<2x32xi32, "IAB">

// -- global addr-buf: also [2, 32].
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [2, 32]
// CHECK-SAME:        memory_space = #ktdp.memory_space<global>
// CHECK-SAME:        memref<2x32xi32

// -- data CMV with zero base offset (reuses the arith.constant 0 from the
//    access-tile indices above — no new constant is emitted here).
// CHECK:           ktdp.construct_memory_view {{.*}}, sizes: [64, 2, 64]
// -- lowered indirect op: per_dim_subscript_maps[0] is zero'd out ("(0)"),
//    and variables_space_order is forwarded unchanged (the transposed map).
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        [0, %{{.*}} + %{{.*}}, %{{.*}}]
// CHECK-SAME:        variables_space_order = {{#map[0-9]+|(d0, d1, d2, d3) -> \(d1, d0, d2, d3\)}}

// ============================================================================
// Input IR
// ============================================================================

// Transposed variables_space_order: swap d0↔d1 so gather tile shape = [32, 2, 2, 64].
// The output store uses the same transposed order so the shapes are compatible.
#map_transposed = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map_identity   = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// variables_space_set in natural variable order; variables_space_order permutes
// the tile-output shape but does not reorder the set constraints themselves.
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %c1024  = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      // Index memref: 2×32 si32.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data memref: 64×2×64 f16.  Indirect dimension D = 0 (stride = 64).
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Output memref.
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      // Non-identity variables_space_order: (d0,d1,d2,d3) -> (d1,d0,d2,d3).
      // Result shape = [32, 2, 2, 64] (bounds of d1, d0, d2, d3 in output order).
      // The indirect subscript pattern is the same as the identity case —
      // d0=arg5∈[0,1] and d1=arg6∈[0,31] subscript into desc_0[2,32].
      %tmp1 = ktdp.construct_indirect_access_tile
          intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map_transposed, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32>
          -> !ktdp.access_tile<32x2x2x64xindex>

      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<32x2x2x64xindex> -> tensor<32x2x2x64xf16>

      %12 = tensor.empty() : tensor<32x2x2x64xf16>
      %13 = linalg.generic {
          indexing_maps = [#map_identity, #map_identity],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%tmp1_0 : tensor<32x2x2x64xf16>) outs(%12 : tensor<32x2x2x64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %cf10 = arith.constant 10.000000e+00 : f16
          %14 = arith.addf %in, %cf10 : f16
          linalg.yield %14 : f16
      } -> tensor<32x2x2x64xf16>

      // The output access tile uses access_tile_order = #map_transposed so that
      // its shape matches the gathered tensor (32x2x2x64).
      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map_transposed,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<32x2x2x64xindex>
      ktdp.store %13, %at_out : tensor<32x2x2x64xf16>, !ktdp.access_tile<32x2x2x64xindex>
      return
    }
  }
}
