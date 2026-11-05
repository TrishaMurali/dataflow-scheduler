// Verify how $ind_addr_buf_dim_positions is derived from the indirect
// dimension's entry in $per_dim_subscript_maps.
//
// The address buffer holds one resolved address per element of the index
// memref, laid out identically, so each IAB subscript has only to name the
// variable that selected that element.  The forms accepted are:
//
//   - a bare intermediate variable                        (%arg6      -> d2)
//   - a sum whose other terms are captured literal zeros  (%c0 + %arg5 -> d0 + d1)
//   - a subscript that is itself a captured literal zero  (%c0        -> d0)
//
// Rejected forms are covered by indirect_subscript_offset_error.mlir and
// indirect_subscript_arithmetic_error.mlir.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @zero_offset_and_bare() : () -> ()
      call @constant_subscript() : () -> ()
      return
    }
    func.func private @zero_offset_and_bare()
    func.func private @constant_subscript()
  }

  // ==========================================================================
  // Case 1: `[%c0 + %arg5, %arg6]` — a zero-offset sum and a bare intermediate.
  // Unified space is (%c0, %arg5, %arg6, %arg7, %arg8), so the indirect map is
  // `(d0..d4) -> (d0 + d1, d2)` and the positions are [1, 2] — %arg5 and %arg6.
  // ==========================================================================

  // The trailing brace keeps CHECK-LABEL off the @<name>_idx_to_addr sibling,
  // whose module name has this one as a prefix.
  // CHECK-LABEL: module @zero_offset_and_bare {
  // CHECK:         func.func @zero_offset_and_bare
  // CHECK:           %[[IAB:.*]] = ktdp_lowering.construct_memory_view
  // CHECK-SAME:        "IAB"
  // CHECK:           ktdp_lowering.construct_indirect_access_tile
  // CHECK-SAME:        intermediate_variables(%[[A5:[^,]*]], %[[A6:[^,]*]], %{{[^,]*}}, %{{[^)]*}})
  // CHECK-SAME:        base_ptr = %[[IAB]][%[[A5]], %[[A6]]]
  module @zero_offset_and_bare {
    func.func @zero_offset_and_bare() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %c1024  = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      %tile = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      %val = ktdp.load %tile : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %val, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }

  // ==========================================================================
  // Case 2: `[%c0, %arg6]` — the row subscript is a captured literal zero, so
  // every dimension it names is zero.  Naming that dimension is still correct:
  // it evaluates to 0, which is the row the subscript selects.
  // ==========================================================================

  // CHECK-LABEL: module @constant_subscript {
  // CHECK:         func.func @constant_subscript
  // -- the captured %c0, i.e. the first of the two zero constants in this
  //    function; the second is the one the rewrite materializes for its tiles.
  // CHECK:           %[[C0:.*]] = arith.constant 0 : index
  // CHECK:           %[[IAB2:.*]] = ktdp_lowering.construct_memory_view
  // CHECK-SAME:        "IAB"
  // CHECK:           ktdp_lowering.construct_indirect_access_tile
  // CHECK-SAME:        intermediate_variables(%{{[^,]*}}, %[[B6:[^,]*]], %{{[^,]*}}, %{{[^)]*}})
  // CHECK-SAME:        base_ptr = %[[IAB2]][%[[C0]], %[[B6]]]
  module @constant_subscript {
    func.func @constant_subscript() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %c1024  = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      %tile = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0, %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      %val = ktdp.load %tile : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %val, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
