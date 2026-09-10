// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s

// Verifies round-trip parsing and printing of
// ktdp_lowering.construct_indirect_access_tile.

// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[SET3D:.*]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-DAG: #[[SET2D:.*]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// Affine maps for the per-dim subscripts used across tests.
// Unified dim ordering: (captured..., intermediate...).
//
// 3-dim base (memref<64x2x64xf16>) with 1 captured (%c0) + 2 iv (%arg7, %arg8):
//   dim 0 → %c0  : affine_map<(d0,d1,d2) -> (d0)>
//   dim 1 → %c0 + %arg7 : affine_map<(d0,d1,d2) -> (d0 + d1)>
//   dim 2 → %arg8 : affine_map<(d0,d1,d2) -> (d2)>
#set3d = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2d = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#map2  = affine_map<(d0, d1) -> (d0, d1)>
#map3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// -----------------------------------------------------------------------
// Test 1: 2-D IAB, both IAB subscripts are intermediate variables.
//
// After IndirectComputeGroupSplit (before capacity legalization):
//   - 4 intermediate vars: %arg5, %arg6 (IAB dims), %arg7, %arg8 (direct dims)
//   - 1 captured: %c0
//   - per_dim_subscript_maps (unified dims = 1 captured + 4 iv = 5):
//       dim 0: affine_map<(d0,d1,d2,d3,d4) -> (d0)>       (%c0)
//       dim 1: affine_map<(d0,d1,d2,d3,d4) -> (d0 + d3)>  (%c0 + %arg7)
//       dim 2: affine_map<(d0,d1,d2,d3,d4) -> (d4)>        (%arg8)
// -----------------------------------------------------------------------
// CHECK-LABEL: func @roundtrip_2d_iab(
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME:    [[IAB:%arg[0-9]+]]: memref<2x32xindex, "IAB">
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:   intermediate_variables([[IV0:%[a-z0-9_]+]], [[IV1:%[a-z0-9_]+]], [[IV2:%[a-z0-9_]+]], [[IV3:%[a-z0-9_]+]])
// CHECK-SAME:   base_ptr = [[IAB]]{{\[}}[[IV0]], [[IV1]]{{\]}}
// CHECK-SAME:   [[BASE]][[[C0]], [[C0]] + [[IV2]], [[IV3]]]
// CHECK-SAME:   variables_space_order
// CHECK-SAME:   variables_space_set
// CHECK-SAME:   : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32x2x64xindex>
func.func @roundtrip_2d_iab(
    %base : memref<64x2x64xf16>,
    %iab  : memref<2x32xindex, "IAB">,
    %c0   : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg5, %arg6, %arg7, %arg8)
      base_ptr = %iab[%arg5, %arg6]
      %base[%c0, %c0 + %arg7, %arg8]
      {variables_space_set = #set3d, variables_space_order = #map3}
      : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
      -> !ktdp.access_tile<2x32x2x64xindex>
  return
}

// -----------------------------------------------------------------------
// Test 2: 1-D IAB, IAB subscript is an intermediate variable.
//
// After capacity legalization: %arg5 (IAB row) absorbed into outer scf.for.
//   - 3 intermediate vars: %arg6 (IAB col), %arg7, %arg8
//   - 1 captured: %c0
// -----------------------------------------------------------------------
// CHECK-LABEL: func @roundtrip_1d_iab(
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME:    [[IAB:%arg[0-9]+]]: memref<32xindex, "IAB">
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:   intermediate_variables([[IV0:%[a-z0-9_]+]], [[IV1:%[a-z0-9_]+]], [[IV2:%[a-z0-9_]+]])
// CHECK-SAME:   base_ptr = [[IAB]]{{\[}}[[IV0]]{{\]}}
// CHECK-SAME:   [[BASE]][[[C0]], [[C0]] + [[IV1]], [[IV2]]]
// CHECK-SAME:   variables_space_order
// CHECK-SAME:   variables_space_set
// CHECK-SAME:   : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<32x2x64xindex>
func.func @roundtrip_1d_iab(
    %base : memref<64x2x64xf16>,
    %iab  : memref<32xindex, "IAB">,
    %c0   : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg6, %arg7, %arg8)
      base_ptr = %iab[%arg6]
      %base[%c0, %c0 + %arg7, %arg8]
      {variables_space_set = #set2d, variables_space_order = #map2}
      : memref<64x2x64xf16>, memref<32xindex, "IAB">
      -> !ktdp.access_tile<32x2x64xindex>
  return
}

// -----------------------------------------------------------------------
// Test 3: 2-D IAB where the first IAB subscript is a captured outer loop IV
// and the second is an intermediate variable (mixed).
//
// After capacity legalization: outer scf.for IV %i1 selects the IAB row
// (captured); %arg6 is the remaining intermediate variable for the IAB column.
//   - 3 intermediate vars: %arg6, %arg7, %arg8
//   - 2 captured: %i1 (outer IV), %c0
// -----------------------------------------------------------------------
// CHECK-LABEL: func @mixed_iab_subscripts(
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME:    [[IAB:%arg[0-9]+]]: memref<2x32xindex, "IAB">
// CHECK-SAME:    [[I1:%arg[0-9]+]]: index
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:   intermediate_variables([[IV0:%[a-z0-9_]+]], [[IV1:%[a-z0-9_]+]], [[IV2:%[a-z0-9_]+]])
// CHECK-SAME:   base_ptr = [[IAB]]{{\[}}[[I1]], [[IV0]]{{\]}}
// CHECK-SAME:   [[BASE]][[[C0]], [[C0]] + [[IV1]], [[IV2]]]
// CHECK-SAME:   variables_space_order
// CHECK-SAME:   variables_space_set
// CHECK-SAME:   : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<32x2x64xindex>
func.func @mixed_iab_subscripts(
    %base : memref<64x2x64xf16>,
    %iab  : memref<2x32xindex, "IAB">,
    %i1   : index,   // captured outer loop IV selecting IAB row
    %c0   : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg6, %arg7, %arg8)
      base_ptr = %iab[%i1, %arg6]
      %base[%c0, %c0 + %arg7, %arg8]
      {variables_space_set = #set2d, variables_space_order = #map2}
      : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
      -> !ktdp.access_tile<32x2x64xindex>
  return
}

// -----------------------------------------------------------------------
// Test 4: After per-entry legalization — IAB subscript is a captured scf.for
// IV (%i2); only direct-dimension intermediate vars remain.
//   - 2 intermediate vars: %arg7, %arg8
//   - 2 captured: %i2 (inner loop IV), %c0
// -----------------------------------------------------------------------
// CHECK-LABEL: func @per_entry_legalized(
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME:    [[IAB:%arg[0-9]+]]: memref<32xindex, "IAB">
// CHECK-SAME:    [[I2:%arg[0-9]+]]: index
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:   intermediate_variables([[IV0:%[a-z0-9_]+]], [[IV1:%[a-z0-9_]+]])
// CHECK-SAME:   base_ptr = [[IAB]]{{\[}}[[I2]]{{\]}}
// CHECK-SAME:   [[BASE]][[[C0]], [[C0]] + [[IV0]], [[IV1]]]
// CHECK-SAME:   variables_space_order
// CHECK-SAME:   variables_space_set
// CHECK-SAME:   : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<2x64xindex>
func.func @per_entry_legalized(
    %base : memref<64x2x64xf16>,
    %iab  : memref<32xindex, "IAB">,
    %i2   : index,   // captured inner loop IV (per-entry loop)
    %c0   : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg7, %arg8)
      base_ptr = %iab[%i2]
      %base[%c0, %c0 + %arg7, %arg8]
      {variables_space_set = #set2d, variables_space_order = #map2}
      : memref<64x2x64xf16>, memref<32xindex, "IAB">
      -> !ktdp.access_tile<2x64xindex>
  return
}
