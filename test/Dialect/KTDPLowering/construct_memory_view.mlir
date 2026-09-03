// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s

// Verifies round-trip parsing and printing of
// ktdp_lowering.construct_memory_view.

// CHECK-DAG: #[[SET1:.*]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK-DAG: #[[SET2:.*]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>

#set1 = affine_set<(d0)     : (d0 >= 0, -d0 + 31 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>

// ── Static 1-D IAB view ──────────────────────────────────────────────────────
// CHECK-LABEL: func @static_1d_iab(
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK:       ktdp_lowering.construct_memory_view [[C0]], sizes: [32], strides: [1]
// CHECK-SAME:    coordinate_set = #[[SET1]], memory_space = "IAB"
// CHECK-SAME:  : memref<32xindex, "IAB">
func.func @static_1d_iab(%c0 : index) {
  %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
      {coordinate_set = #set1, memory_space = "IAB"}
      : memref<32xindex, "IAB">
  return
}

// ── Static 2-D IAB view ──────────────────────────────────────────────────────
// CHECK-LABEL: func @static_2d_iab(
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK:       ktdp_lowering.construct_memory_view [[C0]], sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:    coordinate_set = #[[SET2]], memory_space = "IAB"
// CHECK-SAME:  : memref<2x32xindex, "IAB">
func.func @static_2d_iab(%c0 : index) {
  %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1]
      {coordinate_set = #set2, memory_space = "IAB"}
      : memref<2x32xindex, "IAB">
  return
}

// ── Dynamic sizes/strides with IAB ───────────────────────────────────────────
// CHECK-LABEL: func @dynamic_iab(
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: index
// CHECK-SAME:    [[SZ:%arg[0-9]+]]: index
// CHECK-SAME:    [[ST:%arg[0-9]+]]: index
// CHECK:       ktdp_lowering.construct_memory_view [[BASE]], sizes: {{\[}}[[SZ]]{{\]}}, strides: {{\[}}[[ST]]{{\]}}
// CHECK-SAME:    coordinate_set = #[[SET1]], memory_space = "IAB"
// CHECK-SAME:  : memref<?xindex, "IAB">
func.func @dynamic_iab(%base : index, %sz : index, %st : index) {
  %iab_mv = ktdp_lowering.construct_memory_view %base, sizes: [%sz], strides: [%st]
      {coordinate_set = #set1, memory_space = "IAB"}
      : memref<?xindex, "IAB">
  return
}

// ── Typed ktdp.memory_space attribute ────────────────────────────────────────
// Demonstrates that the op also accepts a Ktdp_MemorySpaceAttr, not only
// plain string attributes.
// CHECK-LABEL: func @typed_memory_space(
// CHECK-SAME:    [[C0:%arg[0-9]+]]: index
// CHECK:       ktdp_lowering.construct_memory_view [[C0]], sizes: [2, 32], strides: [32, 1]
// CHECK-SAME:    coordinate_set = #[[SET2]], memory_space = #ktdp.memory_space<global>
// CHECK-SAME:  : memref<2x32xindex, #ktdp.memory_space<global>>
func.func @typed_memory_space(%c0 : index) {
  %mv = ktdp_lowering.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1]
      {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
      : memref<2x32xindex, #ktdp.memory_space<global>>
  return
}
