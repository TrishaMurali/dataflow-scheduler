// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-addr-buf-legalization %s | FileCheck %s --implicit-check-not=parallel_loop

// IAB view hoisting when the view's offset is defined *inside* the nest.
//
// The hoisted ind_addr_buf memory view is loop-invariant, but that is not the
// same as its operands being *defined* outside the loop nest: here the offset
// (%off) sits after the first construct_access_tile, so it falls inside the
// splice range and lands in the loop body along with the rest of the code.
// Hoisting the view alone would leave %off behind, breaking dominance twice
// over -- once for the hoisted view itself, and once for the addr_buf view
// that shares %off and is relocated into the entry loop's scf.if guard (the
// guard precedes the position %off would be left at).  Both definitions must
// travel with the view.

// The output verifying at all is half the assertion here: were %off left in
// the loop body, the IR below would fail MLIR's dominance check.

// CHECK-LABEL: func.func @in_nest_iab_offset
// The IAB view is above the nest, and %[[OFF]] (hoisted with it) is its offset.
// CHECK:      %[[IAB_VIEW:.+]] = ktdp_lowering.construct_memory_view %[[OFF:.+]], sizes: [32],
// CHECK-SAME:   memory_space = "IAB"
// Both loops come after, so %[[OFF]] dominates the whole nest.
// CHECK:      scf.for %[[I1:.+]] = %c0{{.*}} to %c2 step %c1
// CHECK:        scf.for %[[I2:.+]] = %c0{{.*}} to %c32 step %c1{{.*}} {
// CHECK:          %[[EQ0:.+]] = arith.cmpi eq, %[[I2]], %c0
// CHECK:          scf.if %[[EQ0]] {
// The relocated addr_buf view reads the same hoisted %[[OFF]] from inside the guard.
// CHECK:            ktdp.construct_memory_view %[[OFF]], sizes: [2, 32],
// CHECK:            ktdp.construct_access_tile {{.*}}{{\[}}%[[I1]], {{.*}}{{\]}}
// CHECK-SAME:         -> !ktdp.access_tile<1x32xindex>
// CHECK:            ktdp.load {{.*}} <1x32xindex> -> tensor<1x32xindex>
// CHECK:            tensor.collapse_shape {{.*}} tensor<1x32xindex> into tensor<32xindex>
// CHECK:            ktdp.construct_access_tile %[[IAB_VIEW]][%c0
// CHECK:            ktdp.store {{.*}} tensor<32xindex>, <32xindex>
// CHECK:          }
// CHECK:          ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:       base_ptr = %[[IAB_VIEW]][%[[I2]]]
// CHECK-SAME:       -> !ktdp.access_tile<2x64xindex>
// CHECK:          ktdp.store {{.*}} tensor<2x64xf16>, <2x64xindex>
// CHECK:        }
// CHECK:      }

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @in_nest_iab_offset(%arg5: index, %arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index

    // Source data load + compute precede the addr_buf fill (scatter order).
    // %src_at is the earliest access tile, i.e. the start of the splice range.
    %src = ktdp.construct_memory_view %c10000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %src_at = ktdp.construct_access_tile %src[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set4}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    %data = ktdp.load %src_at
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
    %empty = tensor.empty() : tensor<2x32x2x64xf16>
    %computed = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%data : tensor<2x32x2x64xf16>)
        outs(%empty : tensor<2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %added = arith.addf %in, %cf10 : f16
      linalg.yield %added : f16
    } -> tensor<2x32x2x64xf16>

    // %off is inside the splice range and feeds both the IAB view (hoisted
    // above the nest) and the addr_buf view (relocated into the scf.if guard).
    %off = arith.constant 0 : index
    %iab_mv = ktdp_lowering.construct_memory_view %off,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">
    %addr_buf = ktdp.construct_memory_view %off,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %addr_buf_at = ktdp.construct_access_tile %addr_buf[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set2}
        : memref<2x32xindex, #ktdp.memory_space<global>>
        -> !ktdp.access_tile<2x32xindex>
    %addr_tensor = ktdp.load %addr_buf_at
        : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>
    %iab_at = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set2}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_tensor, %iab_at
        : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    %dst = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %dst[(%c0), (%arg7), (%arg8)]
        {variables_space_order = #map,
         variables_space_set = #set4}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
        -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %computed, %tile
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
