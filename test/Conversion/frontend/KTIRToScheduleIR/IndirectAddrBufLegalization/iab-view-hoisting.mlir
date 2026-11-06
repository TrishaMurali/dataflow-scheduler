// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-addr-buf-legalization %s | FileCheck %s --implicit-check-not=parallel_loop

// IAB view hoisting: verifies that the ind_addr_buf memref view is a pure
// descriptor hoisted above the entire window/entry loop nest (constructed
// exactly once, before the outer window loop), rather than threaded as an
// scf.for iter-arg. This uses a 2×32 IAB (same shape as basic-gather), so
// both the window loop and the entry loop fire.

// CHECK-LABEL: func.func @iab_view_hoisting
// IAB view hoisted outside the entire loop nest — no iter-arg needed.
// CHECK:      %[[IAB_VIEW:.+]] = ktdp_lowering.construct_memory_view %c0, sizes: [32],
// dropped: window dim (extent 2) vanishes entirely; only the per-entry bound survives.
// CHECK-SAME:   coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK-SAME:   memory_space = "IAB"
// Outer window loop, annotated parallel.
// CHECK:      scf.for %[[I1:.+]] = %c0{{.*}} to %c2 step %c1{{.*}}
// CHECK:        scf.for %[[I2:.+]] = %c0{{.*}} to %c32 step %c1{{.*}} {
// CHECK:          %[[EQ0:.+]] = arith.cmpi eq, %[[I2]], %c0
// CHECK:          scf.if %[[EQ0]] {
// CHECK:            ktdp.construct_access_tile {{.*}}{{\[}}%[[I1]], {{.*}}{{\]}}
// pinned: window dim (%[[I1]]) collapses to a single point; entry dim kept at 0..31.
// CHECK-SAME:         access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-SAME:         access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 31 >= 0)>
// CHECK-SAME:         -> !ktdp.access_tile<1x32xindex>
// CHECK:            ktdp.load {{.*}} <1x32xindex> -> tensor<1x32xindex>
// CHECK:            tensor.collapse_shape {{.*}} tensor<1x32xindex> into tensor<32xindex>
// CHECK:            ktdp.construct_access_tile %[[IAB_VIEW]][%c0
// kept: same access_tile_set/order as the IAB view's own (narrowed) coordinate_set/rank.
// CHECK-SAME:         access_tile_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:         access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK:            ktdp.store {{.*}} tensor<32xindex>, <32xindex>
// CHECK:          }
// CHECK:          ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:       base_ptr = %[[IAB_VIEW]][%[[I2]]]
// dropped: window+entry dims vanish (absorbed by base_ptr[iv]); direct dims kept unchanged.
// CHECK-SAME:       variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-SAME:       variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-SAME:       -> !ktdp.access_tile<2x64xindex>
// CHECK:          ktdp.load {{.*}} <2x64xindex> -> tensor<2x64xf16>
// CHECK:          linalg.generic
// CHECK:          ktdp.construct_access_tile {{.*}}{{\[}}%[[I1]], %[[I2]], {{.*}}{{\]}}
// pinned: window+entry dims collapse to a single point; the two direct dims stay full-range.
// CHECK-SAME:       access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-SAME:       access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-SAME:       -> !ktdp.access_tile<1x1x2x64xindex>
// CHECK:          ktdp.store {{.*}} tensor<1x1x2x64xf16>, <1x1x2x64xindex>
// CHECK:        }
// CHECK:      }

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @iab_view_hoisting(%arg5: index, %arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 2000 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %addr_buf_at = ktdp.construct_access_tile %addr_buf[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set3}
        : memref<2x32xindex, #ktdp.memory_space<global>>
        -> !ktdp.access_tile<2x32xindex>
    %addr_tensor_stick = ktdp.load %addr_buf_at
        : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>

    // IAB memory view (2×32 entries, pre-legalization)
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set3, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">
    %iab_at = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set3}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_tensor_stick, %iab_at
        : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_1[(%c0), (%arg7), (%arg8)]
        {variables_space_order = #map,
         variables_space_set = #set4}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
        -> !ktdp.access_tile<2x32x2x64xindex>

    %result = ktdp.load %tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %empty = tensor.empty() : tensor<2x32x2x64xf16>
    %computed = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%result : tensor<2x32x2x64xf16>)
        outs(%empty : tensor<2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %added = arith.addf %in, %cf10 : f16
      linalg.yield %added : f16
    } -> tensor<2x32x2x64xf16>

    %desc_2 = ktdp.construct_memory_view %c10000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %desc_2_at = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set4}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %computed, %desc_2_at
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
