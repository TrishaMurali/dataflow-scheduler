// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-addr-buf-legalization %s | FileCheck %s

// Scatter whose source data is loaded twice from the *same* access tile (as
// in `addf %in_a, %in_b` over two loads of one tile). Pinning the source tile
// replaces it for every user, so the second load must reuse the first
// load's pinning instead of pinning another dimension of the already-pinned
// tile. Both loads therefore end up on one 1x1x2x64 tile, each followed by an
// identical collapse to 2x64.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:         scf.for [[I1:%arg[0-9]+]] = {{.*}} to %c2 step
// CHECK:           scf.for [[I2:%arg[0-9]+]] = %c0{{.*}} to %c32 step %c1{{.*}} {
// CHECK:             scf.if
// CHECK:             [[TILE:%.+]] = ktdp.construct_access_tile {{.*}}{{\[}}[[I1]], [[I2]], {{.*}}{{\]}}
// CHECK-SAME:          -> !ktdp.access_tile<1x1x2x64xindex>
// CHECK:             [[LOAD0:%.+]] = ktdp.load [[TILE]] : <1x1x2x64xindex> -> tensor<1x1x2x64xf16>
// CHECK:             [[COLLAPSE0:%.+]] = tensor.collapse_shape [[LOAD0]] {{\[\[}}0, 1, 2{{\]}}, [3{{\]\]}} : tensor<1x1x2x64xf16> into tensor<2x64xf16>
// CHECK:             [[LOAD1:%.+]] = ktdp.load [[TILE]] : <1x1x2x64xindex> -> tensor<1x1x2x64xf16>
// CHECK:             [[COLLAPSE1:%.+]] = tensor.collapse_shape [[LOAD1]] {{\[\[}}0, 1, 2{{\]}}, [3{{\]\]}} : tensor<1x1x2x64xf16> into tensor<2x64xf16>
// CHECK:             linalg.generic
// CHECK-SAME:          ins([[COLLAPSE0]], [[COLLAPSE1]] : tensor<2x64xf16>, tensor<2x64xf16>)

#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0    = arith.constant 0    : index
    %c1000 = arith.constant 1000 : index
    %addr_buf_base = arith.constant 2000 : index
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %desc_src = ktdp.construct_memory_view %c1000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %src_tile = ktdp.construct_access_tile %desc_src[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set2}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    // Two loads of the same source tile.
    %src_0 = ktdp.load %src_tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
    %src_1 = ktdp.load %src_tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %empty = tensor.empty() : tensor<2x32x2x64xf16>
    %result = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%src_0, %src_1 : tensor<2x32x2x64xf16>, tensor<2x32x2x64xf16>)
        outs(%empty : tensor<2x32x2x64xf16>) {
    ^bb0(%in_a: f16, %in_b: f16, %out: f16):
      %sum = arith.addf %in_a, %in_b : f16
      linalg.yield %sum : f16
    } -> tensor<2x32x2x64xf16>

    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">
    %addr_tile = ktdp.construct_access_tile %addr_buf[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set}
        : memref<2x32xindex, #ktdp.memory_space<global>>
        -> !ktdp.access_tile<2x32xindex>
    %addr_vals = ktdp.load %addr_tile
        : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>
    %iab_tile = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_vals, %iab_tile
        : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    %desc_dst = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>
    %dst_tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_dst[(%c0), (%c0 + %arg7), (%arg8)]
        {variables_space_order = #map, variables_space_set = #set2}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %result, %dst_tile
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
