#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  // ktdf_arch.device @spyre_single_corelet import("/Users/swagathvenkataramani/Documents/GitHub2/dataflow-scheduler/build/../../deeptools/dsc/KTDFArchGraphDevice/spyre_dd2_basic.mlir")
  func.func @identity_0() attributes {grid=[2]} {
    %c64000 = arith.constant 64000 : index
    %0 = ktdp.construct_memory_view %c64000, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x64x64xf16>
    %c0 = arith.constant 0 : index
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<12x64x64xf16> -> !ktdp.access_tile<12x64x64xindex>
    %c162304 = arith.constant 162304 : index
    %4 = ktdp.construct_memory_view %c162304, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x64x64xf16>
    %c0_1 = arith.constant 0 : index
    %5 = ktdp.construct_access_tile %4[%c0_1, %c0_1, %c0_1] {access_tile_order = #map1, access_tile_set = #set1} : memref<12x64x64xf16> -> !ktdp.access_tile<12x64x64xindex>
    %6 = ktdp.load %1 : <12x64x64xindex> -> tensor<12x64x64xf16>
    %8 = tensor.empty() : tensor<12x64x64xf16>
    %9 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6: tensor<12x64x64xf16>) outs(%8 : tensor<12x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<12x64x64xf16>
    ktdp.store %9, %5 : tensor<12x64x64xf16>, <12x64x64xindex>
    return
  }
}