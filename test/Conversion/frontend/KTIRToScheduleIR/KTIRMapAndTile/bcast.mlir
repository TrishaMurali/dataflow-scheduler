// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// CHECK-DAG:   #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
// CHECK-DAG:   #[[SET:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-DAG:   #[[SET1:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>

// CHECK-LABEL: func.func @local_schedule_1()
// CHECK-DAG:   %[[C12:.+]] = arith.constant 12 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C162304:.+]] = arith.constant 162304 : index
// CHECK-DAG:   %[[C113152:.+]] = arith.constant 113152 : index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C64000:.+]] = arith.constant 64000 : index
// CHECK-DAG:   %[[MEMVIEW_0:.+]] = ktdp.construct_memory_view %[[C64000]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[SET]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK-DAG:   %[[ACCTILE_0:.+]] = ktdp.construct_access_tile %[[MEMVIEW_0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {access_tile_order = #[[MAP]], access_tile_set = #[[SET]]} : memref<12x1x64x64xf16> -> !ktdp.access_tile<12x1x64x64xindex>
// CHECK-DAG:   %[[MEMVIEW_1:.+]] = ktdp.construct_memory_view %[[C113152]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[SET]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK-DAG:   %[[ACCTILE_1:.+]] = ktdp.construct_access_tile %[[MEMVIEW_1]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {access_tile_order = #[[MAP]], access_tile_set = #[[SET]]} : memref<12x1x64x64xf16> -> !ktdp.access_tile<12x1x64x64xindex>
// CHECK-DAG:   %[[MEMVIEW_2:.+]] = ktdp.construct_memory_view %[[C162304]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #[[SET1]], memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
// CHECK-DAG:   %[[ACCTILE_2:.+]] = ktdp.construct_access_tile %[[MEMVIEW_2]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]] {access_tile_order = #[[MAP1]], access_tile_set = #[[SET1]]} : memref<12x1x1x64x64xf16> -> !ktdp.access_tile<12x1x1x64x64xindex>
// CHECK:       scf.for %[[IV0:.+]] = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[IV1:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:      scf.for %[[IV2:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:        scf.for %[[IV3:.+]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:          scf.for %[[IV4:.+]] = %[[C0]] to %[[C64]] step %[[C64]] {
// CHECK-DAG:             %[[LOAD_0:.+]] = ktdp_lowering.load %[[ACCTILE_0]][%[[IV0]], %[[IV2]], %[[IV3]], %[[IV4]]] [1, 1, 1, 64] [1, 1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : !ktdp.access_tile<12x1x64x64xindex> -> tensor<1x1x1x64xf16>
// CHECK-DAG:             %[[LOAD_1:.+]] = ktdp_lowering.load %[[ACCTILE_1]][%[[IV0]], %[[IV2]], %[[IV3]], 0] [1, 1, 1, 64] [1, 1, 1, 0] {dataflow_scheduler.throttle = 64 : i64} : !ktdp.access_tile<12x1x64x64xindex> -> tensor<1x1x1x64xf16>
// CHECK-DAG:             %[[VIA:.+]] = ktdf.via["L1"] %[[LOAD_1]] : tensor<1x1x1x64xf16>
// CHECK:                 %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:              ins(%[[LOAD_0]], %[[VIA]] : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16>)
// CHECK-SAME:              attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"}
// CHECK:                 ktdp_lowering.store %[[GENERIC]] into %[[ACCTILE_2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]]] [1, 1, 1, 1, 64] [1, 1, 1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x1x1x1x64xf16> into !ktdp.access_tile<12x1x1x64x64xindex>
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:        } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:    } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:  } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:  return

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, 0)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c162304 = arith.constant 162304 : index
    %c113152 = arith.constant 113152 : index
    %c0 = arith.constant 0 : index
    %c64000 = arith.constant 64000 : index
    %0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<12x1x64x64xf16> -> !ktdp.access_tile<12x1x64x64xindex>
    %2 = ktdp.construct_memory_view %c113152, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %3 = ktdp.construct_access_tile %2[%c0, %c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<12x1x64x64xf16> -> !ktdp.access_tile<12x1x64x64xindex>
    %4 = ktdp.construct_memory_view %c162304, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
    %5 = ktdp.construct_access_tile %4[%c0, %c0, %c0, %c0, %c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<12x1x1x64x64xf16> -> !ktdp.access_tile<12x1x1x64x64xindex>
    %6 = ktdp.load %1 : <12x1x64x64xindex> -> tensor<12x1x64x64xf16>
    %7 = ktdp.load %3 : <12x1x64x64xindex> -> tensor<12x1x64x64xf16>
    %hop_7 = ktdf.via["L1"] %7 : tensor<12x1x64x64xf16>
    %8 = tensor.empty() : tensor<12x1x1x64x64xf16>
    %9 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} 
      ins(%6, %hop_7 : tensor<12x1x64x64xf16>, tensor<12x1x64x64xf16>) 
      outs(%8 : tensor<12x1x1x64x64xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %10 = arith.mulf %in, %in_0 : f16
      linalg.yield %10 : f16
    } -> tensor<12x1x1x64x64xf16>
    ktdp.store %9, %5 : tensor<12x1x1x64x64xf16>, <12x1x1x64x64xindex>
    return
  }
}
