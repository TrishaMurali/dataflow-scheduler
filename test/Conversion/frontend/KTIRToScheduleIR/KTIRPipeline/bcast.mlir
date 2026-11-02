// RUN: dataflow-scheduler-opt --ktir-pipeline %s | FileCheck %s

// CHECK-DAG:   #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, 0)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
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
// CHECK-DAG:   %[[MEMCAST_0:.+]] = memref.memory_space_cast %[[MEMVIEW_0]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK-DAG:   %[[REINTERP_0:.+]] = memref.reinterpret_cast %[[MEMCAST_0]] to offset: [%[[C0]]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-DAG:   %[[MEMVIEW_1:.+]] = ktdp.construct_memory_view %[[C113152]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[SET]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK-DAG:   %[[MEMCAST_1:.+]] = memref.memory_space_cast %[[MEMVIEW_1]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK-DAG:   %[[REINTERP_1:.+]] = memref.reinterpret_cast %[[MEMCAST_1]] to offset: [%[[C0]]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-DAG:   %[[MEMVIEW_2:.+]] = ktdp.construct_memory_view %[[C162304]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #[[SET1]], memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
// CHECK-DAG:   %[[MEMCAST_2:.+]] = memref.memory_space_cast %[[MEMVIEW_2]] : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
// CHECK-DAG:   %[[REINTERP_2:.+]] = memref.reinterpret_cast %[[MEMCAST_2]] to offset: [%[[C0]]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:  scf.for %[[IV0:.+]] = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[IV1:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:      scf.for %[[IV2:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:        scf.for %[[IV3:.+]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:          scf.for %[[IV4:.+]] = %[[C0]] to %[[C64]] step %[[C64]] {
// CHECK-NEXT:            ktdf.pipeline {
// CHECK-NEXT:              %[[PRIVATE:.+]]:9 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.token, memref<1x1x1x64xf16, "L1">)
// CHECK:                     ktdf.private_yield
// CHECK-NEXT:              }
// CHECK-NEXT:              ktdf.stage depends_in(none) depends_out(%[[PRIVATE]]#5) {
// CHECK-NEXT:                ktdf.data_transfer from %[[REINTERP_0]][%[[IV0]], %[[IV2]], %[[IV3]], %[[IV4]]] size [1, 1, 1, 64] to %[[PRIVATE]]#6 size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:                ktdf.data_transfer from %[[REINTERP_1]][%[[IV0]], %[[IV2]], %[[IV3]], 0] size [1, 1, 1, 1] to %[[PRIVATE]]#4 size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64, transfer_mode = #ktdf.transfer_mode<splat>} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
// CHECK-NEXT:              }
// CHECK-NEXT:              ktdf.stage depends_in(%[[PRIVATE]]#5) depends_out(%[[PRIVATE]]#7) {
// CHECK-NEXT:                ktdf.data_transfer from %[[PRIVATE]]#4 size [1, 1, 1, 64] to %[[PRIVATE]]#8[0, 0, 0, 0] size [1, 1, 1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x1x1x64xf16, "L1">
// CHECK-NEXT:              }
// CHECK-NEXT:              ktdf.stage depends_in(%[[PRIVATE]]#7) depends_out(%[[PRIVATE]]#3) {
// CHECK-NEXT:                ktdf.data_transfer from %[[PRIVATE]]#8[0, 0, 0, 0] size [1, 1, 1, 64] to %[[PRIVATE]]#2 size [1, 1, 1, 64] : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
// CHECK-NEXT:              }
// CHECK-NEXT:              ktdf.stage depends_in(%[[PRIVATE]]#3, %[[PRIVATE]]#5) depends_out(%[[PRIVATE]]#1) {
// CHECK-NEXT:                %[[READ_0:.+]] = ktdf.read_from_fifo %[[PRIVATE]]#6 : <"DDR" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK-NEXT:                %[[READ_1:.+]] = ktdf.read_from_fifo %[[PRIVATE]]#2 : <"L1" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK:                     %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} 
// CHECK-SAME:                  ins(%[[READ_0]], %[[READ_1]] : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16>)
// CHECK-SAME:                  attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
// CHECK:                     ktdf.write_to_fifo %[[GENERIC]], %[[PRIVATE]]#0 : tensor<1x1x1x1x64xf16>, <"SFU" -> "DDR", 64xf16>
// CHECK-NEXT:              } {applicable_units = ["SFU"]}
// CHECK-NEXT:              ktdf.stage depends_in(%[[PRIVATE]]#1) depends_out(none) {
// CHECK-NEXT:                ktdf.data_transfer from %[[PRIVATE]]#0 size [1, 1, 1, 1, 64] to %[[REINTERP_2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]]] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:        } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:    } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:  } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:  return

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c162304 = arith.constant 162304 : index
    %c113152 = arith.constant 113152 : index
    %c0 = arith.constant 0 : index
    %c64000 = arith.constant 64000 : index
    %0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %memspacecast = memref.memory_space_cast %0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
    %1 = ktdp.construct_memory_view %c113152, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %memspacecast_0 = memref.memory_space_cast %1 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
    %2 = ktdp.construct_memory_view %c162304, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
    %memspacecast_2 = memref.memory_space_cast %2 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
    %reinterpret_cast_3 = memref.reinterpret_cast %memspacecast_2 to offset: [%c0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
    scf.for %arg0 = %c0 to %c12 step %c1 {
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c64 step %c1 {
            scf.for %arg4 = %c0 to %c64 step %c64 {
              %3 = ktdp_lowering.load %reinterpret_cast[%arg0, %arg2, %arg3, %arg4] [1, 1, 1, 64] [1, 1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR"> -> tensor<1x1x1x64xf16>
              %4 = ktdp_lowering.load %reinterpret_cast_1[%arg0, %arg2, %arg3, 0] [1, 1, 1, 1] [1, 1, 1, 0] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR"> -> tensor<1x1x1x1xf16>
              %extracted_slice = tensor.extract_slice %4[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 0] {ktdf_arch.maps_to = "DDR"} : tensor<1x1x1x1xf16> to tensor<1x1x1x64xf16>
              %5 = ktdf.via["L1"] %extracted_slice : tensor<1x1x1x64xf16>
              %6 = tensor.empty() : tensor<1x1x1x1x64xf16>
              %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16>) outs(%6 : tensor<1x1x1x1x64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %8 = arith.mulf %in, %in_4 : f16
                linalg.yield %8 : f16
              } -> tensor<1x1x1x1x64xf16>
              ktdp_lowering.store %7 into %reinterpret_cast_3[%arg0, %arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1, 64] [1, 1, 1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x1x1x1x64xf16> into memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
            } {loop_type = #ktdf.loop_type<parallel_loop>}
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
