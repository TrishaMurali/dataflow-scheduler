// RUN: dataflow-scheduler-opt --ktir-pipeline %s | FileCheck %s

// A compute with two results, each stored to its own output. The first store
// pulls the compute in while its second result still has a user outside the
// pipeline, so that placement fails. The second store must place it again.

// CHECK-LABEL: func.func @local_schedule_1()
// CHECK-DAG:     %[[OUT_0:.+]] = memref.reinterpret_cast %memspacecast_0
// CHECK-DAG:     %[[OUT_1:.+]] = memref.reinterpret_cast %memspacecast_2
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK:             ktdf.data_transfer from %{{.+}} to %{{.+}} size [2, 1, 64]
// CHECK:           ktdf.stage
// CHECK:             %[[RESULT:.+]]:2 = linalg.generic
// The writes are in the order of the compute's results. The IR does not say so:
// each transfer out of the FIFO is paired with a write by position once lowered,
// and slots are used in the order they were allocated, so result #0 has to be
// written first.
// CHECK:             ktdf.write_to_fifo %[[RESULT]]#0, %[[SLOT_0:[^ ]+]] :
// CHECK-NEXT:        ktdf.write_to_fifo %[[RESULT]]#1, %[[SLOT_1:[^ ]+]] :
// CHECK:           } {applicable_units = ["SFU"]}
// CHECK:           ktdf.stage
// The transfers out are in the same order, one per write.
// CHECK:             ktdf.data_transfer from %[[SLOT_0]] size [1, 64] to %[[OUT_0]][
// CHECK:             ktdf.data_transfer from %[[SLOT_1]] size [1, 64] to %[[OUT_1]][
// CHECK-NOT:     ktdp_lowering.store

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %c12288 = arith.constant 12288 : index
    %0 = ktdp.construct_memory_view %c4096, sizes: [2, 8, 64], strides: [512, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x8x64xf16>
    %memspacecast = memref.memory_space_cast %0 : memref<2x8x64xf16> to memref<2x8x64xf16, "DDR">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%c0], sizes: [2, 8, 64], strides: [512, 64, 1] : memref<2x8x64xf16, "DDR"> to memref<2x8x64xf16, strided<[512, 64, 1], offset: ?>, "DDR">
    %1 = ktdp.construct_memory_view %c8192, sizes: [8, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<8x64xf16>
    %memspacecast_0 = memref.memory_space_cast %1 : memref<8x64xf16> to memref<8x64xf16, "DDR">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [%c0], sizes: [8, 64], strides: [64, 1] : memref<8x64xf16, "DDR"> to memref<8x64xf16, strided<[64, 1], offset: ?>, "DDR">
    %2 = ktdp.construct_memory_view %c12288, sizes: [8, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<8x64xf16>
    %memspacecast_2 = memref.memory_space_cast %2 : memref<8x64xf16> to memref<8x64xf16, "DDR">
    %reinterpret_cast_3 = memref.reinterpret_cast %memspacecast_2 to offset: [%c0], sizes: [8, 64], strides: [64, 1] : memref<8x64xf16, "DDR"> to memref<8x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c64 {
        %3 = ktdp_lowering.load %reinterpret_cast[0, %arg0, %arg1] [2, 1, 64] [1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<2x8x64xf16, strided<[512, 64, 1], offset: ?>, "DDR"> -> tensor<2x1x64xf16>
        %4 = tensor.empty() : tensor<1x64xf16>
        %5 = tensor.empty() : tensor<1x64xf16>
        %6:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%3 : tensor<2x1x64xf16>) outs(%4, %5 : tensor<1x64xf16>, tensor<1x64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
        ^bb0(%in: f16, %out: f16, %out_4: f16):
          %7 = arith.addf %in, %out : f16
          %8 = arith.mulf %in, %in : f16
          %9 = arith.addf %8, %out_4 : f16
          linalg.yield %7, %9 : f16, f16
        } -> (tensor<1x64xf16>, tensor<1x64xf16>)
        ktdp_lowering.store %6#0 into %reinterpret_cast_1[%arg0, %arg1] [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x64xf16> into memref<8x64xf16, strided<[64, 1], offset: ?>, "DDR">
        ktdp_lowering.store %6#1 into %reinterpret_cast_3[%arg0, %arg1] [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x64xf16> into memref<8x64xf16, strided<[64, 1], offset: ?>, "DDR">
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

