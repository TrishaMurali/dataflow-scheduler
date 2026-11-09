// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// Both inputs and the output of an add live in local memory, so every flow
// (L1 -> L1LU -> SFU and SFU -> L1SU -> L1) is already legal and no hop stage
// is needed. Path expansion will still rewrite the pipeline to generate stages
// with correct units attribute. Also, we rewrite the fifos to the units that
// run each end, i.e. from L1 to L1LU or L1SU

// CHECK-LABEL: func.func @all_local_no_hops(
// CHECK-NOT:     "L1" -> "SFU"
// CHECK-NOT:     "SFU" -> "L1",
// CHECK:         ktdf.private
// CHECK:           ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:           ktdf.private_yield
// CHECK:         ktdf.stage
// CHECK:           ktdf.data_transfer from %{{.*}} to %[[IN0:[0-9]+#[0-9]+]] size [1, 64] {{.*}}: memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.data_transfer from %{{.*}} to %[[IN1:[0-9]+#[0-9]+]] size [64] {{.*}}: memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:         } {applicable_units = ["L1LU"]}
// CHECK:         ktdf.stage
// CHECK:           ktdf.read_from_fifo %[[IN0]] : <"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.read_from_fifo %[[IN1]] : <"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.write_to_fifo %{{.*}}, %[[OUT:[0-9]+#[0-9]+]] : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK:         } {applicable_units = ["SFU"]}
// CHECK:         ktdf.stage
// CHECK:           ktdf.data_transfer from %[[OUT]] size [1, 64] to %{{.*}} {{.*}}: !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
// CHECK:         } {applicable_units = ["L1SU"]}
// CHECK-NOT:     "L1" -> "SFU"
// CHECK-NOT:     "SFU" -> "L1",

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @all_local_no_hops(%arg0: index) {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    %c12352 = arith.constant 12352 : index
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.muli %0, %c3 : index
    %2 = arith.addi %1, %arg0 : index
    %3 = arith.muli %2, %c64 : index
    %4 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
    %memspacecast = memref.memory_space_cast %4 : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%3], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
    %5 = ktdp.construct_memory_view %c12288, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<ct_local>} : memref<64xf16, #ktdp.memory_space<ct_local>>
    %memspacecast_0 = memref.memory_space_cast %5 : memref<64xf16, #ktdp.memory_space<ct_local>> to memref<64xf16, "L1">
    %6 = ktdp.construct_memory_view %c12352, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
    %memspacecast_1 = memref.memory_space_cast %6 : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
    %reinterpret_cast_2 = memref.reinterpret_cast %memspacecast_1 to offset: [%3], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
    %7 = tensor.empty() : tensor<1x64xf16>
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf.pipeline {
          %8:5 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.token) {
            %9 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
            %10:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
            %11 = ktdf.create_token : !ktdf.token
            %12 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %9, %10#0, %10#1, %11, %12 : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%8#3) {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %8#1 size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
            ktdf.data_transfer from %memspacecast_0[%arg2] size [64] to %8#2 size [64] {dataflow_scheduler.throttle = 64 : i64} : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
          }
          ktdf.stage depends_in(%8#3) depends_out(%8#4) {
            %9 = ktdf.read_from_fifo %8#1 : <"L1" -> "SFU", 64xf16> -> tensor<1x64xf16>
            %10 = ktdf.read_from_fifo %8#2 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
            %11 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : tensor<1x64xf16>, tensor<64xf16>) outs(%7 : tensor<1x64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
            ^bb0(%in: f16, %in_3: f16, %out: f16):
              %12 = arith.addf %in, %in_3 : f16
              linalg.yield %12 : f16
            } -> tensor<1x64xf16>
            ktdf.write_to_fifo %11, %8#0 : tensor<1x64xf16>, <"SFU" -> "L1", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%8#4) depends_out(none) {
            ktdf.data_transfer from %8#0 size [1, 64] to %reinterpret_cast_2[%arg1, %arg2] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
