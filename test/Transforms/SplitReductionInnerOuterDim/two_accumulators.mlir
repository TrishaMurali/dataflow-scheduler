// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s | FileCheck %s

// A compute that accumulates two things over one input: a sum, and a sum of the
// squares. Both halves split like a lone sum -- an intermediate, an output
// initialiser and a result each -- and the two generics carry both.
//
// What the second generic gets is the interesting part. It is not a copy of the
// original body: that body works out each element's contribution as well as
// accumulating it, and the first generic has done that. Copied, the square would
// be taken again, of the partial sums.

// CHECK-LABEL: module @local_schedule_0 {

// The first generic reduces the outer dimension, and holds the whole body.
// CHECK:       %[[G1:.*]]:2 = linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<2x256x64xf16>)
// CHECK-SAME:      outs(%{{.*}}, %{{.*}} : tensor<2x64xf16>, tensor<2x64xf16>)
// CHECK:         %[[SUM:.*]] = arith.addf
// CHECK:         %[[SQ:.*]] = arith.mulf
// CHECK:         arith.addf %[[SQ]],
// CHECK:         linalg.yield

// The second reduces the inner dimension over both partial results, and
// accumulates each -- no multiply here.
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[G1]]#0, %[[G1]]#1 : tensor<2x64xf16>, tensor<2x64xf16>)
// CHECK-SAME:      outs(%{{.*}}, %{{.*}} : tensor<2xf16>, tensor<2xf16>)
// CHECK-NEXT:  ^bb0(%[[P0:.*]]: f16, %[[P1:.*]]: f16, %[[R0:.*]]: f16, %[[R1:.*]]: f16):
// CHECK-NEXT:    %[[C0:.*]] = arith.addf %[[P0]], %[[R0]] : f16
// CHECK-NEXT:    %[[C1:.*]] = arith.addf %[[P1]], %[[R1]] : f16
// CHECK-NEXT:    linalg.yield %[[C0]], %[[C1]] : f16, f16
// CHECK-NOT:     arith.mulf

#map  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#set  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>

module {
  module {
    func.func @sum_and_squares_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0          = arith.constant 0 : index
      %c1          = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2xf16>
      %memspacecast   = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast           = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2xf16> to memref<2xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2], strides: [1] : memref<2xf16, "DDR"> to memref<2xf16, strided<[1]>, "DDR">
      %cast_2         = memref.cast %reinterpret_cast_1 : memref<2xf16, strided<[1]>, "DDR"> to memref<2xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %priv:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.token, !ktdf.token) {
          %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
          %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>
          %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %f0, %f1, %f2, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%priv#3) {
          ktdf.data_transfer from %cast[%c0, %c0, %c0] size [2, 256, 64] to %priv#0 size [32768] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%priv#3) depends_out(%priv#4) {
          %in    = ktdf.read_from_fifo %priv#0 : <"L1LU" -> "SFU", 32768xf16> -> tensor<2x256x64xf16>
          %init  = tensor.empty() : tensor<2xf16>
          %init2 = tensor.empty() : tensor<2xf16>
          %res:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%in : tensor<2x256x64xf16>) outs(%init, %init2 : tensor<2xf16>, tensor<2xf16>) {
          ^bb0(%a: f16, %acc: f16, %acc2: f16):
            %s  = arith.addf %a, %acc : f16
            %sq = arith.mulf %a, %a : f16
            %q  = arith.addf %sq, %acc2 : f16
            linalg.yield %s, %q : f16, f16
          } -> (tensor<2xf16>, tensor<2xf16>)
          ktdf.write_to_fifo %res#0, %priv#1 : tensor<2xf16>, <"SFU" -> "L1SU", 2xf16>
          ktdf.write_to_fifo %res#1, %priv#2 : tensor<2xf16>, <"SFU" -> "L1SU", 2xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%priv#4) depends_out(none) {
          ktdf.data_transfer from %priv#1 size [2] to %cast_2[%c0] size [2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, strided<[1], offset: ?>, "DDR">
          ktdf.data_transfer from %priv#2 size [2] to %cast_2[%c0] size [2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, strided<[1], offset: ?>, "DDR">
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
