// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s | FileCheck %s

// An absmax: the running value reaches the accumulation through a math.absf
// rather than being an operand of it. So the second generic cannot be given the
// accumulating op over the argument directly -- the abs has to be rebuilt over
// each side, which is what this pins.

// CHECK-LABEL: module @local_schedule_0 {

// CHECK:       %[[G1:.*]] = linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<2x64xf16>)

// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[G1]] : tensor<2x64xf16>)
// CHECK-NEXT:  ^bb0(%[[P:.*]]: f16, %[[R:.*]]: f16):
// CHECK-NEXT:    %[[AP:.*]] = math.absf %[[P]] : f16
// CHECK-NEXT:    %[[AR:.*]] = math.absf %[[R]] : f16
// CHECK-NEXT:    %[[M:.*]] = arith.maxnumf %[[AP]], %[[AR]] : f16
// CHECK-NEXT:    linalg.yield %[[M]] : f16

#map  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#set  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>

module {
  module {
    func.func @absmax_1core() attributes {grid = [1]} {
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
        %priv:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.token, !ktdf.token) {
          %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
          %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%priv#2) {
          ktdf.data_transfer from %cast[%c0, %c0, %c0] size [2, 256, 64] to %priv#0 size [32768] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
          %in    = ktdf.read_from_fifo %priv#0 : <"L1LU" -> "SFU", 32768xf16> -> tensor<2x256x64xf16>
          %init  = tensor.empty() : tensor<2xf16>
          %res = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%in : tensor<2x256x64xf16>) outs(%init : tensor<2xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %aa = math.absf %a : f16
            %ab = math.absf %acc : f16
            %m  = arith.maxnumf %aa, %ab : f16
            linalg.yield %m : f16
          } -> tensor<2xf16>
          ktdf.write_to_fifo %res, %priv#1 : tensor<2xf16>, <"SFU" -> "L1SU", 2xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%priv#3) depends_out(none) {
          ktdf.data_transfer from %priv#1 size [2] to %cast_2[%c0] size [2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, strided<[1], offset: ?>, "DDR">
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
