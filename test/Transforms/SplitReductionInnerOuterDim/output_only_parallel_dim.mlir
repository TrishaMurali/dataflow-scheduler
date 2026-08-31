// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_onstick_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:   }
// CHECK:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 1, 64], strides: [64, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<2x1x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [1, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x1x64xf16> to memref<2x1x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 1, 64], strides: [64, 64, 1] : memref<2x1x64xf16, "DDR"> to memref<2x1x64xf16, strided<[64, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x1x64xf16, strided<[64, 64, 1]>, "DDR"> to memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<1x64xf16> to memref<1x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:           %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] size [2, 1, 64] to %[[VAL_0]]#0 size [128] : memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:         } {applicable_units = ["L1LU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_1:.*]]#2) depends_out(%[[VAL_1]]#3) {
// CHECK-NEXT:           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_1]]#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
// CHECK-NEXT:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<2x1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:           ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_0:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:           } -> tensor<1x64xf16>
// CHECK-NEXT:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[GENERIC_0]] : tensor<1x64xf16>) outs(%[[EMPTY_1]] : tensor<1x64xf16>) {
// CHECK-NEXT:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_1:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:           } -> tensor<1x64xf16>
// CHECK-NEXT:           ktdf.write_to_fifo %[[GENERIC_1]], %[[VAL_1]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_6:.*]]#3) depends_out(none) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_6]]#1 size [64] to %[[CAST_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:         } {applicable_units = ["L1SU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



// Test: parallel loop dim (d3) that appears only in the output map.
//
// The compute stage holds a linalg.generic that reduces tensor<2x1x64xf16>
// to tensor<1x64xf16> with iterator types [reduction, parallel, reduction, parallel]:
//   d0 (size 2,  reduction) — outer dim
//   d1 (size 1,  parallel)  — appears in both maps
//   d2 (size 64, reduction) — inner dim (vector_length=64 for f16)
//   d3 (size 64, parallel)  — appears ONLY in the output map
//
// input_map  = (d0,d1,d2,d3) -> (d0,d1,d2)   tensor<2x1x64xf16>
// output_map = (d0,d1,d2,d3) -> (d1,d3)       tensor<1x64xf16>
//
// After the split:
//   Generic 1 (outer reduction, reduces d0):
//     input:  (d0,d1,d2) -> (d0,d1,d2)         tensor<2x1x64xf16>
//     output: (d0,d1,d2) -> (d1,d2)             tensor<1x64xf16>
//     iterator_types = ["reduction", "parallel", "parallel"]
//
//   Generic 2 (inner reduction, reduces d2 — renamed to new d1):
//     input:  (d0,d1,d2) -> (d0,d1)             tensor<1x64xf16>
//     output: (d0,d1,d2) -> (d0,d2)             tensor<1x64xf16>
//     iterator_types = ["parallel", "reduction", "parallel"]



#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#set  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 0 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @sum_onstick_1core() attributes {grid = [1]} {
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
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 1, 64], strides: [64, 64, 1]
           {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x1x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [1, 64], strides: [64, 1]
           {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %memspacecast   = memref.memory_space_cast %0 : memref<2x1x64xf16> to memref<2x1x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast
           to offset: [0], sizes: [2, 1, 64], strides: [64, 64, 1]
           : memref<2x1x64xf16, "DDR"> to memref<2x1x64xf16, strided<[64, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast
           : memref<2x1x64xf16, strided<[64, 64, 1]>, "DDR">
          to memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0
           to offset: [0], sizes: [1, 64], strides: [64, 1]
           : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1
           : memref<1x64xf16, strided<[64, 1]>, "DDR">
          to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %priv:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>,
                                   !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                                   !ktdf.token, !ktdf.token) {
          %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
          %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %f0, %f1, %t0, %t1
              : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>,
                !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%priv#2) {
          ktdf.data_transfer from %cast[%c0, %c0, %c0] size [2, 1, 64] to %priv#0 size [128]
              : memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">,
                !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
          %in   = ktdf.read_from_fifo %priv#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
          %init = tensor.empty() : tensor<1x64xf16>
          %res  = linalg.generic {
              indexing_maps = [#map, #map1],
              iterator_types = ["reduction", "parallel", "reduction", "parallel"]
          } ins(%in : tensor<2x1x64xf16>) outs(%init : tensor<1x64xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %res, %priv#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%priv#3) depends_out(none) {
          ktdf.data_transfer from %priv#1 size [64] to %cast_2[%c0, %c0] size [1, 64]
              : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
