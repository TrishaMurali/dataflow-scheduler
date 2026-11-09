// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @l1_boundary_test(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_10:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_11:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[CONSTANT_12:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_12]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[CONSTANT_13:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_2:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_13]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[MULI_2]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:     %[[CONSTANT_14:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_3:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_14]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
// CHECK-NEXT:     %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: {{\[}}%[[MULI_3]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_9]] to %[[CONSTANT_10]] step %[[CONSTANT_11]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[FIFO_0]]#0, %[[FIFO_0]]#1, %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_2:.*]]#3) {
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] to %[[VAL_2]]#0 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] to %[[VAL_2]]#1 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["L1LU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_3:.*]]#3) depends_out(%[[VAL_3]]#4) {
// CHECK-NEXT:             %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_3]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:             %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_3]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:             %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]] : tensor<64xf16>, tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:             ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK-NEXT:               %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK-NEXT:               linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:             } -> tensor<64xf16>
// CHECK-NEXT:             ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_3]]#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["SFU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_7:.*]]#4) depends_out(none) {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_7]]#2 size [64] to %[[REINTERPRET_CAST_2]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
// CHECK-NEXT:           } {applicable_units = ["L1SU"]}
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }


#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @l1_boundary_test(%arg0: index) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    %c18432 = arith.constant 18432 : index
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.muli %0, %c3 : index
    %2 = arith.addi %1, %arg0 : index
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %c64_3 = arith.constant 64 : index
    %c64_4 = arith.constant 64 : index
    %3 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
    %c64_5 = arith.constant 64 : index
    %4 = arith.muli %2, %c64_5 : index
    %mem_cast = memref.memory_space_cast %3 : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
    %reinterpret_cast = memref.reinterpret_cast %mem_cast to offset: [%4], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
    %5 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
    %c64_6 = arith.constant 64 : index
    %6 = arith.muli %2, %c64_6 : index
    %mem_cast_2 = memref.memory_space_cast %5 : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
    %reinterpret_cast_7 = memref.reinterpret_cast %mem_cast_2 to offset: [%6], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
    %7 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<96x64xf16, #ktdp.memory_space<ct_local>>
    %c64_8 = arith.constant 64 : index
    %8 = arith.muli %2, %c64_8 : index
    %mem_cast_3 = memref.memory_space_cast %7 : memref<96x64xf16, #ktdp.memory_space<ct_local>> to memref<96x64xf16, "L1">
    %reinterpret_cast_9 = memref.reinterpret_cast %mem_cast_3 to offset: [%8], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "L1"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
    scf.for %arg1 = %c0 to %c1_0 step %c1_1 {
      scf.for %arg2 = %c0_2 to %c64_3 step %c64_4 {
        ktdf.pipeline {
          %9:6 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
            %10:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
            %11 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
            %12 = ktdf.create_token : !ktdf.token
            %13 = ktdf.create_token : !ktdf.token
            %14 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %10#0, %10#1, %11, %12, %13, %14 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%9#3) {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %9#0 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
            ktdf.data_transfer from %reinterpret_cast_7[%arg1, %arg2] size [1, 64] to %9#1 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          }
          ktdf.stage depends_in(%9#3) depends_out(%9#4) {
            %10 = ktdf.read_from_fifo %9#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
            %11 = ktdf.read_from_fifo %9#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
            %12 = tensor.empty() : tensor<64xf16>
            %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %11 : tensor<64xf16>, tensor<64xf16>) outs(%12 : tensor<64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %14 = arith.addf %in, %in_10 : f16
              linalg.yield %14 : f16
            } -> tensor<64xf16>
            ktdf.write_to_fifo %13, %9#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%9#4) depends_out(%9#5) {
            ktdf.data_transfer from %9#2 size [64] to %reinterpret_cast_9[%arg1, %arg2] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "L1">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
