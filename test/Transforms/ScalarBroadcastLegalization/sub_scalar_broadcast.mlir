// RUN: dataflow-scheduler-opt --scalar-broadcast-legalization %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module {
// CHECK-NEXT:     func.func @"Softmax_1177-Sub"() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 64000 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 113152 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 162304 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[CONSTANT_0]]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[CONSTANT_0]]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: {{\[}}%[[CONSTANT_0]]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:10 = ktdf.private -> (memref<1x1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x1x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<1x1x1x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_2:.*]] = memref.alloc() : memref<1x1x1x1x64xf16, "L1">
// CHECK-NEXT:           %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:           %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[ALLOC_2]], %[[FIFO_0]]#0, %[[FIFO_0]]#1, %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : memref<1x1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#6) {
// CHECK-NEXT:           ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 1, 1, 64] to %[[VAL_0]]#0[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
                         // Widening to [1, 1, 1, 64] is done upstream by access-granularity-legalization.
// CHECK-NEXT:           ktdf.data_transfer from %[[REINTERPRET_CAST_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]], 0] size [1, 1, 1, 64] to %[[VAL_0]]#1[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_1:.*]]#6) depends_out(%[[VAL_1]]#7) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_1]]#0[0, 0, 0, 0] size [1, 1, 1, 64] to %[[VAL_1]]#3 size [64] : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                         // Splat mode and slot widening are done upstream; source size stays [1, 1, 1, 1].
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_1]]#1[0, 0, 0, 0] size [1, 1, 1, 1] to %[[VAL_1]]#4 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:         } {applicable_units = ["L1LU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#7) depends_out(%[[VAL_2]]#8) {
// CHECK-NEXT:           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_2]]#3 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
                         // This pass updates read_from_fifo type and replaces the broadcast map.
// CHECK-NEXT:           %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_2]]#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK-NEXT:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x1x1x64xf16>
                         // The linalg.generic second input map is replaced with the identity map.
// CHECK-NEXT:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]] : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x1x1x1x64xf16>) {
// CHECK-NEXT:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:             %[[SUBF_0:.*]] = arith.subf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK-NEXT:             linalg.yield %[[SUBF_0]] : f16
// CHECK-NEXT:           } -> tensor<1x1x1x1x64xf16>
// CHECK-NEXT:           ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_2]]#5 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_6:.*]]#8) depends_out(%[[VAL_6]]#9) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_6]]#5 size [64] to %[[VAL_6]]#2[0, 0, 0, 0, 0] size [1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x1x1x64xf16, "L1">
// CHECK-NEXT:         } {applicable_units = ["L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_7:.*]]#9) depends_out(none) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_7]]#2[0, 0, 0, 0, 0] size [1, 1, 1, 1, 64] to %[[REINTERPRET_CAST_2]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 1, 1, 1, 64] : memref<1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }


ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>

module {
  func.func @"Softmax_1177-Sub"() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c64000 = arith.constant 64000 : index
    %c113152 = arith.constant 113152 : index
    %c162304 = arith.constant 162304 : index
    
    %0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %memspacecast = memref.memory_space_cast %0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
    
    %1 = ktdp.construct_memory_view %c113152, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %memspacecast_1 = memref.memory_space_cast %1 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_1 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
    
    %2 = ktdp.construct_memory_view %c162304, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
    %memspacecast_2 = memref.memory_space_cast %2 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
    %reinterpret_cast_2 = memref.reinterpret_cast %memspacecast_2 to offset: [%c0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
    
    ktdf.pipeline {
      %3:10 = ktdf.private -> (memref<1x1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %alloc = memref.alloc() : memref<1x1x1x64xf16, "L1">
        %alloc_1 = memref.alloc() : memref<1x1x1x64xf16, "L1">
        %alloc_2 = memref.alloc() : memref<1x1x1x1x64xf16, "L1">
        %4:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %6 = ktdf.create_token : !ktdf.token
        %7 = ktdf.create_token : !ktdf.token
        %8 = ktdf.create_token : !ktdf.token
        %9 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %alloc, %alloc_1, %alloc_2, %4#0, %4#1, %5, %6, %7, %8, %9 : memref<1x1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%3#6) {
        ktdf.data_transfer from %reinterpret_cast[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] to %3#0[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
        ktdf.data_transfer from %reinterpret_cast_1[%c0, %c0, %c0, 0] size [1, 1, 1, 64] to %3#1[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
      } {applicable_units = ["MNILU"]}
      ktdf.stage depends_in(%3#6) depends_out(%3#7) {
        ktdf.data_transfer from %3#0[0, 0, 0, 0] size [1, 1, 1, 64] to %3#3 size [64] : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %3#1[0, 0, 0, 0] size [1, 1, 1, 1] to %3#4 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%3#7) depends_out(%3#8) {
        %4 = ktdf.read_from_fifo %3#3 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
        %5 = ktdf.read_from_fifo %3#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
        %6 = tensor.empty() : tensor<1x1x1x1x64xf16>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%4, %5 : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16>) outs(%6 : tensor<1x1x1x1x64xf16>) {
        ^bb0(%in: f16, %in_1: f16, %out: f16):
          %8 = arith.subf %in, %in_1 : f16
          linalg.yield %8 : f16
        } -> tensor<1x1x1x1x64xf16>
        ktdf.write_to_fifo %7, %3#5 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%3#8) depends_out(%3#9) {
        ktdf.data_transfer from %3#5 size [64] to %3#2[0, 0, 0, 0, 0] size [1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x1x1x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
      ktdf.stage depends_in(%3#9) depends_out(none) {
        ktdf.data_transfer from %3#2[0, 0, 0, 0, 0] size [1, 1, 1, 1, 64] to %reinterpret_cast_2[%c0, %c0, %c0, %c0, %c0] size [1, 1, 1, 1, 64] : memref<1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
      } {applicable_units = ["MNISU"]}
    }
    return
  }
}
