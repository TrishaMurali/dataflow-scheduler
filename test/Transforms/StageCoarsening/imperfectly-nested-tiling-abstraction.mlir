// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK-LABEL:   func.func private @"local-schedule-0"() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 113216 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 113152 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 64000 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 12 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
// CHECK-NEXT:     %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [1, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<1x64xf16> to memref<1x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:     %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
// CHECK-NEXT:     %[[CAST_2:.*]] = memref.cast %[[REINTERPRET_CAST_2]] : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[RESERVE_SIZE_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[RESERVE_SIZE_1:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[CEILDIVUI_0:.*]] = arith.ceildivui %[[CONSTANT_6]], %[[RESERVE_SIZE_0]] : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CEILDIVUI_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[CEILDIVUI_1:.*]] = arith.ceildivui %[[CONSTANT_2]], %[[RESERVE_SIZE_1]] : index
// CHECK-NEXT:       %[[DERIVE_SIZE_0:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_0]] : %[[RESERVE_SIZE_0]]], total_size = %[[CONSTANT_6]] : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CEILDIVUI_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:         %[[DERIVE_SIZE_1:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_1]] : %[[RESERVE_SIZE_1]]], total_size = %[[CONSTANT_2]] : index
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:5 = ktdf.private -> (memref<?x?x1x1x1x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token) {
// CHECK-NEXT:             %[[ALLOC_0:.*]] = memref.alloc(%[[RESERVE_SIZE_0]], %[[RESERVE_SIZE_1]]) : memref<?x?x1x1x1x64xf16, "L1">
// CHECK-NEXT:             %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:             %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[FIFO_0]]#0, %[[FIFO_0]]#1, %[[CREATE_TOKEN_0]] : memref<?x?x1x1x1x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_2:.*]]#4) {
// CHECK-NEXT:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[DERIVE_SIZE_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               %[[LINEARIZE_INDEX_0:.*]] = ktdf.tiling.linearize_index {{\[}}%[[VAL_0]] : %[[RESERVE_SIZE_0]]], {{\[}}%[[VAL_3]] : %[[CONSTANT_1]]] : index
// CHECK-NEXT:               scf.for %[[VAL_4:.*]] = %[[CONSTANT_0]] to %[[DERIVE_SIZE_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:                 %[[LINEARIZE_INDEX_1:.*]] = ktdf.tiling.linearize_index {{\[}}%[[VAL_1]] : %[[RESERVE_SIZE_1]]], {{\[}}%[[VAL_4]] : %[[CONSTANT_1]]] : index
// CHECK-NEXT:                 %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[SUBI_1:.*]] = arith.subi %[[VAL_4]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[LINEARIZE_INDEX_0]], %[[CONSTANT_0]], %[[LINEARIZE_INDEX_1]], %[[CONSTANT_0]]] size [1, 1, 1, 64] to %[[VAL_2]]#0{{\[}}%[[DIVSI_0]], %[[DIVSI_1]], 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
// CHECK-NEXT:                 ktdf.data_transfer from %[[CAST_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 64] to %[[VAL_2]]#1[0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {applicable_units = ["MNILU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_5:.*]]#4) depends_out(none) {
// CHECK-NEXT:             scf.for %[[VAL_6:.*]] = %[[CONSTANT_0]] to %[[DERIVE_SIZE_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:               %[[LINEARIZE_INDEX_2:.*]] = ktdf.tiling.linearize_index {{\[}}%[[VAL_0]] : %[[RESERVE_SIZE_0]]], {{\[}}%[[VAL_6]] : %[[CONSTANT_1]]] : index
// CHECK-NEXT:               scf.for %[[VAL_7:.*]] = %[[CONSTANT_0]] to %[[DERIVE_SIZE_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:                 %[[LINEARIZE_INDEX_3:.*]] = ktdf.tiling.linearize_index {{\[}}%[[VAL_1]] : %[[RESERVE_SIZE_1]]], {{\[}}%[[VAL_7]] : %[[CONSTANT_1]]] : index
// CHECK-NEXT:                 %[[SUBI_2:.*]] = arith.subi %[[VAL_6]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[SUBI_3:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_5]]#0{{\[}}%[[DIVSI_2]], %[[DIVSI_3]], 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %[[VAL_5]]#2 size [64] : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_5]]#1[0, 0] size [1, 64] to %[[VAL_5]]#3 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {applicable_units = ["L1LU"]}
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }







#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func private @"local-schedule-0"() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c113216 = arith.constant 113216 : index
    %c113152 = arith.constant 113152 : index
    %c64000 = arith.constant 64000 : index
    %c12 = arith.constant 12 : index
    %0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %memspacecast = memref.memory_space_cast %0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
    %cast = memref.cast %reinterpret_cast : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
    %1 = ktdp.construct_memory_view %c113152, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
    %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64xf16> to memref<1x64xf16, "DDR">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    %2 = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
    %memspacecast_3 = memref.memory_space_cast %2 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
    %reinterpret_cast_4 = memref.reinterpret_cast %memspacecast_3 to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
    %cast_5 = memref.cast %reinterpret_cast_4 : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
    %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %4 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %5 = arith.ceildivui %c12, %3 : index
    scf.for %arg0 = %c0 to %5 step %c1 {
      %6 = arith.ceildivui %c64, %4 : index
      %7 = ktdf.tiling.derive_size [%arg0 : %3], total_size = %c12 : index
      scf.for %arg1 = %c0 to %6 step %c1 {
        %8 = ktdf.tiling.derive_size [%arg1 : %4], total_size = %c64 : index
        scf.for %arg2 = %c0 to %7 step %c1 {
          %9 = ktdf.tiling.linearize_index [%arg0 : %3], [%arg2 : %c1] : index
          scf.for %arg3 = %c0 to %8 step %c1 {
            %10 = ktdf.tiling.linearize_index [%arg1 : %4], [%arg3 : %c1] : index
            ktdf.pipeline {
              %11:10 = ktdf.private -> (memref<1x1x1x64xf16, "L1">, memref<1x64xf16, "L1">, memref<12x1x1x64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
                %alloc = memref.alloc() : memref<1x1x1x64xf16, "L1">
                %alloc_6 = memref.alloc() : memref<1x64xf16, "L1">
                %alloc_7 = memref.alloc() : memref<12x1x1x64x64xf16, "L1">
                %12:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %13 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %14 = ktdf.create_token : !ktdf.token
                %15 = ktdf.create_token : !ktdf.token
                %16 = ktdf.create_token : !ktdf.token
                %17 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %alloc, %alloc_6, %alloc_7, %12#0, %12#1, %13, %14, %15, %16, %17 : memref<1x1x1x64xf16, "L1">, memref<1x64xf16, "L1">, memref<12x1x1x64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%11#6) {
                ktdf.data_transfer from %cast[%9, %c0, %10, %c0] size [1, 1, 1, 64] to %11#0[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
                ktdf.data_transfer from %cast_2[%c0, %c0] size [1, 64] to %11#1[0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
              } {applicable_units = ["MNILU"]}
              ktdf.stage depends_in(%11#6) depends_out(%11#7) {
                ktdf.data_transfer from %11#0[0, 0, 0, 0] size [1, 1, 1, 64] to %11#3 size [64] : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                ktdf.data_transfer from %11#1[0, 0] size [1, 64] to %11#4 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

