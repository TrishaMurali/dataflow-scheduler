// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:   }
// CHECK:   ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [64], strides: [1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<64xf16> to memref<64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<1x2x256x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2x256x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<1x2x256x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] size [2, 256, 64] to %[[VAL_0]]#0{{\[}}%[[VAL_1]], 0, 0, 0] size [1, 2, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<1x2x256x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = arith.constant 4 : index
// CHECK-NEXT:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_4]] {
// CHECK-NEXT:               scf.for %[[VAL_5:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_6]] step %[[CONSTANT_4]] {
// CHECK-NEXT:                 %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_4]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                 %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_5]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                 %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK-NEXT:                 %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:                 %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:                 %[[CONSTANT_9:.*]] = arith.constant 63 : index
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[PRIVATE_1:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     %[[FIFO_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[FIFO_2]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_6:.*]]#3) {
// CHECK-NEXT:                     %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                     %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                     scf.if %[[ANDI_0]] {
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                       ktdf.data_transfer from %[[VAL_2]]#1{{\[}}%[[DIVSI_0]], %[[CONSTANT_3]]] size [1, 64] to %[[VAL_6]]#1 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     %[[CONSTANT_10:.*]] = arith.constant 64 : index
// CHECK-NEXT:                     scf.for %[[VAL_7:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_10]] step %[[CONSTANT_8]] {
// CHECK-NEXT:                       ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_0]], %[[VAL_4]], %[[VAL_5]] * 64 + %[[VAL_7]], %[[CONSTANT_3]]] size [1, 1, 1, 64] to %[[VAL_6]]#0 size [64] : memref<1x2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                   } {applicable_units = ["L1LU"]}
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_8:.*]]#3) depends_out(%[[VAL_8]]#4) {
// CHECK-NEXT:                     %[[IF_0:.*]] = scf.if %[[ANDI_0]] -> (tensor<64xf16>) {
// CHECK-NEXT:                       %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:                       scf.yield %[[EMPTY_0]] : tensor<64xf16>
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                       %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_8]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:                       scf.yield %[[READ_FROM_FIFO_0]] : tensor<64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     %[[CONSTANT_11:.*]] = arith.constant 64 : index
// CHECK-NEXT:                     %[[FOR_0:.*]] = scf.for %[[VAL_9:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_11]] step %[[CONSTANT_8]] iter_args(%[[VAL_10:.*]] = %[[IF_0]]) -> (tensor<64xf16>) {
// CHECK-NEXT:                       %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_8]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
// CHECK-NEXT:                       %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_1]] : tensor<1x1x64xf16>) outs(%[[VAL_10]] : tensor<64xf16>) {
// CHECK-NEXT:                       ^bb0(%[[VAL_11:.*]]: f16, %[[VAL_12:.*]]: f16):
// CHECK-NEXT:                         %[[ADDF_0:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f16
// CHECK-NEXT:                         linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                       } -> tensor<64xf16>
// CHECK-NEXT:                       %[[CMPI_2:.*]] = arith.cmpi eq, %[[VAL_9]], %[[CONSTANT_9]] : index
// CHECK-NEXT:                       scf.if %[[CMPI_2]] {
// CHECK-NEXT:                         ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_8]]#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                       }
// CHECK-NEXT:                       scf.yield %[[GENERIC_0]] : tensor<64xf16>
// CHECK-NEXT:                     } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                   } {applicable_units = ["SFU"]}
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_13:.*]]#4) depends_out(none) {
// CHECK-NEXT:                     %[[CONSTANT_12:.*]] = arith.constant 64 : index
// CHECK-NEXT:                     scf.for %[[VAL_14:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_12]] step %[[CONSTANT_8]] {
// CHECK-NEXT:                       %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_3]] : index
// CHECK-NEXT:                       %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                       %[[CMPI_3:.*]] = arith.cmpi eq, %[[VAL_14]], %[[CONSTANT_9]] : index
// CHECK-NEXT:                       scf.if %[[CMPI_3]] {
// CHECK-NEXT:                         ktdf.data_transfer from %[[VAL_13]]#2 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_1]], %[[CONSTANT_3]]] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:                       }
// CHECK-NEXT:                     } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                   } {applicable_units = ["L1SU"]}
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_15:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_16:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_15]]#1{{\[}}%[[VAL_16]], 0] size [1, 64] to %[[CAST_1]]{{\[}}%[[CONSTANT_0]]] size [64] : memref<1x64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<64xf16> to memref<64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<1x2x256x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<1x2x256x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<1x2x256x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %cast[%c0, %c0, %c0] size [2, 256, 64] to %2#0[%arg0, 0, 0, 0] size [1, 2, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<1x2x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c4 = arith.constant 4 : index
            scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
              scf.for %arg2 = %c0_3 to %c4 step %c1_4 {
                %3 = arith.cmpi eq, %arg1, %c0_3 : index
                %4 = arith.cmpi eq, %arg2, %c0_3 : index
                %5 = arith.andi %3, %4 : i1
                ktdf.pipeline {
                  %6:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                    %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
                    %8 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    %9 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                    %10 = ktdf.create_token : !ktdf.token
                    %11 = ktdf.create_token : !ktdf.token
                    ktdf.private_yield %7, %8, %9, %10, %11 : !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                  }
                  ktdf.stage depends_in(none) depends_out(%6#3) {
                    %7 = arith.subi %arg0, %c0_3 : index
                    %8 = arith.divsi %7, %c1_4 : index
                    ktdf.data_transfer from %2#0[%8, %arg1, %arg2 * 64, %c0_3] size [1, 1, 64, 64] to %6#0 size [4096] : memref<1x2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
                    scf.if %5 {
                    } else {
                      ktdf.data_transfer from %2#1[%8, %c0_3] size [1, 64] to %6#1 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    }
                  } {applicable_units = ["L1LU"]}
                  ktdf.stage depends_in(%6#3) depends_out(%6#4) {
                    %7 = ktdf.read_from_fifo %6#0 : <"L1LU" -> "SFU", 4096xf16> -> tensor<1x64x64xf16>
                    %8 = scf.if %5 -> (tensor<64xf16>) {
                      %10 = tensor.empty() : tensor<64xf16>
                      scf.yield %10 : tensor<64xf16>
                    } else {
                      %10 = ktdf.read_from_fifo %6#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
                      scf.yield %10 : tensor<64xf16>
                    }
                    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "parallel"]} ins(%7 : tensor<1x64x64xf16>) outs(%8 : tensor<64xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %10 = arith.addf %in, %out : f16
                      linalg.yield %10 : f16
                    } -> tensor<64xf16>
                    ktdf.write_to_fifo %9, %6#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
                  } {applicable_units = ["SFU"]}
                  ktdf.stage depends_in(%6#4) depends_out(none) {
                    %7 = arith.subi %arg0, %c0_3 : index
                    %8 = arith.divsi %7, %c1_4 : index
                    ktdf.data_transfer from %6#2 size [64] to %2#1[%8, %c0_3] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
                  } {applicable_units = ["L1SU"]}
                }
              }
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0] size [1, 64] to %cast_2[%c0] size [64] : memref<1x64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}

