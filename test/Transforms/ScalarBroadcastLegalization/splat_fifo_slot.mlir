// The slot a splat transfer fills is widened with the read that takes it.
//
// An operand the compute broadcasts can reach it through a stream whose slot is
// one element wide, which is what an access tile reading one element of a
// dimension makes. Marking the transfer a splat and widening the read is not
// enough on its own: the read has to match the slot it reads from, and the
// transfer has to say the width it fills.
//
// The other operand is the whole row and is left alone, so the two slots start
// different widths and end the same one.

// RUN: dataflow-scheduler-opt --scalar-broadcast-legalization %s | FileCheck %s

// CHECK: #[[PLAIN:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>

// Both slots are the compute's width now, the one-element one among them.
// CHECK:      ktdf.private -> (memref<1x1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">,
// CHECK-SAME:   !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
// CHECK:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

// One element arrives and 64 are filled, which is the splat the load unit has.
// CHECK:      ktdf.data_transfer from %[[P:.*]]#0[0, 0, 0, 0] size [1, 1, 1, 64] to %[[P]]#3 size [64] :
// CHECK-SAME:   !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT: ktdf.data_transfer from %[[P]]#1[0, 0, 0, 0] size [1, 1, 1, 1] to %[[P]]#4 size [64] {transfer_mode = "splat"} :
// CHECK-SAME:   !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

// Each read is the width of the slot it reads, and the broadcast map is gone.
// CHECK:      ktdf.read_from_fifo %[[P]]#3 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK-NEXT: ktdf.read_from_fifo %[[P]]#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK:      linalg.generic {indexing_maps = [#[[PLAIN]], #[[PLAIN]], #{{.*}}]

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
      %3:10 = ktdf.private -> (memref<1x1x1x64xf16, "L1">, memref<1x1x1x1xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %alloc = memref.alloc() : memref<1x1x1x64xf16, "L1">
        %alloc_1 = memref.alloc() : memref<1x1x1x1xf16, "L1">
        %alloc_2 = memref.alloc() : memref<1x1x1x1x64xf16, "L1">
        %4:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>
        %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %6 = ktdf.create_token : !ktdf.token
        %7 = ktdf.create_token : !ktdf.token
        %8 = ktdf.create_token : !ktdf.token
        %9 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %alloc, %alloc_1, %alloc_2, %4#0, %4#1, %5, %6, %7, %8, %9 : memref<1x1x1x64xf16, "L1">, memref<1x1x1x1xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%3#6) {
        ktdf.data_transfer from %reinterpret_cast[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] to %3#0[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
        ktdf.data_transfer from %reinterpret_cast_1[%c0, %c0, %c0, 0] size [1, 1, 1, 1] to %3#1[0, 0, 0, 0] size [1, 1, 1, 1] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x1xf16, "L1">
      } {applicable_units = ["MNILU"]}
      ktdf.stage depends_in(%3#6) depends_out(%3#7) {
        ktdf.data_transfer from %3#0[0, 0, 0, 0] size [1, 1, 1, 64] to %3#3 size [64] : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %3#1[0, 0, 0, 0] size [1, 1, 1, 1] to %3#4 size [1] : memref<1x1x1x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%3#7) depends_out(%3#8) {
        %4 = ktdf.read_from_fifo %3#3 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
        %5 = ktdf.read_from_fifo %3#4 : <"L1LU" -> "SFU", 1xf16> -> tensor<1x1x1x1xf16>
        %6 = tensor.empty() : tensor<1x1x1x1x64xf16>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%4, %5 : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>) outs(%6 : tensor<1x1x1x1x64xf16>) {
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
