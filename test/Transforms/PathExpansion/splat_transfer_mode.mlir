// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// A splat load from DDR into a FIFO that needs a hop through L1.
//
// Stage 0 loads two operands from DDR for the compute stage on SFU:
//   - %a: a plain 64-element load (size [1, 64] -> FIFO of 64).
//   - %b: a splat, one element replicated to the FIFO's 64 (size [1, 1] -> FIFO
//     of 64), marked transfer_mode = splat.
//
// The device has no DDR -> SFU edge, so each load goes DDR -> L1 -> L1LU -> SFU
// and path expansion adds a hop stage on L1LU. The splat is split in two:
//   - DDR -> L1 buffer: moves the one element into memory. Nothing widens, so
//     it is a plain copy (1 -> 1, no transfer_mode).
//   - L1 buffer -> FIFO: the first transfer that writes the FIFO, so it is the
//     one that widens 1 -> 64, and it carries the splat.
// The plain load must stay a copy on both hops.

// CHECK-LABEL: func.func @splat_hops(

// The L1 buffers and FIFOs path expansion creates, referred to below as
// PRIV#N in the order they are declared: a 1x64 buffer for %a, a 1x1 buffer
// for the splat source, then the two L1LU -> SFU FIFOs (one for %a, one for
// the splat).
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        %[[PRIV:.*]]:{{[0-9]+}} = ktdf.private -> (memref<1x64xf16, "L1">, memref<1x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,

// Stage 0, the load stage: both loads end in an L1 buffer, so both copy.
//   PRIV#0 = %a buffer (1x64), PRIV#1 = splat source buffer (1x1).
// CHECK:             ktdf.stage depends_in(none) depends_out(%[[PRIV]]#{{[0-9]+}}) {
// CHECK-NEXT:          ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}] size [1, 64] to %[[PRIV]]#0[0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:          ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}] size [1, 1] to %[[PRIV]]#1[0, 0] size [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x1xf16, "L1">
// CHECK-NEXT:        }

// Stage 1, the new hop stage on L1LU: the L1 buffers feed the FIFOs.
//   PRIV#2 = FIFO for %a, PRIV#3 = FIFO for the splat.
// %a copies 64 -> 64. The splat widens 1 -> 64 and is the only transfer here
// that carries transfer_mode.
// CHECK:             ktdf.stage depends_in(%[[PRIV]]#{{[0-9]+}}) depends_out(%[[PRIV]]#{{[0-9]+}}) {
// CHECK-NEXT:          ktdf.data_transfer from %[[PRIV]]#0[0, 0] size [1, 64] to %[[PRIV]]#2 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:          ktdf.data_transfer from %[[PRIV]]#1[0, 0] size [1, 1] to %[[PRIV]]#3 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:        }

// Stage 2, the compute stage, reads the two FIFOs unchanged.
// CHECK:             ktdf.stage
// CHECK-NEXT:          ktdf.read_from_fifo %[[PRIV]]#2 :
// CHECK-NEXT:          ktdf.read_from_fifo %[[PRIV]]#3 :

// No other transfer in the function may carry a transfer mode.
// CHECK-NOT:         transfer_mode

#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @splat_hops(%arg0: index) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    %c18432 = arith.constant 18432 : index
    %c0 = arith.constant 0 : index
    %3 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
    %mem_cast = memref.memory_space_cast %3 : memref<96x64xf16> to memref<96x64xf16, "DDR">
    %reinterpret_cast = memref.reinterpret_cast %mem_cast to offset: [%c0], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    %5 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
    %mem_cast_7 = memref.memory_space_cast %5 : memref<96x64xf16> to memref<96x64xf16, "DDR">
    %reinterpret_cast_7 = memref.reinterpret_cast %mem_cast_7 to offset: [%c0], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    %7 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
    %mem_cast_9 = memref.memory_space_cast %7 : memref<96x64xf16> to memref<96x64xf16, "DDR">
    %reinterpret_cast_9 = memref.reinterpret_cast %mem_cast_9 to offset: [%c0], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf.pipeline {
          %9:6 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
            %10:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
            %11 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
            %12 = ktdf.create_token : !ktdf.token
            %13 = ktdf.create_token : !ktdf.token
            %14 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %10#0, %10#1, %11, %12, %13, %14 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%9#3) {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %9#0 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
            ktdf.data_transfer from %reinterpret_cast_7[%arg1, %arg2] size [1, 1] to %9#1 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
          }
          ktdf.stage depends_in(%9#3) depends_out(%9#4) {
            %10 = ktdf.read_from_fifo %9#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
            %11 = ktdf.read_from_fifo %9#1 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
            %12 = tensor.empty() : tensor<64xf16>
            %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %11 : tensor<64xf16>, tensor<64xf16>) outs(%12 : tensor<64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %14 = arith.addf %in, %in_10 : f16
              linalg.yield %14 : f16
            } -> tensor<64xf16>
            ktdf.write_to_fifo %13, %9#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%9#4) depends_out(%9#5) {
            ktdf.data_transfer from %9#2 size [64] to %reinterpret_cast_9[%arg1, %arg2] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
