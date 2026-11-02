// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// Data flows that need two hop stages next to each other, and a flow carried
// by a memref instead of a FIFO.
//
// On this device L1LU can stream straight into L2SU. Stage 0 loads %a from
// DDR into the L2 memref that stage 1 reads, so that flow goes
// DDR -> DDRLU -> L1 -> L1LU -> L2SU -> L2. The hop on L1LU and the hop on
// L2SU are next to each other, joined by a new "L1LU" -> "L2SU" FIFO. L2SU
// writes straight into the original L2 memref, which stage 1 keeps reading.
//
// Stage 0 also loads %b for the compute stage through L1 -> L1LU. That hop
// goes to the same L1LU stage, which gets one transfer for each flow.

// CHECK-LABEL: func.func @adjacent_hops(
// CHECK:         %[[P:.+]]:14 = ktdf.private
// CHECK-NEXT:      memref.alloc() : memref<64xf16, "L2">
// CHECK-NEXT:      memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:      memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFUSU", 64xf16>
// CHECK-NEXT:      memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L2LU" -> "SFU", 64xf16>
// CHECK-NEXT:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L2SU", 64xf16>
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[P]]#8) {
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}}[0] size [64] to %[[P]]#1[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}}[0] size [64] to %[[P]]#2[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["DDRLU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#8) depends_out(%[[P]]#9) {
// CHECK-NEXT:      ktdf.data_transfer from %[[P]]#1[0] size [64] to %[[P]]#7 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L2SU", 64xf16>
// CHECK-NEXT:      ktdf.data_transfer from %[[P]]#2[0] size [64] to %[[P]]#3 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L1LU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#9) depends_out(%[[P]]#10) {
// CHECK-NEXT:      ktdf.data_transfer from %[[P]]#7 size [64] to %[[P]]#0[0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L2SU", 64xf16>, memref<64xf16, "L2">
// CHECK-NEXT:    } {applicable_units = ["L2SU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#10) depends_out(%[[P]]#11) {
// CHECK-NEXT:      ktdf.data_transfer from %[[P]]#0[0] size [64] to %[[P]]#6 size [64] : memref<64xf16, "L2">, !ktdf.fifo.slot<"L2LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L2LU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#9, %[[P]]#11) depends_out(%[[P]]#12) {
// CHECK-NEXT:      ktdf.read_from_fifo %[[P]]#3 : <"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:      ktdf.read_from_fifo %[[P]]#6 : <"L2LU" -> "SFU", 64xf16>
// CHECK:           ktdf.write_to_fifo %{{.+}}, %[[P]]#4 : tensor<64xf16>, <"SFU" -> "SFUSU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["SFU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#12) depends_out(%[[P]]#13) {
// CHECK-NEXT:      ktdf.data_transfer from %[[P]]#4 size [64] to %[[P]]#5[0] size [64] : !ktdf.fifo.slot<"SFU" -> "SFUSU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["SFUSU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#13) depends_out(none) {
// CHECK-NEXT:      ktdf.data_transfer from %[[P]]#5[0] size [64] to %{{.+}}[0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
// CHECK-NEXT:    } {applicable_units = ["L1SU"]}

module {
  ktdf_arch.device @chained_units_device attributes {} import("Inputs/chained_units_device.mlir")
  func.func @adjacent_hops(%a: memref<64xf16, "DDR">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:7 = ktdf.private -> (!ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L2" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<64xf16, "L2">, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L2" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
        %l2 = memref.alloc() : memref<64xf16, "L2">
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %l2, %t0, %t1, %t2 : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L2" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<64xf16, "L2">, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#3[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L2">
        ktdf.data_transfer from %b[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#4) depends_out(%p#5) {
        ktdf.data_transfer from %p#3[%c0] size [64] to %p#1 size [64] : memref<64xf16, "L2">, !ktdf.fifo.slot<"L2" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#4, %p#5) depends_out(%p#6) {
        %x = ktdf.read_from_fifo %p#0 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"L2" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %y : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "L1", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#6) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}
