// RUN: dataflow-scheduler-opt --path-expansion --split-input-file \
// RUN:   --verify-diagnostics %s

// The data transfers of a stage must all read from the same place or all
// write to the same place. A place is a memory, for a memref, or a load/store
// unit named in a FIFO type.

// The FIFOs here name "DDR", which is a memory and only a placeholder, so they
// do not count. Here the load stage reads from L1 and from DDR and
// writes to two different FIFOs, so it has no single memory and is rejected.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @mixed_sources_and_dests(%a: memref<64xf16, "L1">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %d: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:6 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %f3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %f3, %t0, %t1 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-error @below {{path-expansion: the data transfers of a stage must all read from the same place or all write to the same place}}
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#4) depends_out(%p#5) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %y : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
        ktdf.write_to_fifo %x, %p#3 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#5) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
        ktdf.data_transfer from %p#3 size [64] to %d[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}

// -----

// The load stage reads DDR, but its FIFOs start at L1LU, which reads L1. The
// two ends disagree about where the stage runs.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unit_mismatch(%a: memref<64xf16, "DDR">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
      }
      // expected-error @below {{path-expansion: the stage uses memory "DDR", but its FIFOs place it on unit "L1LU", which works on "L1"}}
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %y : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#4) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}
