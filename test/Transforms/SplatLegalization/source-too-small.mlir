// RUN: dataflow-scheduler-opt --splat-legalization --verify-diagnostics --split-input-file %s

// Verify that SplatLegalizationPass refuses to widen a splat load past the end
// of its source buffer.  For f32 on L1LU the load grows from 1 element to 2
// (see basic.mlir for the numbers), so the source needs room for 2 elements
// from where the load starts.

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// The buffer holds a single element, so a 2-element load cannot fit.
module {
  func.func @one_element_buffer(%src: memref<1x1x1xf32, "L1">) {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token) {
        %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %slot, %t : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        // expected-error @+1 {{the hardware loads 2 elements for this splat, but its source buffer has only 1 element(s) from where the load starts}}
        ktdf.data_transfer from %src[0, 0, 0] size [1, 1, 1] to %p#0 size [32]
            {transfer_mode = #ktdf.transfer_mode<splat>}
            : memref<1x1x1xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      } {applicable_units = ["L1LU"]}
    }
    return
  }
}

// -----

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// The buffer is wide enough overall, but the load starts at its last element.
module {
  func.func @load_starts_at_last_element(%src: memref<1x1x4xf32, "L1">) {
    %c3 = arith.constant 3 : index
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token) {
        %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %slot, %t : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        // expected-error @+1 {{its source buffer has only 1 element(s) from where the load starts}}
        ktdf.data_transfer from %src[0, 0, %c3] size [1, 1, 1] to %p#0 size [32]
            {transfer_mode = #ktdf.transfer_mode<splat>}
            : memref<1x1x4xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      } {applicable_units = ["L1LU"]}
    }
    return
  }
}
