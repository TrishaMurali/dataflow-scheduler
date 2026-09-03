// A register in the body of a reduction is sized by the parallel dimensions
// only. The reduction dimension is iterated rather than held, so counting it
// would size the register to every step of the reduction instead of the
// accumulator it carries -- here 4 by 64 rather than 64.

// RUN: dataflow-scheduler-opt -materialize-registers %s | FileCheck %s

// CHECK-LABEL: func.func @register_in_a_reduction(
// CHECK:       memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NOT:   memref.alloca() : memref<256xf16, "SFU_REG">

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>
module {
  func.func @register_in_a_reduction(%in: tensor<4x64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#in, #out],
                              iterator_types = ["reduction", "parallel"]}
        ins(%in : tensor<4x64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %held = memref.alloca() : memref<f16, "SFU_REG">
      memref.store %x, %held[] : memref<f16, "SFU_REG">
      %read = memref.load %held[] : memref<f16, "SFU_REG">
      %sum = arith.addf %read, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
