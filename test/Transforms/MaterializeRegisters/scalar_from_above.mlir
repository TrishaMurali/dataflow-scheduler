// A register the body does not compute per element. The scale is a scalar the
// caller passes in, so storing it fills the whole register rather than one lane.
// The data register keeps its per-element store, which becomes lane zero.
//
// The arithmetic computing the scale already stands in front of the generic:
// hoist-invariants runs before this pass and takes pure ops out of the body.

// RUN: dataflow-scheduler-opt -materialize-registers %s | FileCheck %s

// CHECK-LABEL: func.func @scalar_from_above(
// CHECK-SAME:      %{{.*}}: tensor<64xf16>,
// CHECK-SAME:      %[[SCALE_IN:.*]]: index)
// CHECK:       %[[NARROW:.*]] = arith.index_castui %[[SCALE_IN]] : index to i32
// CHECK:       %[[WIDE:.*]] = arith.uitofp %[[NARROW]] : i32 to f16
// CHECK:       %[[IN:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[SCALE:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:  linalg.fill ins(%[[WIDE]] : f16) outs(%[[SCALE]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:  %[[OUT:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK:       linalg.generic
// CHECK-NEXT:  ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:    memref.store %[[X]], %[[IN]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:    ktdf.opaque "fake_scale"
// CHECK-NEXT:      ins(%[[IN]], %[[SCALE]]: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
// CHECK-NEXT:      outs(%[[OUT]]: memref<64xf16, "SFU_REG">)
// CHECK:         %[[E:.*]] = memref.load %[[OUT]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:    linalg.yield %[[E]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @scalar_from_above(%in: tensor<64xf16>, %scale: index) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %narrow = arith.index_castui %scale : index to i32
    %widened = arith.uitofp %narrow : i32 to f16
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %data = memref.alloca() : memref<f16, "SFU_REG">
      memref.store %x, %data[] : memref<f16, "SFU_REG">
      %scalar = memref.alloca() : memref<f16, "SFU_REG">
      memref.store %widened, %scalar[] : memref<f16, "SFU_REG">
      %outreg = memref.alloca() : memref<f16, "SFU_REG">
      ktdf.opaque "fake_scale" {func_name = "fake_scale",
                                dataflow_scheduler.register_names = ["in0", "scale", "out0"]}
        ins(%data, %scalar : memref<f16, "SFU_REG">, memref<f16, "SFU_REG">)
        outs(%outreg : memref<f16, "SFU_REG">)
      %e = memref.load %outreg[] : memref<f16, "SFU_REG">
      linalg.yield %e : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
