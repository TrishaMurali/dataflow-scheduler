// What a device's substitution leaves in a generic's body, and what this pass
// makes of it. The sample device substitutes exp for an opaque over four
// registers: a constant the template reads, one for the data each way, and one
// it computes in.
//
// The pattern allocates each as a single element, since where it matched it
// cannot say how many lanes one holds. They come out in front of the generic
// sized to the tile it covers, and the scalar store and load the body did
// become lane zero of the register they were on.
//
// Substituting on its own leaves the opaque printed generically: the device is
// what asks for it, and nothing in that run loads its dialect. This pass names
// it among its own, so the second run prints it as itself.

// RUN: dataflow-scheduler-opt %s -allow-unregistered-dialect \
// RUN:   -ktdfarch-apply-patterns=groups=pre_scheduling \
// RUN:   | FileCheck %s --check-prefix=SUBST
// RUN: dataflow-scheduler-opt %s \
// RUN:   -ktdfarch-apply-patterns=groups=pre_scheduling -materialize-registers \
// RUN:   | FileCheck %s

// The substitution on its own leaves the registers in the body, one element
// each and nothing said about how many lanes one holds. The constant is a
// constant still, carrying the memory it belongs to.
// SUBST:      %[[C0:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.000000e+00 : f16
// SUBST:      linalg.generic
// SUBST-NEXT: ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// SUBST-NEXT:   %[[IN:.*]] = memref.alloca() : memref<f16, "SFU_REG">
// SUBST-NEXT:   memref.store %[[X]], %[[IN]][] : memref<f16, "SFU_REG">
// SUBST-NEXT:   %[[OUT:.*]] = memref.alloca() : memref<f16, "SFU_REG">
// SUBST-NEXT:   %[[T0:.*]] = memref.alloca() : memref<f16, "SFU_REG">
// SUBST-NEXT:   "ktdf.opaque"(%[[C0]], %[[IN]], %[[OUT]], %[[T0]])
// SUBST:        %[[E:.*]] = memref.load %[[OUT]][] : memref<f16, "SFU_REG">
// SUBST-NEXT:   linalg.yield %[[E]] : f16

// Made real, every register stands in front of the generic and holds 64 lanes of
// f16: the tile the generic covers, and what the device gives that element. The
// constant is filled across the whole of its own.
// CHECK:      %[[C0:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.000000e+00 : f16
// CHECK:      %[[C0_REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT: linalg.fill ins(%[[C0]] : f16) outs(%[[C0_REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT: %[[IN:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[OUT:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT: %[[T0:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:   memref.store %[[X]], %[[IN]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   ktdf.opaque "fake_exp" {dataflow_scheduler.register_names = ["c0", "in0", "out0", "t0_0"], func_name = "fake_exp"}
// CHECK-NEXT:     ins(%[[C0_REG]], %[[IN]]: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
// CHECK-NEXT:     outs(%[[OUT]], %[[T0]]: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
// CHECK:        %[[E:.*]] = memref.load %[[OUT]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   linalg.yield %[[E]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @exp(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %e = spyreop.exp %x : f16
      linalg.yield %e : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
