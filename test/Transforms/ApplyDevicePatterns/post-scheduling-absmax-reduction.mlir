// RUN: dataflow-scheduler-opt %s -allow-unregistered-dialect -apply-device-patterns='groups=post_scheduling' | FileCheck %s

// CHECK-LABEL:   func.func @absmax_reduction() {
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<64x32xf16, "L1">
// CHECK-NEXT:     %[[ALLOC_1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:     "test.simd_reduction_absmax"(%[[ALLOC_0]], %[[ALLOC_1]]) : (memref<64x32xf16, "L1">, memref<64xf16, "L1">) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// Verify that the post_scheduling PDL pattern in the sample device rewrites a
// linalg.generic absmax-reduction (inner-dim, output via subview) into a bare
// test.simd_absmax op whose second operand is the subview's *source*.

#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  func.func @absmax_reduction() {
    %alloc_in  = memref.alloc() : memref<64x32xf16, "L1">
    %alloc_acc = memref.alloc() : memref<64xf16, "L1">

    // The pattern requires the output to be a memref.subview of the accumulator.
    %subview = memref.subview %alloc_acc[0][64][1]
        : memref<64xf16, "L1"> to memref<64xf16, "L1">

    linalg.generic {
      indexing_maps = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]
    } ins(%alloc_in : memref<64x32xf16, "L1">)
      outs(%subview  : memref<64xf16, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %abs_in  = math.absf %in  : f16
      %abs_out = math.absf %out : f16
      %result  = arith.maxnumf %abs_in, %abs_out : f16
      linalg.yield %result : f16
    }

    return
  }
}
