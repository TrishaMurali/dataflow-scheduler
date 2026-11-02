// RUN: dataflow-scheduler-opt --access-granularity-legalization %s | FileCheck %s

// Verify that AccessGranularityLegalizationPass widens a load transfer to the
// load unit's minimum access granularity.
//
// === f32, L1LU, L1 memory ===
// L1LU word_size = 1 byte, access_granularity for "L1": [64, 8, 2] words.
// For f32 (4 bytes/elem): elem_words = 4, min_words = 1*4 = 4,
//   fitAccess(4) -> {size_in_words = 8}
//   granule_elements = 8 / 4 = 2   (widening from 1 to 2 occurs)
//
// === f16, L1LU, L1 memory ===
// For f16 (2 bytes/elem): elem_words = 2, min_words = 1*2 = 2,
//   fitAccess(2) -> {size_in_words = 2}
//   granule_elements = 2 / 2 = 1   (no widening)

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// ---------------------------------------------------------------------------
// f32: widening occurs — source and destination sizes grow from 1 to 2, and
// the destination buffer is resized from memref<1xf32,"L1"> to memref<2xf32,"L1">.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @widen_f32
// CHECK: ktdf.data_transfer from %{{.*}}[0] size [2] to %{{.*}}[0] size [2]
// CHECK-SAME:   : memref<2xf32, "L1">, memref<2xf32, "L1">

module {
  func.func @widen_f32(%src: memref<2xf32, "L1">) {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (memref<1xf32, "L1">, !ktdf.token) {
        %buf = memref.alloc() : memref<1xf32, "L1">
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %buf, %t : memref<1xf32, "L1">, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0] size [1] to %p#0[0] size [1]
            : memref<2xf32, "L1">, memref<1xf32, "L1">
      } {applicable_units = ["L1LU"]}
    }
    return
  }
}

// ---------------------------------------------------------------------------
// f16: no widening — sizes stay at 1, buffer unchanged.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @no_widen_f16
// CHECK: ktdf.data_transfer from %{{.*}}[0] size [1] to %{{.*}}[0] size [1]
// CHECK-SAME:   : memref<2xf16, "L1">, memref<1xf16, "L1">

module {
  func.func @no_widen_f16(%src: memref<2xf16, "L1">) {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (memref<1xf16, "L1">, !ktdf.token) {
        %buf = memref.alloc() : memref<1xf16, "L1">
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %buf, %t : memref<1xf16, "L1">, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.data_transfer from %src[0] size [1] to %p#0[0] size [1]
            : memref<2xf16, "L1">, memref<1xf16, "L1">
      } {applicable_units = ["L1LU"]}
    }
    return
  }
}
