// RUN: dataflow-scheduler-opt --path-expansion --split-input-file %s | FileCheck %s

// Stages with several data transfers. Either all transfers of a stage read
// from the same place or all of them write to the same place; the other ends
// may differ. A place is a memory, for a memref, or a load/store unit named in
// a FIFO type.

// The load stage reads DDR twice and the store stage writes DDR twice. Each
// gets one stage_resource (DDR), and every transfer is routed through L1.
// CHECK-LABEL: func.func @shared_ends(
// CHECK:         ktdf.stage depends_in(none)
// CHECK-NEXT:    ktdf.data_transfer from %arg0[0] size [64] to %{{.*}}[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.data_transfer from %arg1[0] size [64] to %{{.*}}[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK-NEXT:    ktdf.data_transfer {{.*}} : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    ktdf.data_transfer {{.*}} : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L1LU"]}
// CHECK:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK-NEXT:    ktdf.data_transfer {{.*}} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.data_transfer {{.*}} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["L1SU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK-NEXT:    ktdf.data_transfer from %{{.*}}[0] size [64] to %arg2[0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
// CHECK-NEXT:    ktdf.data_transfer from %{{.*}}[0] size [64] to %arg3[0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @shared_ends(%a: memref<64xf16, "DDR">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %d: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
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
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
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

// The load stage reads DDR into two FIFOs and also copies DDR to an L1 memref.
// The destinations differ but every source is DDR, so the stage is a load
// stage on DDR. The DDR -> L1 copy needs no hop and is kept as it is.
// CHECK-LABEL: func.func @load_mixed_dests(
// CHECK:         ktdf.stage depends_in(none)
// CHECK-NEXT:    ktdf.data_transfer from %arg1[0] size [64] to %{{.*}}[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.data_transfer from %arg2[0] size [64] to %{{.*}}[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.data_transfer from %arg2[%{{.*}}] size [64] to %arg0[%{{.*}}] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK-NEXT:    ktdf.data_transfer {{.*}} : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    ktdf.data_transfer {{.*}} : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L1LU"]}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @load_mixed_dests(%l: memref<64xf16, "L1">, %a: memref<64xf16, "DDR">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %d: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
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
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        ktdf.data_transfer from %b[%c0] size [64] to %l[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
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

// The load stage reads two L1 memrefs into two FIFOs of type "L1LU" -> "SFU".
// Every FIFO starts at L1LU, so the stage runs on L1LU and its memory is L1,
// which L1LU reads from. Its FIFOs need no new type and no hop is added
// between it and the compute stage.
// CHECK-LABEL: func.func @unit_from_fifo(
// CHECK:         ktdf.stage depends_in(none)
// CHECK-NEXT:    ktdf.data_transfer from %arg0[%{{.*}}] size [64] to %{{.*}} size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    ktdf.data_transfer from %arg1[%{{.*}}] size [64] to %{{.*}} size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L1LU"]}
// CHECK-NEXT:    ktdf.stage
// CHECK:         } {applicable_units = ["SFU"]}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unit_from_fifo(%a: memref<64xf16, "L1">, %b: memref<64xf16, "L1">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
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
