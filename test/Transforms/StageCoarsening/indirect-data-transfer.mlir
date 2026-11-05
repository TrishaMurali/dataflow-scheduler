// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s

// CHECK-LABEL: func.func @indirect_transfer_coarsening
// CHECK-SAME:    (%[[IAB:.*]]: memref<32xi32, strided<[1], offset: ?>, "IAB">, %[[SRC:.*]]: memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         ktdf.pipeline {
// CHECK-NEXT:      %[[PRIV:.*]]:3 = ktdf.private -> (memref<2x64xf16, "L1">, memref<64xf16, "SFU_REG">, !ktdf.token) {
// CHECK-NEXT:        %[[ALLOC0:.*]] = memref.alloc() : memref<2x64xf16, "L1">
// CHECK-NEXT:        %[[ALLOC1:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:        %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:        ktdf.private_yield %[[ALLOC0]], %[[ALLOC1]], %[[TOK]] : memref<2x64xf16, "L1">, memref<64xf16, "SFU_REG">, !ktdf.token
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(none) depends_out(%[[PRIV]]#2) {
// CHECK-NEXT:        scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:          %[[SUB:.*]] = arith.subi %[[IV]], %[[C0]] : index
// CHECK-NEXT:          %[[NORM_IV:.*]] = arith.divsi %[[SUB]], %[[C1]] : index
// CHECK-NEXT:          ktdf.ind_data_transfer
// CHECK-NEXT:            ind_src = %[[IAB]][%[[C0]]]
// CHECK-NEXT:            dir_src = %[[SRC]][0, %[[IV]], 0] size [1, 1, 64]
// CHECK-NEXT:            ind_dst = none
// CHECK-NEXT:            dir_dst = %[[PRIV]]#0[%[[NORM_IV]], %[[C0]]] size [1, 64]
// CHECK-NEXT:            : memref<32xi32, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, memref<2x64xf16, "L1">
// CHECK-NEXT:        } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      } {applicable_units = ["MNILU"]}
// CHECK-NEXT:      ktdf.stage depends_in(%[[PRIV]]#2) depends_out(none) {
// CHECK-NEXT:        scf.for %[[IV2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:          %[[SUB2:.*]] = arith.subi %[[IV2]], %[[C0]] : index
// CHECK-NEXT:          %[[NORM_IV2:.*]] = arith.divsi %[[SUB2]], %[[C1]] : index
// CHECK-NEXT:          ktdf.data_transfer from %[[PRIV]]#0[%[[NORM_IV2]], 0] size [1, 64] to %[[PRIV]]#1[0] size [64] : memref<2x64xf16, "L1">, memref<64xf16, "SFU_REG">
// CHECK-NEXT:        } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      } {applicable_units = ["SFU"]}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @indirect_transfer_coarsening(%iab: memref<32xi32, strided<[1], offset: ?>, "IAB">,
                                         %src: memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    scf.for %m = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %14:4 = ktdf.private -> (memref<64xf16, "L1">, memref<64xf16, "SFU_REG">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<64xf16, "L1">
          %alloc_reg = memref.alloc() : memref<64xf16, "SFU_REG">
          %tok0 = ktdf.create_token : !ktdf.token
          %tok1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_reg, %tok0, %tok1 : memref<64xf16, "L1">, memref<64xf16, "SFU_REG">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%14#2) {
          ktdf.ind_data_transfer
           ind_src = %iab[%c0]
           dir_src = %src[0, %m, 0] size [1, 1, 64]
           ind_dst = none
           dir_dst = %14#0[%c0] size [64]
           : memref<32xi32, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, memref<64xf16, "L1">
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%14#2) depends_out(%14#3) {
          ktdf.data_transfer from %14#0[0] size [64] to %14#1[0] size [64] : memref<64xf16, "L1">, memref<64xf16, "SFU_REG">
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
