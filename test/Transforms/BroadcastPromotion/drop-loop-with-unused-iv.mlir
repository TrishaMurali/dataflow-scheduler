// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A donor stage whose loop nest the transfer only half uses. The indices the
// transfer reads come from %i, so the loop over it is cloned with the transfer.
// Nothing the transfer reads comes from %j, and the clone is pruned to the
// transfer, so all that is left in that loop is the transfer and the pure ops
// computing its indices -- every iteration the same values and the same write.
// One run stands for all of them, and the loop goes.
//
// It has to go, not merely for tidiness. Its bound is derived from the induction
// variable of the loop just hoisted above, so a clone that keeps the loop reads a
// value defined below itself and the module stops verifying.
//
// What stops this where the iterations are not all alike is in
// no-hoist-loop-carries-a-value.mlir and keep-loop-with-side-effects.mlir.

// CHECK-LABEL:   func.func @drop_loop_with_unused_iv
// CHECK:           scf.for %[[I:.*]] = %[[C0:.*]] to %{{.*}} step %[[C1:.*]] {
// CHECK-NEXT:        %[[FROM:.*]] = arith.addi %[[I]], %[[C0]]
// CHECK-NEXT:        %[[INTO:.*]] = arith.addi %[[I]], %[[C1]]
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}{{\[}}%[[FROM]]] size [64] to %{{.*}}{{\[}}%[[INTO]]] size [64]
// CHECK-NEXT:      }
// The clone holds that one loop and nothing else, so the loop over %j is gone
// and with it the only use of the bound derived inside the loop hoisted above.
// CHECK-NOT:       scf.for
// CHECK-NOT:       test.other
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

module {
  func.func @drop_loop_with_unused_iv(%A: memref<?xf16, "DDR">,
                                      %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %n = %c0 to %N step %c1 {
      // Derived from the loop the transfer is hoisted above, so it is not
      // available to anything placed in front of that loop.
      %bound = "test.derive"(%n) : (index) -> index
      ktdf.pipeline {
        %r:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
          %a = memref.alloc() : memref<64xf16, "L1">
          %k = ktdf.create_token : !ktdf.token
          ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%r#1) {
          scf.for %i = %c0 to %M step %c1 {
            scf.for %j = %c0 to %bound step %c1 {
              // Reads %j, and is not cloned with the transfer. Whether the loop
              // over %j is needed is about what the clone keeps, not about what
              // the donor holds.
              "test.other"(%j) : (index) -> ()
              %from = arith.addi %i, %c0 : index
              %into = arith.addi %i, %c1 : index
              ktdf.data_transfer from %A[%from] size [64] to %r#0[%into] size [64]
                : memref<?xf16, "DDR">, memref<64xf16, "L1">
            }
          }
        }
        ktdf.stage depends_in(%r#1) depends_out(none) {
          "test.use"(%r#0) : (memref<64xf16, "L1">) -> ()
        }
      }
    }
    return
  }
}
