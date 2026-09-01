// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A transfer under a loop that carries a value out is not offered for hoisting.
//
// The clone a hoist makes is pruned to the transfer, and what the loop yields is
// computed by ops the pruning drops -- leaving the yield with nothing to hand
// back. So the transfer stays where it is, loop and all.

// CHECK-LABEL:   func.func @loop_carries_a_value
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// The one pipeline is the one that was there; no sibling in front of the loop.
// CHECK-NEXT:      scf.for
// CHECK-NEXT:        ktdf.pipeline {
// CHECK:               ktdf.stage
// CHECK:                 scf.for
// CHECK:                   %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK:                     ktdf.data_transfer

module {
  func.func @loop_carries_a_value(%A: memref<?xf16, "DDR">,
                                         %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %n = %c0 to %N step %c1 {
      ktdf.pipeline {
        %r:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
          %a = memref.alloc() : memref<64xf16, "L1">
          %k = ktdf.create_token : !ktdf.token
          ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%r#1) {
          scf.for %i = %c0 to %M step %c1 {
            %count = scf.for %j = %c0 to %M step %c1
                iter_args(%acc = %c0) -> (index) {
              %from = arith.addi %i, %c0 : index
              %into = arith.addi %i, %c1 : index
              ktdf.data_transfer from %A[%from] size [64] to %r#0[%into] size [64]
                : memref<?xf16, "DDR">, memref<64xf16, "L1">
              %next = arith.addi %acc, %c1 : index
              scf.yield %next : index
            }
            "test.use"(%count) : (index) -> ()
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
