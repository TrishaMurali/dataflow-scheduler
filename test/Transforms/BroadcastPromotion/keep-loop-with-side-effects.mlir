// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// The transfer does not read the variable of either loop, but the index it does
// read comes from a load, so the iterations are not all the same one: what the
// load answers is about memory, and memory is not the pruner's to reason about.
// The inner loop is cloned rather than collapsed into a single run of its body.

// CHECK-LABEL:   func.func @keeps_loop_with_a_load
// CHECK:           ktdf.pipeline {
// CHECK:             ktdf.stage
// CHECK:               scf.for
// CHECK-NEXT:            scf.for
// CHECK-NEXT:              %[[FROM:.*]] = memref.load
// CHECK:                   ktdf.data_transfer from %{{.*}}{{\[}}%[[FROM]]]

module {
  func.func @keeps_loop_with_a_load(%A: memref<?xf16, "DDR">,
                                    %where: memref<1xindex, "L1">,
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
            scf.for %j = %c0 to %M step %c1 {
              %from = memref.load %where[%c0] : memref<1xindex, "L1">
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
