// RUN: dataflow-scheduler-opt %s --allow-unregistered-dialect \
// RUN:   --pass-pipeline="builtin.module(builtin.module(func.func(simplify-pipeline)))" \
// RUN:   | FileCheck %s

// Tests that a pipeline nested in a stage of another, as produced for a
// scheduled matmul in a nested module, has its stages ordered in front of their
// consumers.

// CHECK-LABEL: func.func @matmul
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.mnilu"
// CHECK:           ktdf.stage
// CHECK:             ktdf.parallel
// CHECK-NEXT:          ktdf.pipeline {
// CHECK:                 ktdf.stage
// CHECK-NEXT:              "test.l1lu"
// CHECK:                 ktdf.stage
// CHECK-NEXT:              "test.l0su"
// CHECK:                 ktdf.stage
// CHECK-NEXT:              "test.l0lu"
// CHECK:                 ktdf.stage
// CHECK-NEXT:              "test.mpe"
// CHECK:                 ktdf.stage
// CHECK-NEXT:              "test.l1su"
// CHECK:               }
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.mnisu"
module {
  module @local_schedule_0 {
    func.func @matmul() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      ktdf.pipeline {
        %t:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %t0, %t1 : !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%t#0) {
          "test.mnilu"() : () -> ()
        }
        ktdf.stage depends_in(%t#0) depends_out(%t#1) {
          ktdf.parallel (%i, %j) = (%c0) to (%c64) step (%c1) distribute(num_instances = 2) {
            ktdf.pipeline {
              %u:4 = ktdf.private -> (!ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
                %u0 = ktdf.create_token : !ktdf.token
                %u1 = ktdf.create_token : !ktdf.token
                %u2 = ktdf.create_token : !ktdf.token
                %u3 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %u0, %u1, %u2, %u3 : !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(%u#0) depends_out(%u#1) {
                "test.l0su"() : () -> ()
              }
              ktdf.stage depends_in(none) depends_out(%u#0) {
                "test.l1lu"() : () -> ()
              }
              ktdf.stage depends_in(%u#0, %u#3) depends_out(%u#2) {
                "test.mpe"() : () -> ()
              }
              ktdf.stage depends_in(%u#1) depends_out(%u#3) {
                "test.l0lu"() : () -> ()
              }
              ktdf.stage depends_in(%u#2) depends_out(none) {
                "test.l1su"() : () -> ()
              }
            }
            ktdf.parallel_yield
          }
        }
        ktdf.stage depends_in(%t#1) depends_out(none) {
          "test.mnisu"() : () -> ()
        }
      }
      return
    }
  }
}
