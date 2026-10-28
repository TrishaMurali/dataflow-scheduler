// RUN: dataflow-scheduler-opt --simplify-pipeline --allow-unregistered-dialect %s | FileCheck %s

// Tests that the ktdf.private op is moved to the top of its pipeline.

// The private is hoisted above the stages preceding it.
// CHECK-LABEL: func.func @private_after_stage
// CHECK:         ktdf.pipeline {
// CHECK-NEXT:      ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.first"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.second"
func.func @private_after_stage() {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      "test.first"() : () -> ()
    }
    %m = ktdf.private -> (memref<4xf16>) {
      %m = memref.alloc() : memref<4xf16>
      ktdf.private_yield %m : memref<4xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      "test.second"(%m) : (memref<4xf16>) -> ()
    }
  }
  return
}

// Stages using results of the private are ordered in front of their consumers
// once it is hoisted.
// CHECK-LABEL: func.func @private_operands
// CHECK:         ktdf.pipeline {
// CHECK-NEXT:      ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.producer"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.consumer"
func.func @private_operands() {
  %t = ktdf.create_token : !ktdf.token
  ktdf.pipeline {
    ktdf.stage depends_in(%t) depends_out(none) {
      "test.consumer"() : () -> ()
    }
    %m = ktdf.private -> (memref<4xf16>) {
      %m = memref.alloc() : memref<4xf16>
      ktdf.private_yield %m : memref<4xf16>
    }
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer"(%m) : (memref<4xf16>) -> ()
    }
  }
  return
}
