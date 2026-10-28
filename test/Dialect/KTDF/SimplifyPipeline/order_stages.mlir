// RUN: dataflow-scheduler-opt --simplify-pipeline --allow-unregistered-dialect %s | FileCheck %s

// Tests that stages are ordered in front of the stages awaiting their tokens.

// A producer is moved in front of its consumer.
// CHECK-LABEL: func.func @producer_after_consumer
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.producer"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.consumer"
func.func @producer_after_consumer() {
  ktdf.pipeline {
    %t = ktdf.private -> (!ktdf.token) {
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t : !ktdf.token
    }
    ktdf.stage depends_in(%t) depends_out(none) {
      "test.consumer"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer"() : () -> ()
    }
  }
  return
}

// A chain is ordered transitively.
// CHECK-LABEL: func.func @reversed_chain
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.a"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.b"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.c"
func.func @reversed_chain() {
  ktdf.pipeline {
    %t:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t0, %t1 : !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(%t#1) depends_out(none) {
      "test.c"() : () -> ()
    }
    ktdf.stage depends_in(%t#0) depends_out(%t#1) {
      "test.b"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t#0) {
      "test.a"() : () -> ()
    }
  }
  return
}

// Stages already in front of their consumers, and independent stages, keep
// their order.
// CHECK-LABEL: func.func @already_ordered
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.independent"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.producer"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.consumer"
func.func @already_ordered() {
  ktdf.pipeline {
    %t = ktdf.private -> (!ktdf.token) {
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t : !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(none) {
      "test.independent"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer"() : () -> ()
    }
    ktdf.stage depends_in(%t) depends_out(none) {
      "test.consumer"() : () -> ()
    }
  }
  return
}

// A cycle keeps its relative order, and is placed in front of its consumers
// and behind its producers.
// CHECK-LABEL: func.func @cycle
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.producer"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.cycle_0"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.cycle_1"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.consumer"
func.func @cycle() {
  ktdf.pipeline {
    %t:4 = ktdf.private -> (!ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %t2 = ktdf.create_token : !ktdf.token
      %t3 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t0, %t1, %t2, %t3 : !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(%t#3) depends_out(none) {
      "test.consumer"() : () -> ()
    }
    ktdf.stage depends_in(%t#0, %t#2) depends_out(%t#1) {
      "test.cycle_0"() : () -> ()
    }
    ktdf.stage depends_in(%t#1) depends_out(%t#2, %t#3) {
      "test.cycle_1"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t#0) {
      "test.producer"() : () -> ()
    }
  }
  return
}

// Independent stages keep their relative order while being moved.
// CHECK-LABEL: func.func @independent_pairs
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.producer_0"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.consumer_0"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.producer_1"
// CHECK:           ktdf.stage
// CHECK-NEXT:        "test.consumer_1"
func.func @independent_pairs() {
  ktdf.pipeline {
    %t:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t0, %t1 : !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(%t#0) depends_out(none) {
      "test.consumer_0"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t#0) {
      "test.producer_0"() : () -> ()
    }
    ktdf.stage depends_in(%t#1) depends_out(none) {
      "test.consumer_1"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t#1) {
      "test.producer_1"() : () -> ()
    }
  }
  return
}

