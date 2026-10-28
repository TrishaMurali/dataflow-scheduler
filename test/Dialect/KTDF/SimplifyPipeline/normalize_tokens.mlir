// RUN: dataflow-scheduler-opt --simplify-pipeline --allow-unregistered-dialect %s | FileCheck %s

// Tests that the dependency tokens of a pipeline are normalized without
// changing its dependency edges.

// Tokens repeated in the inputs or outputs of a stage are deduplicated.
// CHECK-LABEL: func.func @repeated_tokens
// CHECK:         %[[T:.+]] = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]) {
// CHECK:         ktdf.stage depends_in(%[[T]]) depends_out(none) {
func.func @repeated_tokens() {
  ktdf.pipeline {
    %t = ktdf.private -> (!ktdf.token) {
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t : !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t, %t) {
      "test.producer"() : () -> ()
    }
    ktdf.stage depends_in(%t, %t) depends_out(none) {
      "test.consumer"() : () -> ()
    }
  }
  return
}

// A token output by several stages is split into one per producer, which the
// consumers await.
// CHECK-LABEL: func.func @shared_output
// CHECK:         %[[T:.+]]:2 = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]#0) {
// CHECK-NEXT:      "test.producer_0"
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]#1) {
// CHECK-NEXT:      "test.producer_1"
// CHECK:         ktdf.stage depends_in(%[[T]]#0, %[[T]]#1) depends_out(none) {
// CHECK-NEXT:      "test.consumer"
func.func @shared_output() {
  ktdf.pipeline {
    %t = ktdf.private -> (!ktdf.token) {
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t : !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer_0"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer_1"() : () -> ()
    }
    ktdf.stage depends_in(%t) depends_out(none) {
      "test.consumer"() : () -> ()
    }
  }
  return
}

// The several outputs of a stage are merged into one, which all consumers
// await.
// CHECK-LABEL: func.func @several_outputs
// CHECK:         %[[T:.+]] = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]) {
// CHECK-NEXT:      "test.producer"
// CHECK:         ktdf.stage depends_in(%[[T]]) depends_out(none) {
// CHECK-NEXT:      "test.consumer_0"
// CHECK:         ktdf.stage depends_in(%[[T]]) depends_out(none) {
// CHECK-NEXT:      "test.consumer_1"
func.func @several_outputs() {
  ktdf.pipeline {
    %t:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t0, %t1 : !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t#0, %t#1) {
      "test.producer"() : () -> ()
    }
    ktdf.stage depends_in(%t#0) depends_out(none) {
      "test.consumer_0"() : () -> ()
    }
    ktdf.stage depends_in(%t#1) depends_out(none) {
      "test.consumer_1"() : () -> ()
    }
  }
  return
}

// Edges implied by other edges are kept.
// CHECK-LABEL: func.func @implied_edge_kept
// CHECK:         %[[T:.+]]:2 = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]#0) {
// CHECK-NEXT:      "test.a"
// CHECK:         ktdf.stage depends_in(%[[T]]#0) depends_out(%[[T]]#1) {
// CHECK-NEXT:      "test.b"
// CHECK:         ktdf.stage depends_in(%[[T]]#0, %[[T]]#1) depends_out(none) {
// CHECK-NEXT:      "test.c"
func.func @implied_edge_kept() {
  ktdf.pipeline {
    %t:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t0, %t1 : !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t#0) {
      "test.a"() : () -> ()
    }
    ktdf.stage depends_in(%t#0) depends_out(%t#1) {
      "test.b"() : () -> ()
    }
    ktdf.stage depends_in(%t#0, %t#1) depends_out(none) {
      "test.c"() : () -> ()
    }
  }
  return
}

// Tokens used outside of the stages of the pipeline are kept as they are.
// CHECK-LABEL: func.func @foreign_token
// CHECK:         %[[T:.+]] = ktdf.create_token
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]) {
// CHECK-NEXT:      "test.producer_0"
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]) {
// CHECK-NEXT:      "test.producer_1"
// CHECK:         ktdf.stage depends_in(%[[T]]) depends_out(none) {
// CHECK-NEXT:      "test.consumer"
// CHECK:         "test.user"(%[[T]])
func.func @foreign_token() {
  %t = ktdf.create_token : !ktdf.token
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer_0"() : () -> ()
    }
    ktdf.stage depends_in(none) depends_out(%t) {
      "test.producer_1"() : () -> ()
    }
    ktdf.stage depends_in(%t) depends_out(none) {
      "test.consumer"() : () -> ()
    }
  }
  "test.user"(%t) : (!ktdf.token) -> ()
  return
}
