// RUN: dataflow-scheduler-opt --simplify-pipeline="reduce-dependencies" --allow-unregistered-dialect %s | FileCheck %s

// Tests that dependency edges implied by other edges are erased.

// An edge implied by a longer path is erased.
// CHECK-LABEL: func.func @implied_edge
// CHECK:         %[[T:.+]]:2 = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]#0) {
// CHECK-NEXT:      "test.a"
// CHECK:         ktdf.stage depends_in(%[[T]]#0) depends_out(%[[T]]#1) {
// CHECK-NEXT:      "test.b"
// CHECK:         ktdf.stage depends_in(%[[T]]#1) depends_out(none) {
// CHECK-NEXT:      "test.c"
func.func @implied_edge() {
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

// An edge implied by a path through a cycle is erased, but the edges of the
// cycle are kept.
// CHECK-LABEL: func.func @implied_through_cycle
// CHECK:         %[[T:.+]]:3 = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[T]]#0) {
// CHECK-NEXT:      "test.producer"
// CHECK:         ktdf.stage depends_in(%[[T]]#0, %[[T]]#2) depends_out(%[[T]]#1) {
// CHECK-NEXT:      "test.cycle_0"
// CHECK:         ktdf.stage depends_in(%[[T]]#1) depends_out(%[[T]]#2) {
// CHECK-NEXT:      "test.cycle_1"
// CHECK:         ktdf.stage depends_in(%[[T]]#2) depends_out(none) {
// CHECK-NEXT:      "test.consumer"
func.func @implied_through_cycle() {
  ktdf.pipeline {
    %t:3 = ktdf.private -> (!ktdf.token, !ktdf.token, !ktdf.token) {
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %t2 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %t0, %t1, %t2 : !ktdf.token, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t#0) {
      "test.producer"() : () -> ()
    }
    ktdf.stage depends_in(%t#0, %t#2) depends_out(%t#1) {
      "test.cycle_0"() : () -> ()
    }
    ktdf.stage depends_in(%t#1) depends_out(%t#2) {
      "test.cycle_1"() : () -> ()
    }
    ktdf.stage depends_in(%t#0, %t#2) depends_out(none) {
      "test.consumer"() : () -> ()
    }
  }
  return
}

// Edges over a token used outside of the stages of the pipeline are kept,
// while the local edges they would imply are still reduced.
// CHECK-LABEL: func.func @foreign_token
// CHECK:         %[[T:.+]] = ktdf.create_token
// CHECK:         %[[P:.+]]:2 = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[P]]#0) {
// CHECK-NEXT:      "test.a"
// CHECK:         ktdf.stage depends_in(%[[P]]#0) depends_out(%[[T]], %[[P]]#1) {
// CHECK-NEXT:      "test.b"
// CHECK:         ktdf.stage depends_in(%[[T]], %[[P]]#1) depends_out(none) {
// CHECK-NEXT:      "test.c"
// CHECK:         "test.user"(%[[T]])
func.func @foreign_token() {
  %t = ktdf.create_token : !ktdf.token
  ktdf.pipeline {
    %p:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
      %p0 = ktdf.create_token : !ktdf.token
      %p1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %p0, %p1 : !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p#0) {
      "test.a"() : () -> ()
    }
    ktdf.stage depends_in(%p#0) depends_out(%t, %p#1) {
      "test.b"() : () -> ()
    }
    ktdf.stage depends_in(%p#0, %t, %p#1) depends_out(none) {
      "test.c"() : () -> ()
    }
  }
  "test.user"(%t) : (!ktdf.token) -> ()
  return
}

// A local edge implied only by a path over a foreign token is kept.
// CHECK-LABEL: func.func @implied_by_foreign
// CHECK:         %[[T:.+]] = ktdf.create_token
// CHECK:         %[[P:.+]] = ktdf.private
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[P]]) {
// CHECK-NEXT:      "test.a"
// CHECK:         ktdf.stage depends_in(%[[P]]) depends_out(%[[T]]) {
// CHECK-NEXT:      "test.b"
// CHECK:         ktdf.stage depends_in(%[[P]], %[[T]]) depends_out(none) {
// CHECK-NEXT:      "test.c"
func.func @implied_by_foreign() {
  %t = ktdf.create_token : !ktdf.token
  ktdf.pipeline {
    %p = ktdf.private -> (!ktdf.token) {
      %p = ktdf.create_token : !ktdf.token
      ktdf.private_yield %p : !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p) {
      "test.a"() : () -> ()
    }
    ktdf.stage depends_in(%p) depends_out(%t) {
      "test.b"() : () -> ()
    }
    ktdf.stage depends_in(%p, %t) depends_out(none) {
      "test.c"() : () -> ()
    }
  }
  "test.user"(%t) : (!ktdf.token) -> ()
  return
}
