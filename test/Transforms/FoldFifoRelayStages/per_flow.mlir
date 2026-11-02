// Each relay folds on its own, into the stage that keeps the transfers of each
// stage sharing an end that no other stage has. The relays of one stage may
// go to different stages and move between different memories.

// RUN: dataflow-scheduler-opt -fold-fifo-relay-stages %s | FileCheck %s

// The relay into L0 folds into the producer, which keeps reading from global.
// Folding into the consumer would give it a second stage reading from global.
// The emptied consumer is erased, and the stage that awaited it awaits the
// producer instead.

// CHECK-LABEL:   func.func @fold_into_producer(
// CHECK-SAME:        %[[G:[^:]*]]: memref<1x64xf16, "global">,
// CHECK-SAME:        %[[H:[^:]*]]: memref<1x64xf16, "global">,
// CHECK:             %[[P:.*]]:7 = ktdf.private
// CHECK:           ktdf.stage depends_in(none) depends_out(%[[P]]#3) {
// CHECK-NEXT:        ktdf.data_transfer from %[[G]][0, 0] size [1, 64] to %[[P]]#6[0, 0] size [1, 64] : memref<1x64xf16, "global">, memref<1x64xf16, "L0">
// CHECK-NEXT:        ktdf.data_transfer from %[[H]][0, 0] size [1, 64] to %[[P]]#0 size [1, 64]
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(%[[P]]#3) depends_out(%[[P]]#4) {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#6[0, 0] size [1, 64] to %[[P]]#1 size [1, 64]
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(%[[P]]#3, %[[P]]#4) depends_out(%[[P]]#5) {
// CHECK:           ktdf.stage depends_in(%[[P]]#5) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#2
// CHECK-NEXT:      }
// CHECK-NEXT:    }

func.func @fold_into_producer(%g: memref<1x64xf16, "global">, %h: memref<1x64xf16, "global">, %out: memref<1x64xf16, "global">) {
  ktdf.pipeline {
    %p:9 = ktdf.private -> (!ktdf.fifo.slot<"global" -> "L0", 64xf16>, !ktdf.fifo.slot<"global" -> "PT", 64xf16>, !ktdf.fifo.slot<"L0" -> "PT", 64xf16>, !ktdf.fifo.slot<"PT" -> "global", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, memref<1x64xf16, "L0">) {
      %slots:4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"global" -> "L0", 64xf16>, !ktdf.fifo.slot<"global" -> "PT", 64xf16>, !ktdf.fifo.slot<"L0" -> "PT", 64xf16>, !ktdf.fifo.slot<"PT" -> "global", 64xf16>
      %t1 = ktdf.create_token : !ktdf.token
      %t2 = ktdf.create_token : !ktdf.token
      %t3 = ktdf.create_token : !ktdf.token
      %t4 = ktdf.create_token : !ktdf.token
      %l0 = memref.alloc() : memref<1x64xf16, "L0">
      ktdf.private_yield %slots#0, %slots#1, %slots#2, %slots#3, %t1, %t2, %t3, %t4, %l0 : !ktdf.fifo.slot<"global" -> "L0", 64xf16>, !ktdf.fifo.slot<"global" -> "PT", 64xf16>, !ktdf.fifo.slot<"L0" -> "PT", 64xf16>, !ktdf.fifo.slot<"PT" -> "global", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, memref<1x64xf16, "L0">
    }
    ktdf.stage depends_in(none) depends_out(%p#4) {
      ktdf.data_transfer from %g[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "global">, !ktdf.fifo.slot<"global" -> "L0", 64xf16>
      ktdf.data_transfer from %h[0, 0] size [1, 64] to %p#1 size [1, 64] : memref<1x64xf16, "global">, !ktdf.fifo.slot<"global" -> "PT", 64xf16>
    }
    ktdf.stage depends_in(%p#4) depends_out(%p#5) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#8[0, 0] size [1, 64] : !ktdf.fifo.slot<"global" -> "L0", 64xf16>, memref<1x64xf16, "L0">
    }
    ktdf.stage depends_in(%p#5) depends_out(%p#6) {
      ktdf.data_transfer from %p#8[0, 0] size [1, 64] to %p#2 size [1, 64] : memref<1x64xf16, "L0">, !ktdf.fifo.slot<"L0" -> "PT", 64xf16>
    }
    ktdf.stage depends_in(%p#4, %p#6) depends_out(%p#7) {
      %a = ktdf.read_from_fifo %p#2 : <"L0" -> "PT", 64xf16> -> tensor<1x64xf16>
      %b = ktdf.read_from_fifo %p#1 : <"global" -> "PT", 64xf16> -> tensor<1x64xf16>
      ktdf.write_to_fifo %a, %p#3 : tensor<1x64xf16>, <"PT" -> "global", 64xf16>
    } {applicable_units = ["PT"]}
    ktdf.stage depends_in(%p#7) depends_out(none) {
      ktdf.data_transfer from %p#3 size [1, 64] to %out[0, 0] size [1, 64] : !ktdf.fifo.slot<"PT" -> "global", 64xf16>, memref<1x64xf16, "global">
    }
  }
  return
}

// Folding into the producer would leave its transfers with no shared end, so
// the relay folds into the consumer.

// CHECK-LABEL:   func.func @fold_into_consumer(
// CHECK-SAME:        %[[A:[^:]*]]: memref<1x64xf16, "DDR">,
// CHECK-SAME:        %[[D:[^:]*]]: memref<1x64xf16, "L2">) {
// CHECK:           ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[D]]{{.*}} : memref<1x64xf16, "L2">, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[A]][0, 0] size [1, 64] to %{{.*}}[0, 0] size [1, 64] : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        ktdf.read_from_fifo

func.func @fold_into_consumer(%a: memref<1x64xf16, "DDR">, %d: memref<1x64xf16, "L2">) {
  ktdf.pipeline {
    %p:3 = ktdf.private -> (!ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>, memref<1x64xf16, "L1">) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.private_yield %slots#0, %slots#1, %l1 : !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>
      ktdf.data_transfer from %d[0, 0] size [1, 64] to %p#1 size [1, 64] : memref<1x64xf16, "L2">, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#2[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %x = ktdf.read_from_fifo %p#1 : <"MNILU" -> "SFU", 64xf16> -> tensor<1x64xf16>
    } {applicable_units = ["SFU"]}
  }
  return
}

// The relays of one producer go to different consumers. Each folds into the
// producer, which erases the consumers.

// CHECK-LABEL:   func.func @split_consumers(
// CHECK-SAME:        %[[A:[^:]*]]: memref<1x64xf16, "DDR">,
// CHECK-SAME:        %[[B:[^:]*]]: memref<1x32xf16, "DDR">) {
// CHECK-NOT:         ktdf.fifo.allocate
// CHECK:           ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[A]][0, 0] size [1, 64] to %{{.*}}[0, 0] size [1, 64] : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:        ktdf.data_transfer from %[[B]][0, 0] size [1, 32] to %{{.*}}[0, 0] size [1, 32] : memref<1x32xf16, "DDR">, memref<1x32xf16, "L1">
// CHECK-NEXT:      }
// CHECK-NEXT:    }

func.func @split_consumers(%a: memref<1x64xf16, "DDR">, %b: memref<1x32xf16, "DDR">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>
      %l1a = memref.alloc() : memref<1x64xf16, "L1">
      %l1b = memref.alloc() : memref<1x32xf16, "L1">
      ktdf.private_yield %slots#0, %slots#1, %l1a, %l1b : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      ktdf.data_transfer from %b[0, 0] size [1, 32] to %p#1 size [1, 32] : memref<1x32xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#2[0, 0] size [1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#1 size [1, 32] to %p#3[0, 0] size [1, 32] : !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, memref<1x32xf16, "L1">
    }
  }
  return
}

// The relays move between different memories. Folding the first into the
// producer would leave it with no shared end, so it folds into the consumer,
// which keeps writing to L1. The second then folds into the consumer too, which
// erases the producer.

// CHECK-LABEL:   func.func @mixed_end_points(
// CHECK-SAME:        %[[A:[^:]*]]: memref<1x64xf16, "DDR">,
// CHECK-SAME:        %[[B:[^:]*]]: memref<1x32xf16, "L2">) {
// CHECK-NOT:         ktdf.fifo.allocate
// CHECK:           ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[A]][0, 0] size [1, 64] to %{{.*}}[0, 0] size [1, 64] : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:        ktdf.data_transfer from %[[B]][0, 0] size [1, 32] to %{{.*}}[0, 0] size [1, 32] : memref<1x32xf16, "L2">, memref<1x32xf16, "L1">
// CHECK-NEXT:      }
// CHECK-NEXT:    }

func.func @mixed_end_points(%a: memref<1x64xf16, "DDR">, %b: memref<1x32xf16, "L2">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "L1LU", 32xf16>, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "L1LU", 32xf16>
      %l1a = memref.alloc() : memref<1x64xf16, "L1">
      %l1b = memref.alloc() : memref<1x32xf16, "L1">
      ktdf.private_yield %slots#0, %slots#1, %l1a, %l1b : !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "L1LU", 32xf16>, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>
      ktdf.data_transfer from %b[0, 0] size [1, 32] to %p#1 size [1, 32] : memref<1x32xf16, "L2">, !ktdf.fifo.slot<"MNILU" -> "L1LU", 32xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#2[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
      ktdf.data_transfer from %p#1 size [1, 32] to %p#3[0, 0] size [1, 32] : !ktdf.fifo.slot<"MNILU" -> "L1LU", 32xf16>, memref<1x32xf16, "L1">
    }
  }
  return
}

// The token of the producer is also awaited by another stage, so the relay
// cannot move its read into the consumer. It folds into the producer.

// CHECK-LABEL:   func.func @shared_token(
// CHECK-SAME:        %[[A:[^:]*]]: memref<1x64xf16, "DDR">) {
// CHECK:             %[[P:.*]]:3 = ktdf.private
// CHECK:           ktdf.stage depends_in(none) depends_out(%[[P]]#0) {
// CHECK-NEXT:        ktdf.data_transfer from %[[A]][0, 0] size [1, 64] to %[[P]]#1[0, 0] size [1, 64] : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(%[[P]]#0) depends_out(none) {
// CHECK-NEXT:        linalg.fill
// CHECK-NEXT:      }
// CHECK-NEXT:    }

func.func @shared_token(%a: memref<1x64xf16, "DDR">) {
  %cst = arith.constant 0.0 : f16
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, memref<1x64xf16, "L1">, memref<1x64xf16, "L1">) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      %t = ktdf.create_token : !ktdf.token
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      %other = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.private_yield %slot, %t, %l1, %other : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, memref<1x64xf16, "L1">, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(%p#1) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    }
    ktdf.stage depends_in(%p#1) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#2[0, 0] size [1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(%p#1) depends_out(none) {
      linalg.fill ins(%cst : f16) outs(%p#3 : memref<1x64xf16, "L1">)
    }
  }
  return
}
