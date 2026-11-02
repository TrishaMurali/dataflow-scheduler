// A transfer from memory into a FIFO slot and the transfer out of it into
// memory are folded into one memory-to-memory transfer. If both stages would
// be left empty, it goes into the producer, and the emptied consumer is erased.
// The slots and tokens left without users are dropped.

// RUN: dataflow-scheduler-opt -fold-fifo-relay-stages %s | FileCheck %s

// CHECK-LABEL:   func.func @single_relay(
// CHECK-SAME:        %[[DDR:.*]]: memref<1x64xf16, "DDR">) {
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        %[[P:.*]]:2 = ktdf.private -> (!ktdf.token, memref<1x64xf16, "L1">) {
// CHECK-NOT:           ktdf.fifo.allocate
// CHECK:             }
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(%[[P]]#0) {
// CHECK-NEXT:          ktdf.data_transfer from %[[DDR]][0, 0] size [1, 64] to %[[P]]#1[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:        }
// CHECK-NEXT:        ktdf.stage depends_in(%[[P]]#0) depends_out(none) {
// CHECK-NEXT:          linalg.fill
// CHECK-NEXT:        }
// CHECK-NEXT:      }

func.func @single_relay(%ddr: memref<1x64xf16, "DDR">) {
  %cst = arith.constant 0.0 : f16
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.private_yield %slot, %t0, %t1, %l1 : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(%p#1) {
      ktdf.data_transfer from %ddr[0, 0] size [1, 64] to %p#0 size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    }
    ktdf.stage depends_in(%p#1) depends_out(%p#2) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#3[0, 0] size [1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(%p#2) depends_out(none) {
      linalg.fill ins(%cst : f16) outs(%p#3 : memref<1x64xf16, "L1">)
    }
  }
  return
}

// Several relays between the same memories fold into the producer, in its
// order, onto the units both stages can run on. A slot sharing their
// allocation that is not relayed stays, and the folded stage keeps the token
// the consumer awaits from another stage, so it moves after that stage.

// CHECK-LABEL:   func.func @multiple_relays(
// CHECK-SAME:        %[[A:[^:]*]]: memref<1x64xf16, "DDR">,
// CHECK-SAME:        %[[B:[^:]*]]: memref<1x32xf16, "DDR">,
// CHECK-SAME:        %[[C:[^:]*]]: memref<1x16xf16, "DDR">) {
// CHECK:             ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 16xf16>{{$}}
// CHECK-NOT:         ktdf.fifo.allocate
// CHECK:           ktdf.stage depends_in(none) depends_out(%[[OTHER:.*]]) {
// CHECK-NEXT:        ktdf.data_transfer from %[[C]]
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage depends_in(%[[OTHER]]) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[A]][0, 0] size [1, 64] to %{{.*}}[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:        ktdf.data_transfer from %[[B]][0, 0] size [1, 32] to %{{.*}}[0, 0] size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<1x32xf16, "DDR">, memref<1x32xf16, "L1">
// CHECK-NEXT:      } {applicable_units = ["MNILU"]}
// CHECK-NEXT:      ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}} size [1, 16]

func.func @multiple_relays(%a: memref<1x64xf16, "DDR">, %b: memref<1x32xf16, "DDR">, %c: memref<1x16xf16, "DDR">) {
  %cst = arith.constant 0.0 : f16
  ktdf.pipeline {
    %p:8 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 16xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">, memref<1x16xf16, "L1">) {
      %slots:3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 16xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %l1a = memref.alloc() : memref<1x64xf16, "L1">
      %l1b = memref.alloc() : memref<1x32xf16, "L1">
      %l1c = memref.alloc() : memref<1x16xf16, "L1">
      ktdf.private_yield %slots#0, %slots#1, %slots#2, %t0, %t1, %l1a, %l1b, %l1c : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 16xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">, memref<1x16xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(%p#3) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      ktdf.data_transfer from %b[0, 0] size [1, 32] to %p#1 size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<1x32xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>
    } {applicable_units = ["MNILU", "L1LU"]}
    ktdf.stage depends_in(none) depends_out(%p#4) {
      ktdf.data_transfer from %c[0, 0] size [1, 16] to %p#2 size [1, 16] : memref<1x16xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 16xf16>
    }
    ktdf.stage depends_in(%p#3, %p#4) depends_out(none) {
      ktdf.data_transfer from %p#1 size [1, 32] to %p#6[0, 0] size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, memref<1x32xf16, "L1">
      ktdf.data_transfer from %p#0 size [1, 64] to %p#5[0, 0] size [1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#2 size [1, 16] to %p#7[0, 0] size [1, 16] : !ktdf.fifo.slot<"DDR" -> "L1", 16xf16>, memref<1x16xf16, "L1">
      linalg.fill ins(%cst : f16) outs(%p#7 : memref<1x16xf16, "L1">)
    }
  }
  return
}
