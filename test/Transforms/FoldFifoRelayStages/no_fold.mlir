// A relay is only folded when it can be without breaking what path expansion
// expects of the stages. A relay that cannot keeps both stages as they are.

// RUN: dataflow-scheduler-opt -fold-fifo-relay-stages %s | FileCheck %s

// Different throttles on the two halves of a pair.

// CHECK-LABEL:   func.func @conflicting_throttle(
// CHECK-COUNT-2:   ktdf.data_transfer from {{.*}} !ktdf.fifo.slot
// CHECK-NOT:       ktdf.data_transfer from {{.*}} : memref<{{.*}}>, memref<

func.func @conflicting_throttle(%a: memref<1x64xf16, "DDR">, %b: memref<1x32xf16, "DDR">) {
  ktdf.pipeline {
    %p:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, !ktdf.token, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>
      %t = ktdf.create_token : !ktdf.token
      %l1a = memref.alloc() : memref<1x64xf16, "L1">
      %l1b = memref.alloc() : memref<1x32xf16, "L1">
      ktdf.private_yield %slots#0, %slots#1, %t, %l1a, %l1b : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, !ktdf.token, memref<1x64xf16, "L1">, memref<1x32xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      ktdf.data_transfer from %b[0, 0] size [1, 32] to %p#1 size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<1x32xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>
    }
    ktdf.stage depends_in(%p#2) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#3[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
      ktdf.data_transfer from %p#1 size [1, 32] to %p#4[0, 0] size [1, 32] {dataflow_scheduler.throttle = 16 : i64} : !ktdf.fifo.slot<"DDR" -> "L1", 32xf16>, memref<1x32xf16, "L1">
    }
  }
  return
}

// The relay stays within one memory, where a destination may alias a source.

// CHECK-LABEL:   func.func @same_memory(
// CHECK-COUNT-2:   ktdf.data_transfer from {{.*}} !ktdf.fifo.slot

func.func @same_memory(%a: memref<1x64xf16, "L1">) {
  ktdf.pipeline {
    %p:2 = ktdf.private -> (!ktdf.fifo.slot<"L1" -> "L1", 64xf16>, memref<1x64xf16, "L1">) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "L1", 64xf16>
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.private_yield %slot, %l1 : !ktdf.fifo.slot<"L1" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "L1", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#1[0, 0] size [1, 64] : !ktdf.fifo.slot<"L1" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
  }
  return
}

// The stages cannot run on a common unit.

// CHECK-LABEL:   func.func @disjoint_units(
// CHECK-COUNT-2:   ktdf.data_transfer from {{.*}} !ktdf.fifo.slot

func.func @disjoint_units(%a: memref<1x64xf16, "DDR">) {
  ktdf.pipeline {
    %p:2 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.private_yield %slot, %l1 : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#1[0, 0] size [1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    } {applicable_units = ["L1LU"]}
  }
  return
}

// Folding into the producer leaves its transfers with no shared end. Folding
// into the consumer makes it read from DDR, which another stage already does.

// CHECK-LABEL:   func.func @no_anchor(
// CHECK-COUNT-3:   ktdf.data_transfer from {{.*}} !ktdf.fifo.slot

func.func @no_anchor(%a: memref<1x64xf16, "DDR">, %d: memref<1x64xf16, "L2">, %z: memref<1x64xf16, "DDR">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>, memref<1x64xf16, "L1">, memref<1x64xf16, "L3">) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      %other = memref.alloc() : memref<1x64xf16, "L3">
      ktdf.private_yield %slots#0, %slots#1, %l1, %other : !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>, memref<1x64xf16, "L1">, memref<1x64xf16, "L3">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>
      ktdf.data_transfer from %d[0, 0] size [1, 64] to %p#1 size [1, 64] : memref<1x64xf16, "L2">, !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#2[0, 0] size [1, 64] : !ktdf.fifo.slot<"MNILU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %z[0, 0] size [1, 64] to %p#3[0, 0] size [1, 64] : memref<1x64xf16, "DDR">, memref<1x64xf16, "L3">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %x = ktdf.read_from_fifo %p#1 : <"MNILU" -> "SFU", 64xf16> -> tensor<1x64xf16>
    } {applicable_units = ["SFU"]}
  }
  return
}

// Folding into the producer makes it await a stage that awaits it. Folding
// into the consumer moves the read of the source past another stage that
// awaits the producer.

// CHECK-LABEL:   func.func @cycle(
// CHECK-COUNT-2:   ktdf.data_transfer from {{.*}} !ktdf.fifo.slot

func.func @cycle(%a: memref<1x64xf16, "DDR">) {
  %cst = arith.constant 0.0 : f16
  ktdf.pipeline {
    %p:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">, memref<1x64xf16, "L1">) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %l1 = memref.alloc() : memref<1x64xf16, "L1">
      %other = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.private_yield %slot, %t0, %t1, %l1, %other : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">, memref<1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(%p#1) {
      ktdf.data_transfer from %a[0, 0] size [1, 64] to %p#0 size [1, 64] : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    }
    ktdf.stage depends_in(%p#1) depends_out(%p#2) {
      linalg.fill ins(%cst : f16) outs(%p#4 : memref<1x64xf16, "L1">)
    }
    ktdf.stage depends_in(%p#1, %p#2) depends_out(none) {
      ktdf.data_transfer from %p#0 size [1, 64] to %p#3[0, 0] size [1, 64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
  }
  return
}
