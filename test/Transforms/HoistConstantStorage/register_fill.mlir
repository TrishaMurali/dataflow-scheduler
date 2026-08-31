// A register filled with a runtime scalar, once. The fill is loop invariant and
// the register is a memref.alloc, so both go outside the loop: filled per
// iteration the value would become an immediate, which a symbol has none of
// until the symbols are resolved.
//
// An alloca is left where it is even when its fill is invariant. Hoisting one
// over a parallel loop would raise the question of what aliases what, which an
// alloc does not: a resource allocated that way may be shared.
//
// A fill that varies with the loop stays too, allocation and all.

// RUN: dataflow-scheduler-opt -hoist-constant-storage %s | FileCheck %s

// CHECK-LABEL:   func.func @register_filled_from_a_scalar(
// CHECK-SAME:        %[[SRC:.*]]: memref<32xf32, #ktdp.memory_space<global>>,
// CHECK-SAME:        %[[SCALE:.*]]: f32) {
// CHECK:           %[[REG:.*]] = memref.alloc() : memref<32xf32, "SFU_REG">
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage
// CHECK-NEXT:          linalg.fill ins(%[[SCALE]] : f32) outs(%[[REG]] : memref<32xf32, "SFU_REG">)
// CHECK:           scf.for
// CHECK-NOT:         linalg.fill
// CHECK:             ktdf.data_transfer from %[[REG]]

// CHECK-LABEL:   func.func @register_filled_per_iteration(
// CHECK:           scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             ktdf.pipeline {
// CHECK:               ktdf.stage
// CHECK:                 %[[REG:.*]] = memref.alloc() : memref<32xf32, "SFU_REG">
// CHECK-NEXT:            linalg.fill ins(%{{.*}} : f32) outs(%[[REG]] : memref<32xf32, "SFU_REG">)

func.func @register_filled_from_a_scalar(%src: memref<32xf32, #ktdp.memory_space<global>>, %scale: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %token = ktdf.private -> (!ktdf.token) {
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %t : !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%token) {
        %reg = memref.alloc() : memref<32xf32, "SFU_REG">
        linalg.fill ins(%scale : f32) outs(%reg : memref<32xf32, "SFU_REG">)
        ktdf.data_transfer from %reg[%c0] size [32] to %src[%c0] size [32] : memref<32xf32, "SFU_REG">, memref<32xf32, #ktdp.memory_space<global>>
      }
    }
  }
  return
}

func.func @register_filled_per_iteration(%src: memref<32xf32, #ktdp.memory_space<global>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %token = ktdf.private -> (!ktdf.token) {
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %t : !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%token) {
        %reg = memref.alloc() : memref<32xf32, "SFU_REG">
        %narrow = arith.index_castui %i : index to i32
        %wide = arith.uitofp %narrow : i32 to f32
        linalg.fill ins(%wide : f32) outs(%reg : memref<32xf32, "SFU_REG">)
        ktdf.data_transfer from %reg[%c0] size [32] to %src[%c0] size [32] : memref<32xf32, "SFU_REG">, memref<32xf32, #ktdp.memory_space<global>>
      }
    }
  }
  return
}

// CHECK-LABEL:   func.func @register_alloca_stays(
// CHECK:           scf.for
// CHECK:             ktdf.pipeline {
// CHECK:               ktdf.stage
// CHECK:                 %[[REG:.*]] = memref.alloca() : memref<32xf32, "SFU_REG">
// CHECK-NEXT:            linalg.fill ins(%{{.*}} : f32) outs(%[[REG]] : memref<32xf32, "SFU_REG">)

func.func @register_alloca_stays(%src: memref<32xf32, #ktdp.memory_space<global>>, %scale: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %token = ktdf.private -> (!ktdf.token) {
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %t : !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%token) {
        %reg = memref.alloca() : memref<32xf32, "SFU_REG">
        linalg.fill ins(%scale : f32) outs(%reg : memref<32xf32, "SFU_REG">)
        ktdf.data_transfer from %reg[%c0] size [32] to %src[%c0] size [32] : memref<32xf32, "SFU_REG">, memref<32xf32, #ktdp.memory_space<global>>
      }
    }
  }
  return
}
