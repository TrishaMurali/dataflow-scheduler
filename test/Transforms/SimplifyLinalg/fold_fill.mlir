// RUN: dataflow-scheduler-opt %s \
// RUN:   --linalg-morph-ops="named-to-generic=true category-to-generic=true" \
// RUN:   --simplify-linalg | FileCheck %s

// Tests that generalized `linalg.fill` ops reduce to `arith.constant` whenever
// possible.

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @fill_f32
// CHECK-NOT:     linalg.generic
// CHECK:         %[[C:.+]] = arith.constant dense<1.000000e+00> : tensor<8xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[C]]
func.func @fill_f32() -> tensor<8xf32> {
  %c = arith.constant 1.0 : f32
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @fill_i32
// CHECK:         %[[C:.+]] = arith.constant dense<7> : tensor<4xi32>
// CHECK:         return %[[C]]
func.func @fill_i32() -> tensor<4xi32> {
  %c = arith.constant 7 : i32
  %e = tensor.empty() : tensor<4xi32>
  %0 = linalg.fill ins(%c : i32) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @fill_index
// CHECK:         %[[C:.+]] = arith.constant dense<3> : tensor<4xindex>
// CHECK:         return %[[C]]
func.func @fill_index() -> tensor<4xindex> {
  %c = arith.constant 3 : index
  %e = tensor.empty() : tensor<4xindex>
  %0 = linalg.fill ins(%c : index) outs(%e : tensor<4xindex>)
      -> tensor<4xindex>
  return %0 : tensor<4xindex>
}

// CHECK-LABEL: func.func @fill_rank0
// CHECK:         %[[C:.+]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK:         return %[[C]]
func.func @fill_rank0() -> tensor<f32> {
  %c = arith.constant 2.0 : f32
  %e = tensor.empty() : tensor<f32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// The fill overwrites the entire init, so its contents do not matter.
// CHECK-LABEL: func.func @fill_init
// CHECK:         %[[C:.+]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK:         return %[[C]]
func.func @fill_init(%init: tensor<8xf32>) -> tensor<8xf32> {
  %c = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%c : f32) outs(%init : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// The fill value only becomes constant through folding.
// CHECK-LABEL: func.func @fill_folded
// CHECK:         %[[C:.+]] = arith.constant dense<3.000000e+00> : tensor<8xf32>
// CHECK:         return %[[C]]
func.func @fill_folded() -> tensor<8xf32> {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.addf %a, %b : f32
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Dynamic shapes can not become constants.
// CHECK-LABEL: func.func @fill_dynamic
// CHECK:         %[[CST:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[E:.+]] = tensor.empty
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[CST]] : f32) outs(%[[E]] : tensor<?xf32>) {
// CHECK:         }
func.func @fill_dynamic(%n: index) -> tensor<?xf32> {
  %c = arith.constant 1.0 : f32
  %e = tensor.empty(%n) : tensor<?xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// Non-constant fill values can not become constants.
// CHECK-LABEL: func.func @fill_nonconstant
// CHECK-SAME:    %[[V:.+]]: f32
// CHECK:         %[[E:.+]] = tensor.empty
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[V]] : f32) outs(%[[E]] : tensor<8xf32>) {
// CHECK:         }
func.func @fill_nonconstant(%v: f32) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%v : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A fill from a constant element of a constant tensor becomes a constant.
// CHECK-LABEL: func.func @fill_tensor
// CHECK-NOT:     linalg.generic
// CHECK:         %[[C:.+]] = arith.constant dense<3.000000e+00> : tensor<8xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[C]]
func.func @fill_tensor() -> tensor<8xf32> {
  %t = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (2)>, #id],
                       iterator_types = ["parallel"]}
       ins(%t : tensor<4xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
