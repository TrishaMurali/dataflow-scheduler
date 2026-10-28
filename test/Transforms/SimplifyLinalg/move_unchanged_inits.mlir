// RUN: dataflow-scheduler-opt --simplify-linalg %s | FileCheck %s

// Tests that inits of a `linalg.generic` which are yielded unchanged are moved
// to the inputs.

#id = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// The moved init's result is dead afterwards, and is erased along with the
// empty init that replaced it.
// CHECK-LABEL: func.func @parallel
// CHECK-SAME:    %[[A:.+]]: tensor<8xf32>, %[[B:.+]]: tensor<8xf32>, %[[C:.+]]: tensor<8xf32>
// CHECK-NOT:     tensor.empty
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<8xf32>, tensor<8xf32>)
// CHECK-SAME:      outs(%[[C]] : tensor<8xf32>) {
// CHECK:         ^bb0(%[[X:.+]]: f32, %[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK:           %[[S:.+]] = arith.addf %[[X]], %[[Y]]
// CHECK:           linalg.yield %[[S]]
// CHECK:         }
// CHECK:         return %[[B]], %[[R]]
func.func @parallel(%a: tensor<8xf32>, %b: tensor<8xf32>, %c: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>) {
  %0:2 = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel"]}
         ins(%a : tensor<8xf32>) outs(%b, %c : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %y, %s : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// An unchanged init in a reduction is read-only.
// CHECK-LABEL: func.func @reduction
// CHECK-SAME:    %[[A:.+]]: tensor<8x4xf32>, %[[B:.+]]: tensor<8xf32>, %[[C:.+]]: tensor<8xf32>
// CHECK-NOT:     tensor.empty
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<8x4xf32>, tensor<8xf32>)
// CHECK-SAME:      outs(%[[C]] : tensor<8xf32>) {
// CHECK:         ^bb0(%[[X:.+]]: f32, %[[Y:.+]]: f32, %[[O:.+]]: f32):
// CHECK:           %[[M:.+]] = arith.mulf %[[X]], %[[Y]]
// CHECK:           %[[S:.+]] = arith.addf %[[M]], %[[O]]
// CHECK:           linalg.yield %[[S]]
// CHECK:         }
// CHECK:         return %[[R]]
func.func @reduction(%a: tensor<8x4xf32>, %b: tensor<8xf32>, %c: tensor<8xf32>)
    -> tensor<8xf32> {
  %0:2 = linalg.generic {indexing_maps = [#id2, #row, #row],
                         iterator_types = ["parallel", "reduction"]}
         ins(%a : tensor<8x4xf32>) outs(%b, %c : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %m, %o : f32
    linalg.yield %y, %s : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  return %0#1 : tensor<8xf32>
}

// Updated inits are left alone.
// CHECK-LABEL: func.func @updated
// CHECK-SAME:    %[[A:.+]]: tensor<8x4xf32>, %[[C:.+]]: tensor<8xf32>
// CHECK-NOT:     tensor.empty
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[A]] : tensor<8x4xf32>) outs(%[[C]] : tensor<8xf32>)
func.func @updated(%a: tensor<8x4xf32>, %c: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {indexing_maps = [#id2, #row],
                       iterator_types = ["parallel", "reduction"]}
       ins(%a : tensor<8x4xf32>) outs(%c : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = arith.addf %x, %o : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
