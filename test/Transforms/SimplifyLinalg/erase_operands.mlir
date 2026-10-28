// RUN: dataflow-scheduler-opt --simplify-linalg %s | FileCheck %s

// Tests that unnecessary operands and results of `linalg.generic` ops are
// removed, preserving discardable attributes.

#id = affine_map<(d0) -> (d0)>

// A dead result is removed along with its init and payload.
// CHECK-LABEL: func.func @dead_result
// CHECK-SAME:    %[[A:.+]]: tensor<8xf32>
// CHECK:         %[[E:.+]] = tensor.empty()
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[A]] : tensor<8xf32>) outs(%[[E]] : tensor<8xf32>)
// CHECK-SAME:      ktdf_arch.maps_to = "SFU"} {
// CHECK:           %[[N:.+]] = arith.negf
// CHECK-NOT:       arith.mulf
// CHECK:           linalg.yield %[[N]]
// CHECK:         }
// CHECK:         return %[[R]]
func.func @dead_result(%a: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0:2 = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel"]}
         ins(%a : tensor<8xf32>) outs(%e, %e : tensor<8xf32>, tensor<8xf32>)
         attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o0: f32, %o1: f32):
    %n = arith.negf %x : f32
    %m = arith.mulf %x, %x : f32
    linalg.yield %n, %m : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8xf32>
}

// Duplicate inputs are merged into one.
// CHECK-LABEL: func.func @duplicate_inputs
// CHECK-SAME:    %[[A:.+]]: tensor<8xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[A]] : tensor<8xf32>)
// CHECK-SAME:      ktdf_arch.maps_to = "SFU"} {
// CHECK:         ^bb0(%[[X:.+]]: f32, %{{.+}}: f32):
// CHECK:           arith.mulf %[[X]], %[[X]]
// CHECK:         }
func.func @duplicate_inputs(%a: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %a : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A copy that is shrunk to the accessed slice is an identity, and is erased.
// CHECK-LABEL: func.func @shrunk_copy
// CHECK-SAME:    %[[A:.+]]: tensor<16xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[A]][2] [8] [1]
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[S]]
func.func @shrunk_copy(%a: tensor<16xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 + 2)>, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<16xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
