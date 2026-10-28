// RUN: dataflow-scheduler-opt --simplify-linalg %s | FileCheck %s

// Tests that unit dimensions of `linalg.generic` ops are dropped, preserving
// discardable attributes.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @unit_dim
// CHECK-SAME:    %[[A:.+]]: tensor<1x8xf32>
// CHECK:         %[[C:.+]] = tensor.collapse_shape %[[A]]
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel"]
// CHECK-SAME:      ins(%[[C]] : tensor<8xf32>)
// CHECK-SAME:      ktdf_arch.maps_to = "SFU"} {
// CHECK:         }
// CHECK:         %[[E:.+]] = tensor.expand_shape %[[R]]
// CHECK:         return %[[E]]
func.func @unit_dim(%a: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %e = tensor.empty() : tensor<1x8xf32>
  %0 = linalg.generic {indexing_maps = [#id2, #id2],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a : tensor<1x8xf32>) outs(%e : tensor<1x8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
