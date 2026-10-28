// RUN: dataflow-scheduler-opt --simplify-linalg %s | FileCheck %s

// Tests that the operands of a `linalg.generic` are shrunk to the slices that
// are actually accessed, and that strided accesses are normalized.

#id = affine_map<(d0) -> (d0)>

// CHECK-DAG: #[[$ID:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$WIN:.+]] = affine_map<(d0, d1) -> (d0 + d1)>

// A constant index accesses a hyperlane.
// CHECK-LABEL: func.func @constant
// CHECK-SAME:    %[[A:.+]]: tensor<8x16xf32>
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[A]][3, 0] [1, 16] [1, 1]
// CHECK:         %[[C:.+]] = tensor.collapse_shape %[[S]]
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME:      ins(%[[C]] : tensor<16xf32>)
func.func @constant(%a: tensor<8x16xf32>) -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (3, d0)>, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<8x16xf32>) outs(%e : tensor<16xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// An offset access is rebased onto the slice.
// CHECK-LABEL: func.func @offset
// CHECK-SAME:    %[[A:.+]]: tensor<16xf32>
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[A]][2] [8] [1]
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME:      ins(%[[S]] : tensor<8xf32>)
func.func @offset(%a: tensor<16xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 + 2)>, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<16xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A strided access is normalized to unit stride.
// CHECK-LABEL: func.func @stride
// CHECK-SAME:    %[[A:.+]]: tensor<15xf32>
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[A]][0] [8] [2]
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME:      ins(%[[S]] : tensor<8xf32>)
func.func @stride(%a: tensor<15xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 * 2)>, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<15xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Shrinking and stride normalization compose into a single slice.
// CHECK-LABEL: func.func @shrink_and_stride
// CHECK-SAME:    %[[A:.+]]: tensor<20xf32>
// CHECK-NOT:     tensor.extract_slice
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[A]][1] [8] [2]
// CHECK-NOT:     tensor.extract_slice
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME:      ins(%[[S]] : tensor<8xf32>)
func.func @shrink_and_stride(%a: tensor<20xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 * 2 + 1)>, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<20xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A sliding window accesses the union of all windows.
// CHECK-LABEL: func.func @window
// CHECK-SAME:    %[[A:.+]]: tensor<32xf32>, %[[W:.+]]: tensor<3xf32>
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[A]][0] [10] [1]
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$WIN]],
// CHECK-SAME:      ins(%[[S]], %[[W]] : tensor<10xf32>, tensor<3xf32>)
func.func @window(%a: tensor<32xf32>, %w: tensor<3xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>,
                                        affine_map<(d0, d1) -> (d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
                       iterator_types = ["parallel", "reduction"]}
       ins(%a, %w : tensor<32xf32>, tensor<3xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %m, %o : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// An init is shrunk, and the result is inserted back into it.
// CHECK-LABEL: func.func @init
// CHECK-SAME:    %[[A:.+]]: tensor<8xf32>, %[[INIT:.+]]: tensor<16xf32>
// CHECK:         %[[S:.+]] = tensor.extract_slice %[[INIT]][4] [8] [1]
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME:      ins(%[[A]] : tensor<8xf32>) outs(%[[S]] : tensor<8xf32>)
// CHECK:         %[[I:.+]] = tensor.insert_slice %[[R]] into %[[INIT]][4] [8] [1]
// CHECK:         return %[[I]]
func.func @init(%a: tensor<8xf32>, %init: tensor<16xf32>) -> tensor<16xf32> {
  %0 = linalg.generic {indexing_maps = [#id, affine_map<(d0) -> (d0 + 4)>],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<8xf32>) outs(%init : tensor<16xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// Operands that are accessed entirely are left alone.
// CHECK-LABEL: func.func @tight
// CHECK-NOT:     tensor.extract_slice
// CHECK:         linalg.generic
func.func @tight(%a: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A rebased access must not become the source of an unknown loop range.
// CHECK-LABEL: func.func @dynamic_range
// CHECK-NOT:     tensor.extract_slice
// CHECK:         linalg.generic
func.func @dynamic_range(%a: tensor<16xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 + 2)>, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<16xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// Ops with buffer semantics are ignored.
// CHECK-LABEL: func.func @buffer
// CHECK-NOT:     memref.subview
// CHECK:         linalg.generic
func.func @buffer(%a: memref<16xf32>, %b: memref<8xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 + 2)>, #id],
                  iterator_types = ["parallel"]}
      ins(%a : memref<16xf32>) outs(%b : memref<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  }
  return
}
