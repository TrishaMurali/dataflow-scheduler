// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s
#map = affine_map<(d0) -> (d0)>
// A kernel holds a runtime scalar as index and casts it in the body where it is
// used. The cast is legal wherever it stands: below it is the base and the stride
// of an address computation, and on its own in the second function.
// CHECK-LABEL: func.func @index_cast_for_address
func.func @index_cast_for_address(%idx: tensor<4xi32>, %o: tensor<4xi32>,
                                  %base: index, %stride: index) -> tensor<4xi32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%idx : tensor<4xi32>) outs(%o : tensor<4xi32>) {
  ^bb0(%in: i32, %out: i32):
    %b = arith.index_castui %base : index to i32
    %s = arith.index_castui %stride : index to i32
    %a = spyreop.idx32toaddr %in base %b stride %s
    linalg.yield %a : i32
  } -> tensor<4xi32>
  return %r : tensor<4xi32>
}

// CHECK-LABEL: func.func @index_cast_alone
func.func @index_cast_alone(%idx: tensor<4xi32>, %o: tensor<4xi32>,
                            %scalar: index) -> tensor<4xi32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%idx : tensor<4xi32>) outs(%o : tensor<4xi32>) {
  ^bb0(%in: i32, %out: i32):
    %c = arith.index_castui %scalar : index to i32
    linalg.yield %c : i32
  } -> tensor<4xi32>
  return %r : tensor<4xi32>
}
