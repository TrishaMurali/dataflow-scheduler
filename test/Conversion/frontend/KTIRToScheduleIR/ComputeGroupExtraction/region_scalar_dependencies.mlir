// RUN: dataflow-scheduler-opt --compute-group-extraction %s | FileCheck %s

// The compute reads two scalars inside its body rather than through its
// operands, so the extracted function takes them as parameters and the call
// passes them on.

// CHECK-LABEL:   func.func @scalar_in_body(
// CHECK-SAME:        %[[BASE:.*]]: index,
// CHECK-SAME:        %[[STRIDE:.*]]: index) {
// CHECK:           call @local_schedule_0(%[[BASE]], %[[STRIDE]]) : (index, index) -> ()
// CHECK:         func.func private @local_schedule_0(index, index)

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-LABEL:     func.func @local_schedule_0(
// CHECK-SAME:          %[[BASE:.*]]: index,
// CHECK-SAME:          %[[STRIDE:.*]]: index) {
// CHECK:             linalg.generic
// CHECK:               %[[B:.*]] = arith.index_castui %[[BASE]] : index to i32
// CHECK:               %[[S:.*]] = arith.index_castui %[[STRIDE]] : index to i32
// CHECK:               spyreop.idx32toaddr %{{.*}} base %[[B]] stride %[[S]]

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>

module {
  func.func @scalar_in_body(%base: index, %stride: index) {
    %c0 = arith.constant 0 : index
    %in_addr = arith.constant 1024 : index
    %out_addr = arith.constant 4096 : index

    %in_view = ktdp.construct_memory_view %in_addr, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x32xi32>
    %out_view = ktdp.construct_memory_view %out_addr, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x32xi32>

    %in_tile = ktdp.construct_access_tile %in_view[%c0, %c0]
        {access_tile_order = #map, access_tile_set = #set}
        : memref<2x32xi32> -> !ktdp.access_tile<2x32xindex>
    %out_tile = ktdp.construct_access_tile %out_view[%c0, %c0]
        {access_tile_order = #map, access_tile_set = #set}
        : memref<2x32xi32> -> !ktdp.access_tile<2x32xindex>

    %idx = ktdp.load %in_tile : !ktdp.access_tile<2x32xindex> -> tensor<2x32xi32>
    %init = tensor.empty() : tensor<2x32xi32>
    %addr = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%idx : tensor<2x32xi32>) outs(%init : tensor<2x32xi32>) {
    ^bb0(%in: i32, %out: i32):
      %b = arith.index_castui %base : index to i32
      %s = arith.index_castui %stride : index to i32
      %a = spyreop.idx32toaddr %in base %b stride %s
      linalg.yield %a : i32
    } -> tensor<2x32xi32>
    ktdp.store %addr, %out_tile : tensor<2x32xi32>, !ktdp.access_tile<2x32xindex>
    return
  }
}
