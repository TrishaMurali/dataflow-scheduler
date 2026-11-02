// RUN: dataflow-scheduler-opt --ktir-pipeline %s | FileCheck %s

// Tests that loads of stride 0 become broadcasting data transfers. The input is
// the output of KTIRMapAndTile/bcast_load.mlir after --ktir-bufferize, with its
// memory views replaced by arguments.

#id = affine_map<(d0) -> (d0)>
#bcast = affine_map<(d0) -> (0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// A load of stride 0 transfers one element into the full FIFO slot.
// CHECK-LABEL: func.func @broadcast
// CHECK-SAME:    %{{.+}}: memref<128xf16, "DDR">, %[[B:[^:]+]]: memref<1xf16, "DDR">
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.data_transfer from %[[B]][0] size [1] to %{{.+}} size [64]
// CHECK-SAME:          transfer_mode = #ktdf.transfer_mode<splat>
// CHECK:           }
// CHECK:         }
func.func @broadcast(%a: memref<128xf16, "DDR">, %b: memref<1xf16, "DDR">, %out: memref<128xf16, "DDR">) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    %a_tile = ktdp_lowering.load %a[%i] [64] [1] {dataflow_scheduler.throttle = 64 : i64} : memref<128xf16, "DDR"> -> tensor<64xf16>
    %init = tensor.empty() : tensor<64xf16>
    %b_tile = ktdp_lowering.load %b[0] [64] [0] {dataflow_scheduler.throttle = 64 : i64} : memref<1xf16, "DDR"> -> tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#id, #bcast, #id], iterator_types = ["parallel"]} ins(%a_tile, %b_tile : tensor<64xf16>, tensor<64xf16>) outs(%init : tensor<64xf16>) attrs = {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %s = arith.addf %x, %y : f16
      linalg.yield %s : f16
    } -> tensor<64xf16>
    ktdp_lowering.store %result into %out[%i] [64] [1] {dataflow_scheduler.throttle = 64 : i64} : tensor<64xf16> into memref<128xf16, "DDR">
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}

// A load of stride 0 through a via transfers one element into the FIFO slot
// to the via memory, which holds the full tile.
// CHECK-LABEL: func.func @broadcast_through_via
// CHECK-SAME:    %{{.+}}: memref<128xf16, "DDR">, %[[B:[^:]+]]: memref<1xf16, "DDR">
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.data_transfer from %[[B]][0] size [1] to %[[SLOT:.+]] size [64]
// CHECK-SAME:          transfer_mode = #ktdf.transfer_mode<splat>
// CHECK:           }
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.data_transfer from %[[SLOT]] size [64] to %[[VIA:.+]][0] size [64]
// CHECK:           }
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.data_transfer from %[[VIA]][0] size [64] to %{{.+}} size [64]
// CHECK:           }
// CHECK:         }
func.func @broadcast_through_via(%a: memref<128xf16, "DDR">, %b: memref<1xf16, "DDR">, %out: memref<128xf16, "DDR">) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    %a_tile = ktdp_lowering.load %a[%i] [64] [1] {dataflow_scheduler.throttle = 64 : i64} : memref<128xf16, "DDR"> -> tensor<64xf16>
    %init = tensor.empty() : tensor<64xf16>
    %b_tile = ktdp_lowering.load %b[0] [64] [0] {dataflow_scheduler.throttle = 64 : i64} : memref<1xf16, "DDR"> -> tensor<64xf16>
    %b_l1 = ktdf.via["L1"] %b_tile : tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#id, #bcast, #id], iterator_types = ["parallel"]} ins(%a_tile, %b_l1 : tensor<64xf16>, tensor<64xf16>) outs(%init : tensor<64xf16>) attrs = {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %s = arith.addf %x, %y : f16
      linalg.yield %s : f16
    } -> tensor<64xf16>
    ktdp_lowering.store %result into %out[%i] [64] [1] {dataflow_scheduler.throttle = 64 : i64} : tensor<64xf16> into memref<128xf16, "DDR">
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
