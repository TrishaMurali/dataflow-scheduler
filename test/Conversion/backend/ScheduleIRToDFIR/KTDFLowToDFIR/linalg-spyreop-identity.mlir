// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// spyreop.identity lowers to an OR with zero, one vectorchain.binary on the
// vector the unit receives, sent on as the result. There is no template, no
// register and no opaque: the OR is the compute, so the stage is not empty and
// nothing erases it. It is done on the value's own type, so a float's bits pass
// through as they are.

// CHECK-LABEL: func.func @identity_f16(
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:           %[[IN:.*]] = dataflow.receive %{{.*}} : vector<64xf16>
// CHECK-NEXT:      %[[OR:.*]] = vectorchain.binary %[[IN]], %[[ZERO]] {binary_op = #vectorchain<binary_operator or0>, op_specific_map = #{{.*}}} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NOT:       spyreop.identity
// CHECK:           dataflow.send %{{.*}}, %[[OR]] : vector<64xf16>
// CHECK-LABEL: func.func @identity_f32(
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:           %[[IN:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK-NEXT:      %[[OR:.*]] = vectorchain.binary %[[IN]], %[[ZERO]] {binary_op = #vectorchain<binary_operator or0>, op_specific_map = #{{.*}}} : vector<32xf32>, vector<32xf32>, vector<32xf32>
// CHECK-NOT:       spyreop.identity
// CHECK:           dataflow.send %{{.*}}, %[[OR]] : vector<32xf32>
// The zero is a constant, so it is hoisted to the top of the function.
// CHECK-LABEL: func.func @identity_i32(
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:           %[[IN:.*]] = dataflow.receive %{{.*}} : vector<32xi32>
// CHECK-NEXT:      %[[OR:.*]] = vectorchain.binary %[[IN]], %[[ZERO]] {binary_op = #vectorchain<binary_operator or0>, op_specific_map = #{{.*}}} : vector<32xi32>, vector<32xi32>, vector<32xi32>
// CHECK-NOT:       spyreop.identity
// CHECK:           dataflow.send %{{.*}}, %[[OR]] : vector<32xi32>

#id = affine_map<(d0) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @identity_f16() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %v = spyreop.identity %in : f16
        linalg.yield %v : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }

  func.func @identity_f32() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x32xf32, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 32xf32>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 32] to %in_fifo size [32] : memref<1x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %init   = tensor.empty() : tensor<32xf32>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<32xf32>) outs(%init : tensor<32xf32>) {
      ^bb0(%in: f32, %out: f32):
        %v = spyreop.identity %in : f32
        linalg.yield %v : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %result, %out_fifo : tensor<32xf32>, <"SFU" -> "L1LU", 32xf32>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [32] to %alloc_l1[%c0, %c0] size [1, 32] : !ktdf.fifo.slot<"SFU" -> "L1LU", 32xf32>, memref<1x32xf32, "L1">
    }
    return
  }

  func.func @identity_i32() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x32xi32, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xi32>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 32xi32>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 32] to %in_fifo size [32] : memref<1x32xi32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xi32>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xi32> -> tensor<32xi32>
      %init   = tensor.empty() : tensor<32xi32>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<32xi32>) outs(%init : tensor<32xi32>) {
      ^bb0(%in: i32, %out: i32):
        %v = spyreop.identity %in : i32
        linalg.yield %v : i32
      } -> tensor<32xi32>
      ktdf.write_to_fifo %result, %out_fifo : tensor<32xi32>, <"SFU" -> "L1LU", 32xi32>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [32] to %alloc_l1[%c0, %c0] size [1, 32] : !ktdf.fifo.slot<"SFU" -> "L1LU", 32xi32>, memref<1x32xi32, "L1">
    }
    return
  }
}
