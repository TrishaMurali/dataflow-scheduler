// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -split-input-file -verify-diagnostics %s | FileCheck %s

// Tests lowering zero-padding data transfers to a padding shuffle.

// A back-padded load is followed by the source, then zeros.
// CHECK-LABEL: func.func @zero_pad_back
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[LOAD:.+]] = agen.vector_load {{.+}} vector<2xf16>
// CHECK:         %[[PAD:.+]] = vectorchain.shuffle input(%[[LOAD]]), pad(%[[ZERO]])
// CHECK-SAME:      indices = [0 : i32, 1 : i32, -1 : i32, -1 : i32, -1 : i32, -1 : i32, -1 : i32, -1 : i32], repetition = 1 : i32
// CHECK-SAME:      vector<2xf16>, f16, vector<8xf16>
// CHECK:         dataflow.send %{{.+}}, %[[PAD]] : vector<8xf16>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @zero_pad_back() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %0]):index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu = uniform.def_immutable_mapping([%c0 -> %1]):index
    %u_sfu = uniform.query_map(map:%map_sfu, key:%tile_id) : index
    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      %alloc = memref.alloc() : memref<2xf16, "L1">
      %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8xf16>
      ktdf_lowering.execute_on %u_l1lu {
        ktdf.data_transfer from %alloc[%c0] size [2] to %fifo size [8] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : memref<2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8xf16>
      }
    }
    return
  }
}

// -----

// A front-padded load is preceded by zeros, then the source.
// CHECK-LABEL: func.func @zero_pad_front
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[LOAD:.+]] = agen.vector_load {{.+}} vector<2xf16>
// CHECK:         %[[PAD:.+]] = vectorchain.shuffle input(%[[LOAD]]), pad(%[[ZERO]])
// CHECK-SAME:      indices = [-1 : i32, -1 : i32, -1 : i32, -1 : i32, -1 : i32, -1 : i32, 0 : i32, 1 : i32], repetition = 1 : i32
// CHECK-SAME:      vector<2xf16>, f16, vector<8xf16>
// CHECK:         dataflow.send %{{.+}}, %[[PAD]] : vector<8xf16>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @zero_pad_front() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %0]):index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu = uniform.def_immutable_mapping([%c0 -> %1]):index
    %u_sfu = uniform.query_map(map:%map_sfu, key:%tile_id) : index
    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      %alloc = memref.alloc() : memref<2xf16, "L1">
      %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8xf16>
      ktdf_lowering.execute_on %u_l1lu {
        ktdf.data_transfer from %alloc[%c0] size [2] to %fifo size [8] {transfer_mode = #ktdf.transfer_mode<zero_pad_front>} : memref<2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8xf16>
      }
    }
    return
  }
}

// -----

// Padding is only lowered on a load into a FIFO.
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  // expected-error @below {{failed to run operation lowerings for zero_pad_into_memory}}
  func.func @zero_pad_into_memory() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr = memref.alloc() : memref<2xf16, "DDR">
    %l1 = memref.alloc() : memref<8xf16, "L1">
    ktdf_lowering.execute_on %unit {
      // expected-error @below {{splat/pad transfer must load from memory into a FIFO}}
      // expected-error @below {{failed to legalize operation 'ktdf.data_transfer' that was explicitly marked illegal}}
      ktdf.data_transfer from %ddr[%c0] size [2] to %l1[%c0] size [8] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : memref<2xf16, "DDR">, memref<8xf16, "L1">
    }
    return
  }
}
