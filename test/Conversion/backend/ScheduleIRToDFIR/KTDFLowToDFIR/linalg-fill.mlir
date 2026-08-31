// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify linalg.fill lowers to constant_bitstream + shuffle, then:
//   - memref output: agen.vector_store into the alloc
//   - tensor output: shuffle result replaces the fill SSA value

// ---- memref case ----
// CHECK-LABEL: func.func @fill_zero_memref
// CHECK:         dataflow.program_unit
// CHECK:           %[[BS:.+]] = vectorchain.constant_bitstream {value = [0x0]} : vector<1xf16>
// CHECK-NEXT:      %[[SH:.+]] = vectorchain.shuffle input(%[[BS]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NEXT:      agen.vector_store %[[SH]]

// ---- tensor case ----
// CHECK-LABEL: func.func @fill_zero_tensor
// CHECK:           %[[BS2:.+]] = vectorchain.constant_bitstream {value = [0x0]} : vector<1xf16>
// CHECK-NEXT:      %[[SH2:.+]] = vectorchain.shuffle input(%[[BS2]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NOT:       agen.vector_store

// ---- scalar case ----
// The value is not a constant, so the bitstream carries a symbol instead and
// what the symbol is is declared next to the program, one symbol per step of the
// arithmetic.
// CHECK-LABEL: func.func @fill_from_input(
// CHECK-SAME:      %[[BASE:.*]]: index)
// CHECK:         dataflow.program_unit
// CHECK:           %[[BS3:.+]] = vectorchain.constant_bitstream {is_symbol = true, value = {{\[}}[[ID:-[0-9]+]]]} : vector<1xf16>
// CHECK-NEXT:      %[[SH3:.+]] = vectorchain.shuffle input(%[[BS3]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:           agen.vector_store %[[SH3]]
// CHECK:         %[[NARROW:.+]] = arith.index_castui %[[BASE]] : index to i32
// CHECK-NEXT:    symbol.create_id %[[NARROW]] {symbol_id = {{-[0-9]+}} : si64} : i32
// CHECK-NEXT:    %[[WIDE:.+]] = arith.uitofp %[[NARROW]] : i32 to f16
// CHECK-NEXT:    symbol.create_id %[[WIDE]] {symbol_id = [[ID]] : si64} : f16

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // memref case: fill writes into an alloc
  func.func @fill_zero_memref() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    ktdf_lowering.execute_on %unit {
      %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
    }
    return
  }

  // tensor case: fill result flows into write_to_fifo
  func.func @fill_zero_tensor() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    ktdf_lowering.execute_on %unit {
      %init = tensor.empty() : tensor<64xf16>
      %zero = arith.constant 0.0 : f16
      %filled = linalg.fill ins(%zero : f16) outs(%init : tensor<64xf16>) -> tensor<64xf16>
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNILU" -> "MNILU", 64xf16>
      ktdf.write_to_fifo %filled, %fifo#0 : tensor<64xf16>, <"MNILU" -> "MNILU", 64xf16>
    }
    return
  }

  // scalar case: the fill value is computed from an input, so it is a symbol
  func.func @fill_from_input(%base: index) attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    ktdf_lowering.execute_on %unit {
      %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %narrow = arith.index_castui %base : index to i32
      %wide = arith.uitofp %narrow : i32 to f16
      linalg.fill ins(%wide : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
    }
    return
  }
}
