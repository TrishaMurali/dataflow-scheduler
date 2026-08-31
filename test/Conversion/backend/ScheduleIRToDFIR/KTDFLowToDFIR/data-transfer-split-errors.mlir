// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -verify-diagnostics -split-input-file %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/







// An AGEN transfer moves one hardware vector per time step. Requests wider than
// that walk their non-unit outer dims (plus a split innermost dim, if needed)
// as AGEN time dimensions, but only when the shape allows an exact split.
// Anything else is rejected rather than silently mis-lowered.
// The vector width here is 64 lanes.

// Splitting requires the innermost source/destination size to be a multiple of
// the vector width; a 96-lane innermost dim is not a multiple of 64 and cannot
// be covered by whole 64-lane vectors.
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  // expected-error @below {{failed to run operation lowerings for innermost_size_not_multiple_of_vector_width}}
  func.func @innermost_size_not_multiple_of_vector_width() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr_buf = memref.alloc() : memref<2x128x96xf16, "DDR">
    %l1_buf = memref.alloc() : memref<2x1x128x96xf16, "L1">
    ktdf_lowering.execute_on %unit {
      // expected-error @below {{data transfer of 12288 elements exceeds the hardware vector width of 64; splitting requires the innermost source and destination sizes to be a multiple of the vector width, but they are 96 and 96}}
      // expected-error @below {{failed to legalize operation 'ktdf.data_transfer' that was explicitly marked illegal}}
      ktdf.data_transfer from %ddr_buf[%c0, 0, 0] size [1, 128, 96]
                         to %l1_buf[%c0, 0, 0, 0] size [1, 1, 128, 96]
        : memref<2x128x96xf16, "DDR">, memref<2x1x128x96xf16, "L1">
    }
    return
  }
}

// -----

// The ordered walked dimensions must agree between the two sides, otherwise
// there is no single loop nest that walks both in lockstep.
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  // expected-error @below {{failed to run operation lowerings for outer_sizes_mismatch}}
  func.func @outer_sizes_mismatch() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr_buf = memref.alloc() : memref<2x256x64xf16, "DDR">
    %l1_buf = memref.alloc() : memref<2x128x64xf16, "L1">
    ktdf_lowering.execute_on %unit {
      // expected-error @below {{source and destination walked dimensions must match to split a transfer of 16384 elements across multiple vectors}}
      // expected-error @below {{failed to legalize operation 'ktdf.data_transfer' that was explicitly marked illegal}}
      ktdf.data_transfer from %ddr_buf[%c0, 0, 0] size [1, 256, 64]
                         to %l1_buf[0, 0, 0] size [2, 128, 64]
        : memref<2x256x64xf16, "DDR">, memref<2x128x64xf16, "L1">
    }
    return
  }
}

// -----

// A walked dimension whose two sides advance by different distances cannot stay
// on the time axis, but it can be walked by an enclosing loop instead. Doing
// that for two such dimensions at once would mean choosing a nesting order
// between them, which nothing in the transfer determines.
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  // expected-error @below {{failed to run operation lowerings for two_divergent_walked_dimensions}}
  func.func @two_divergent_walked_dimensions() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    // Per step of the outer walked dim: 8*64 = 512 elements in the source,
    // 8*128 = 1024 in the destination. Per step of the inner one: 64 and 128.
    %ddr_buf = memref.alloc() : memref<4x8x64xf16, "DDR">
    %l1_buf = memref.alloc() : memref<4x8x128xf16, "L1">
    ktdf_lowering.execute_on %unit {
      // expected-error @below {{source and destination advance by different distances in 2 walked dimensions of a transfer of 2048 elements; only one such dimension can be resolved}}
      // expected-error @below {{failed to legalize operation 'ktdf.data_transfer' that was explicitly marked illegal}}
      ktdf.data_transfer from %ddr_buf[0, 0, 0] size [4, 8, 64]
                         to %l1_buf[0, 0, 0] size [4, 8, 64]
        : memref<4x8x64xf16, "DDR">, memref<4x8x128xf16, "L1">
    }
    return
  }
}
