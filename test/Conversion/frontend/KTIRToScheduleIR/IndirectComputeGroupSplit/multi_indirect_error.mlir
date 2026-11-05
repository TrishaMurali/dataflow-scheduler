
// Diagnostic tests covering multiple calls or multiple ops in one func.
//
// No ktdf_arch.device is needed — the P3 check fires before any device lookup.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split \
// RUN:   --split-input-file --verify-diagnostics %s

// -----

// Verify that a child-module function containing MORE THAN ONE
// ktdp.construct_indirect_access_tile triggers precondition P3:
// the pass emits an error on the FuncOp and signals failure.
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set_idx  = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set_base = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>

module {
  module {
    func.func @main() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  module @local_schedule_0 {
    // expected-error @+1 {{indirect-compute-group-split: more than one indirect tensor in function 'local_schedule_0' is not supported}}
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0    = arith.constant 0    : index
      %c100  = arith.constant 100  : index
      %c1024 = arith.constant 1024 : index
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set_base,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>
      // First indirect op.
      %tile_a = ktdp.construct_indirect_access_tile intermediate_variables(%arg0, %arg1, %arg2, %arg3)
          %desc_1[ind(%desc_0[%c0 + %arg0, %c0 + %arg1]), (%c0 + %arg2), (%arg3)]
          {variables_space_order = #map, variables_space_set = #set_idx}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      // Second indirect op — same structure, triggers P3.
      %tile_b = ktdp.construct_indirect_access_tile intermediate_variables(%arg4, %arg5, %arg6, %arg7)
          %desc_1[ind(%desc_0[%c0 + %arg4, %c0 + %arg5]), (%c0 + %arg6), (%arg7)]
          {variables_space_order = #map, variables_space_set = #set_idx}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      %loaded_a = ktdp.load %tile_a : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      %loaded_b = ktdp.load %tile_b : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      return
    }
  }
}

// -----

// Verify that an indirect compute group the orchestrator calls more than once
// is rejected with a diagnostic.
//
// The address buffer is filled by a single @<name>_idx_to_addr call inserted
// before the gather/scatter call site (§4.6).  With two call sites the second
// one would gather using addresses resolved for the first one's %base/%stride,
// so the pass must fail rather than silently update only the first.
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      // expected-error @+1 {{indirect-compute-group-split: the orchestrator calls 'local_schedule_0' more than once; only a single call site per indirect compute group is supported}}
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %c1024  = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      %tmp1 = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %tmp1_0, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
