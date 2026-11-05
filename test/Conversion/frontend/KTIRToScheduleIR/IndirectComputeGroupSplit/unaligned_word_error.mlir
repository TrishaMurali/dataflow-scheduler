// Verify that a compile-time constant base address or stride of the indirect
// access that is not a whole number of words is rejected with a diagnostic.
//
// spyreop.idx32toaddr takes its base and stride in the words the indirect
// transfer addresses the data's memory space in: 64 bytes for DDR on the
// sample device, i.e. 32 f16 elements.  A constant that does not convert
// exactly cannot be expressed, so the pass emits an error and fails.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split \
// RUN:   --split-input-file --verify-diagnostics %s

// -----

// Gather whose base address, 1000 f16 elements (2000 bytes), is not a
// multiple of the 64-byte word.
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    // expected-error @+1 {{indirect-compute-group-split: base address of the indirect access (1000 elements, 2000 bytes) is not a multiple of the 64-byte word the indirect transfer addresses memory in}}
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %cbase  = arith.constant 1000 : index
      %c10240 = arith.constant 10240 : index

      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      %desc_1 = ktdp.construct_memory_view %cbase, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      %desc_2 = ktdp.construct_memory_view %c10240, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
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

// -----

// Gather whose stride along the indirect dimension, 48 f16 elements (96
// bytes), is not a multiple of the 64-byte word.
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    // expected-error @+1 {{indirect-compute-group-split: stride of the indirect access (48 elements, 96 bytes) is not a multiple of the 64-byte word the indirect transfer addresses memory in}}
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %cbase  = arith.constant 1024 : index
      %c10240 = arith.constant 10240 : index

      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      %desc_1 = ktdp.construct_memory_view %cbase, sizes: [64, 2, 64], strides: [48, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      %desc_2 = ktdp.construct_memory_view %c10240, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
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

// -----

// Scatter whose base address, 10250 f16 elements (20500 bytes), is not a
// multiple of the store unit's 64-byte word.
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator child module.
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    // expected-error @+1 {{indirect-compute-group-split: base address of the indirect access (10250 elements, 20500 bytes) is not a multiple of the 64-byte word the indirect transfer addresses memory in}}
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0      : index
      %c100   = arith.constant 100    : index
      %c1000  = arith.constant 1000   : index
      %c10250 = arith.constant 10250  : index

      // Index memref: 2×32 si32.
      %desc_idx = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Source (direct) memref: 2×32×2×64 f16.
      %desc_src = ktdp.construct_memory_view %c1000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      // Destination (scatter target) memref: 64×2×64 f16, base = %c10250.
      // Indirect dimension D = 0 (stride[0] = 64 — the static stride for dim 0).
      %desc_dst = ktdp.construct_memory_view %c10250, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Direct load from source.
      %src_at = ktdp.construct_access_tile %desc_src[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      %src = ktdp.load %src_at : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      // Element-wise compute.
      %empty = tensor.empty() : tensor<2x32x2x64xf16>
      %result = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%src : tensor<2x32x2x64xf16>) outs(%empty : tensor<2x32x2x64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %cf10 = arith.constant 10.000000e+00 : f16
          %sum = arith.addf %in, %cf10 : f16
          linalg.yield %sum : f16
      } -> tensor<2x32x2x64xf16>

      // Indirect access tile for scatter destination (dim 0 is indirect).
      %dst_tile = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_dst[ind(%desc_idx[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      // Scatter: indirect store.
      ktdp.store %result, %dst_tile : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
