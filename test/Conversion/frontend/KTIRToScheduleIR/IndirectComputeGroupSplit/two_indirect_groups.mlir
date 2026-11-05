// Two extracted child modules, each with its own indirect access tile: verify
// that both are split independently.
//
// Exercises the state the pass resolves once and reuses across candidates: the
// top-level symbol table (each idx_to_addr module must get its own name), the
// orchestrator's symbol table (each needs its own forward declaration), the
// orchestrator call-site map, and the memory tracker (each address buffer must
// get its own address — 2x32 i32 entries = 64 entries, or 256 bytes, apart).
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// CHECK-LABEL: func.func @orchestrator
// CHECK:         call @local_schedule_0_idx_to_addr() : () -> ()
// CHECK:         call @local_schedule_0() : () -> ()
// CHECK:         call @local_schedule_1_idx_to_addr() : () -> ()
// CHECK:         call @local_schedule_1() : () -> ()
// CHECK:       func.func private @local_schedule_0()
// CHECK:       func.func private @local_schedule_1()
// CHECK:       func.func private @local_schedule_0_idx_to_addr()
// CHECK:       func.func private @local_schedule_1_idx_to_addr()

// Each group gets its own address buffer: the first at offset 0, the second one
// 2*32 = 64 entries (256 bytes) further along.  View offsets count entries.
// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         %[[BUF0:.*]] = arith.constant 0 : index
// CHECK:         ktdp.construct_memory_view %[[BUF0]], sizes: [2, 32]
// CHECK-SAME:      memref<2x32xi32

// CHECK-LABEL: module @local_schedule_1_idx_to_addr
// CHECK:         %[[BUF1:.*]] = arith.constant 64 : index
// CHECK:         ktdp.construct_memory_view %[[BUF1]], sizes: [2, 32]
// CHECK-SAME:      memref<2x32xi32

// ============================================================================
// Input IR
// ============================================================================

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      call @local_schedule_1() : () -> ()
      return
    }
    func.func private @local_schedule_0()
    func.func private @local_schedule_1()
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

  module @local_schedule_1 {
    func.func @local_schedule_1() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c200   = arith.constant 200   : index
      %c2048  = arith.constant 2048  : index
      %c20000 = arith.constant 20000 : index

      %desc_0 = ktdp.construct_memory_view %c200, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>
      %desc_1 = ktdp.construct_memory_view %c2048, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>
      %desc_2 = ktdp.construct_memory_view %c20000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
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
