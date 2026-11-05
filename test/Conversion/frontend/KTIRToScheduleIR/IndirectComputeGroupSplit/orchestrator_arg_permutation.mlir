// The orchestrator's signature need not match the gather/scatter function's
// argument list: verify that the forwarded arguments are picked from the
// *call site* by the child's argument number, not from the orchestrator's
// own arguments by position.
//
// Here the orchestrator takes (%count, %stride, %base) — three arguments, in a
// different order from the child's — and calls
//   @local_schedule_0(%base, %stride)
// so child argument 0 is bound to the orchestrator's argument 2 and child
// argument 1 to its argument 1.  Mapping child argument i onto orchestrator
// argument i would forward %count as the base address and %stride as the row
// stride: silently wrong addresses.
//
// The data base is additionally computed inside the child as
//   %data_base = arith.addi %arg0, %c500
// but that computation is no longer cloned into the orchestrator at all: the
// orchestrator only forwards the raw block-argument dependency (child
// argument 0, i.e. %base), and @local_schedule_0_idx_to_addr rebuilds the
// arith.addi internally from its own forwarded argument.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// ============================================================================
// Orchestrator checks
// ============================================================================

// CHECK-LABEL: func.func @orchestrator
// CHECK-SAME:      %[[COUNT:[a-z0-9_]+]]: index, %[[OSTRIDE:[a-z0-9_]+]]: index, %[[OBASE:[a-z0-9_]+]]: index
// -- raw forwarding: %[[OBASE]] (child argument 0) and %[[OSTRIDE]] (child
//    argument 1) are passed straight through, with no arith.addi/constant
//    materialized here.
// CHECK:         call @local_schedule_0_idx_to_addr(%[[OBASE]], %[[OSTRIDE]])
// CHECK-SAME:      (index, index) -> ()
// CHECK:         call @local_schedule_0(%[[OBASE]], %[[OSTRIDE]]) : (index, index) -> ()

// ============================================================================
// @local_schedule_0_idx_to_addr — rebuilds the addi chain from its own args
// ============================================================================

// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK:         func.func @local_schedule_0_idx_to_addr
// CHECK-SAME:        (%[[FBASE_ARG:.*]]: index, %[[FSTRIDE_ELEMS:.*]]: index)
// CHECK:           %[[C500:.*]] = arith.constant 500 : index
// CHECK:           %[[FBASE_ELEMS:.*]] = arith.addi %[[FBASE_ARG]], %[[C500]] : index
// -- base and stride in 64-byte words: a runtime f16 count is divided by 32
// CHECK:           %[[FBASE:.*]] = arith.divsi %[[FBASE_ELEMS]], %{{.*}} : index
// CHECK:           %[[FSTRIDE:.*]] = arith.divsi %[[FSTRIDE_ELEMS]], %{{.*}} : index
// CHECK:           linalg.generic
// CHECK:             arith.index_castui %[[FBASE]] : index to i32
// CHECK:             arith.index_castui %[[FSTRIDE]] : index to i32
// CHECK:             spyreop.idx32toaddr

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
    // (%count, %row_stride, %base_addr): neither the arity nor the order lines
    // up with @local_schedule_0's argument list.  %count is not passed on.
    func.func @orchestrator(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg2, %arg1) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }

  module @local_schedule_0 {
    // %arg0 = base address of the data tensor, %arg1 = its dynamic stride
    // along the indirect dimension (D = 0).
    func.func @local_schedule_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %c500   = arith.constant 500   : index
      %c10000 = arith.constant 10000 : index

      // The data base is a computation over the function argument.
      %data_base = arith.addi %arg0, %c500 : index

      // Index memref: static base and strides.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data memref: base = %data_base, stride[0] = %arg1 (dynamic).
      %desc_1 = ktdp.construct_memory_view %data_base, sizes: [64, 2, 64], strides: [%arg1, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Output memref.
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
