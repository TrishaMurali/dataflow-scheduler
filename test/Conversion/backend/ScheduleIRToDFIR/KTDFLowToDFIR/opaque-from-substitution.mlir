// The whole way from an intrinsic to DFIR. The sample device substitutes exp
// for an opaque over four registers, hoisting sizes them to the tile, address
// assignment places them, and this lowering turns the `ktdf.opaque` into a
// `dataflow.opaque` naming the register each operand landed in.
//
// The addresses a register file is viewed at count elements, not bytes, so the
// second register of a file holding 64 lanes of f16 is at 64 and is R1. Only
// the registers something reads or writes keep a view; the scratch the template
// computes in is named in the dictionary and nothing else.

// RUN: dataflow-scheduler-opt %s -pass-pipeline="builtin.module( \
// RUN:   func.func(ktdfarch-apply-patterns{groups=pre_scheduling}), \
// RUN:   materialize-registers, \
// RUN:   address-assignment, \
// RUN:   ktdflowering-to-dfir)" | FileCheck %s

// CHECK-LABEL:   func.func @exp() attributes {grid = [2]} {
// CHECK-NEXT:      %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:      %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:      %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:      %[[SFU1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK:           %[[REGS:.*]] = dataflow.get_unit {name = "sfu_reg", type = "sfu_reg"} : index

// The compute unit: the constant register is filled across its lanes, the data
// arriving is stored into the input one, and the result is read back out of the
// output one -- either side of the opaque that does the work.
// CHECK:           dataflow.program_unit iter_arg : %{{.*}} -> (%[[SFU0]], %[[SFU1]]) : {
// CHECK:             %[[IN_VEC:.*]] = dataflow.receive
// CHECK-NEXT:        %[[C0_REG:.*]] = dataflow.get_logical_memory_view %[[REGS]], %[[ZERO]] {{.*}} : index, index, memref<64xf16>
// CHECK-NEXT:        %[[BITS:.*]] = vectorchain.constant_bitstream {value = [0x3c00]} : vector<1xf16>
// CHECK-NEXT:        %[[SPLAT:.*]] = vectorchain.shuffle input(%[[BITS]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NEXT:        agen.vector_store %[[SPLAT]], %[[C0_REG]]{{\[}}%[[ZERO]]]
// CHECK-NEXT:        %[[IN_REG:.*]] = dataflow.get_logical_memory_view %[[REGS]], %[[C64]] {{.*}} : index, index, memref<64xf16>
// CHECK-NEXT:        %[[OUT_REG:.*]] = dataflow.get_logical_memory_view %[[REGS]], %[[C128]] {{.*}} : index, index, memref<64xf16>
// CHECK-NEXT:        agen.vector_store %[[IN_VEC]], %[[IN_REG]]{{\[}}%[[ZERO]]]
// CHECK-NEXT:        dataflow.opaque {dataflow_scheduler.register_names = ["c0", "in0", "out0", "t0_0"], func_name = "fake_exp", parameter_dictionary = {}, read_only_register_dictionary = {c0 = "R0", in0 = "R1"}, read_write_register_dictionary = {out0 = "R2", t0_0 = "R3"}}
// CHECK-NEXT:        %[[OUT_VEC:.*]] = agen.vector_load %[[OUT_REG]]{{\[}}%[[ZERO]]]
// CHECK:             dataflow.send %{{.*}}, %[[OUT_VEC]] : vector<64xf16>

#id = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @exp() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %sfu1 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %l1su0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %l1su1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu = uniform.def_immutable_mapping([%c0 -> %sfu0], [%c1 -> %sfu1]) : index
    %u_sfu = uniform.query_map(map:%map_sfu, key:%tile_id) : index
    %map_l1su = uniform.def_immutable_mapping([%c0 -> %l1su0], [%c1 -> %l1su1]) : index
    %u_l1su = uniform.query_map(map:%map_l1su, key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {indexing_maps = [#id, #id],
                                iterator_types = ["parallel"]}
          ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%x: f16, %out: f16):
        %e = spyreop.exp %x : f16
        linalg.yield %e : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1su {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }
}
