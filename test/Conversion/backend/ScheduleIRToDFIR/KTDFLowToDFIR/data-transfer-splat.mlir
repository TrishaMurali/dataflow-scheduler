// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that ktdf.data_transfer with a splat transfer mode produces:
//   1. agen.vector_load with the SOURCE size vector (not dest size)
//   2. vectorchain.shuffle to broadcast to dest size
//   3. dataflow.send using the shuffle result (not the load result)

// CHECK-LABEL: func.func @splat_transfer
// CHECK:         dataflow.program_unit
// CHECK:           agen.vector_load
// CHECK-SAME:        vector<1xf16>
// CHECK-NEXT:      %[[SHUFFLE:.+]] = vectorchain.shuffle input(%[[LOAD:.+]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:           dataflow.send %{{.*}}, %[[SHUFFLE]] : vector<64xf16>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @splat_transfer() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %2], [%c1 -> %3]):index
    %u_sfu  = uniform.query_map(map:%map_sfu,  key:%tile_id) : index
    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      %alloc = memref.alloc() : memref<128xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf_lowering.execute_on %u_l1lu {
        scf.for %i = %c0 to %c12 step %c1 {
          // Source size [1], dest size [64]: splat 1 element to 64
          ktdf.data_transfer from %alloc[%c0] size [1]
                             to %fifo#0 size [64]
                             {transfer_mode = #ktdf.transfer_mode<splat>}
              : memref<128xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    }
    return
  }
}
