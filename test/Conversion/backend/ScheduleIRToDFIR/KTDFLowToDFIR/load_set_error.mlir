// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 128 + d1 * 64 + d2 * 64 + d3 * 64 + d4 * 64 + d5)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0) -> (0, 0, 0, 0)>
// CHECK: #[[$ATTR_8:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0) -> (0, 0, 0, 0, 0, 0)>
// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 * 128 + d1 * 64 + d2 * 64 + d3 * 64 + d4 * 64 + d5 * 64 + d6)>
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 4096 + d2 * 4096 + d3 * 64 + d4)>
// CHECK: #[[$ATTR_13:.+]] = affine_map<(d0) -> (0, 0, 0, 0, 0, 0, 0)>
// CHECK: #[[$ATTR_14:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[$ATTR_15:.+]] = affine_map<(d0) -> (0, 0, 0, 0, 0)>
// CHECK: #[[$ATTR_16:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_17:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_18:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_19:.+]] = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 == 0, d5 >= 0, -d5 + 63 >= 0)>
// CHECK: #[[$ATTR_20:.+]] = affine_set<(d0, d1, d2, d3, d4, d5, d6) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 == 0, d5 == 0, d6 >= 0, -d6 + 63 >= 0)>
// CHECK: #[[$ATTR_21:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK:   module {
// CHECK:     module {
// CHECK:     func.func @Add_1414() attributes {grid = [2]} {
// CHECK-NEXT:       call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @"local-schedule-0"()
// CHECK-NEXT:   }

// CHECK:     module {
// CHECK:     func.func private @"local-schedule-0"() attributes {grid = [2]} {
// CHECK-NEXT:       %[[CONSTANT_12:.*]] = arith.constant 2816 : index
// CHECK-NEXT:       %[[CONSTANT_13:.*]] = arith.constant 768 : index
// CHECK-NEXT:       %[[CONSTANT_7:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 1664 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1152 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 640 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 128 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 64000 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 113152 : index
// CHECK-NEXT:       %[[CONSTANT_6:.*]] = arith.constant 113216 : index
// CHECK-NEXT:       %[[CONSTANT_8:.*]] = arith.constant 6 : index
// CHECK-NEXT:       %[[CONSTANT_9:.*]] = arith.constant 32 : index
// CHECK-NEXT:       %[[CONSTANT_10:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_11:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:       %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:       %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[GET_UNIT_8:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:       %[[GET_UNIT_9:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:       %[[GET_UNIT_10:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:       %[[GET_UNIT_11:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:       %[[GET_UNIT_12:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:         %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_4]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:         %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_5]] {layout_map = #[[$ATTR_1]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:         %[[GET_LOGICAL_MEMORY_VIEW_2:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_11]] {layout_map = #[[$ATTR_1]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:         agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_11]], %[[CONSTANT_11]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_2]][0, 0]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_1:.*]]:vector<64xf16>)
// CHECK-NEXT:          {load_order = #[[$ATTR_2]], load_set = #[[$ATTR_16]], load_time_addr_map = #[[$ATTR_3]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_16]], store_time_addr_map = #[[$ATTR_3]], time_order = #[[$ATTR_4]], time_set = #[[$ATTR_17]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<1x64xf16>, memref<1x64xf16>
// CHECK-NEXT:         %[[FOR_0:.*]] = scf.for %[[VAL_1]] = %[[CONSTANT_11]] to %[[CONSTANT_8]] step %[[CONSTANT_10]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_2]]) -> (index) {
// CHECK-NEXT:           %[[FOR_1:.*]] = scf.for %[[VAL_3:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_9]] step %[[CONSTANT_10]] iter_args(%[[VAL_4:.*]] = %[[VAL_2]]) -> (index) {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_13]], %[[VAL_4]] : index
// CHECK-NEXT:             %[[GET_LOGICAL_MEMORY_VIEW_3:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[SUBI_0]] {layout_map = #[[$ATTR_5]]} : index, index, memref<2x2x1x1x1x64xf16>
// CHECK-NEXT:             scf.for %[[VAL_5:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:               scf.for %[[VAL_6:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:                 agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[VAL_1]] * 2 + %[[VAL_5]], 0, %[[VAL_3]] * 2 + %[[VAL_6]], 0] dst:%[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[VAL_5]], %[[VAL_6]], 0, 0, 0, 0]
// CHECK-NEXT:                  time_symbols(), load_iv(%[[VAL_7:.*]]:vector<64xf16>)
// CHECK-NEXT:                  {load_order = #[[$ATTR_6]], load_set = #[[$ATTR_18]], load_time_addr_map = #[[$ATTR_7]], store_order = #[[$ATTR_8]], store_set = #[[$ATTR_19]], store_time_addr_map = #[[$ATTR_9]], time_order = #[[$ATTR_4]], time_set = #[[$ATTR_17]]}
// CHECK-NEXT:                 {
// CHECK-NEXT:                   agen.yield
// CHECK-NEXT:                 } : memref<12x1x64x64xf16>, memref<2x2x1x1x1x64xf16>
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_0]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_0]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_0]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_1]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_2]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_3]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_1]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_2]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_3]] : index
// CHECK-NEXT:             scf.yield %[[SUBI_0]] : index
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           scf.yield %[[FOR_1]] : index
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_8:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[VAL_8]]) : index
// CHECK-NEXT:         %[[GET_LOGICAL_MEMORY_VIEW_4:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_4]], %[[CONSTANT_11]] {layout_map = #[[$ATTR_1]]} : index, index, memref<1x64xf16>
// CHECK-NEXT:         %[[FOR_2:.*]] = scf.for %[[VAL_9:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_8]] step %[[CONSTANT_10]] iter_args(%[[VAL_10:.*]] = %[[CONSTANT_3]]) -> (index) {
// CHECK-NEXT:           %[[FOR_3:.*]] = scf.for %[[VAL_11:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_9]] step %[[CONSTANT_10]] iter_args(%[[VAL_12:.*]] = %[[VAL_10]]) -> (index) {
// CHECK-NEXT:             %[[GET_LOGICAL_MEMORY_VIEW_5:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_4]], %[[VAL_12]] {layout_map = #[[$ATTR_5]]} : index, index, memref<2x2x1x1x1x64xf16>
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[VAL_8]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[VAL_8]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[VAL_8]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_5]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_6]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_7]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_5]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_6]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_7]] : index
// CHECK-NEXT:             scf.for %[[VAL_13:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:               scf.for %[[VAL_14:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:                 %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[GET_LOGICAL_MEMORY_VIEW_5]]{{\[}}%[[VAL_13]], %[[VAL_14]], 0, 0, 0, 0] {load_order = #[[$ATTR_8]], load_set = #[[$ATTR_19]]} : memref<2x2x1x1x1x64xf16>, vector<64xf16>
// CHECK-NEXT:                 %[[DEF_IMMUTABLE_MAPPING_8:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:                 %[[QUERY_MAP_8:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_8]], key:%[[VAL_8]]) : index
// CHECK-NEXT:                 dataflow.send %[[QUERY_MAP_8]], %[[VECTOR_LOAD_0]] : vector<64xf16>
// CHECK-NEXT:                 %[[VECTOR_LOAD_1:.*]] = agen.vector_load %[[GET_LOGICAL_MEMORY_VIEW_4]][0, 0] {load_order = #[[$ATTR_2]], load_set = #[[$ATTR_16]]} : memref<1x64xf16>, vector<64xf16>
// CHECK-NEXT:                 %[[DEF_IMMUTABLE_MAPPING_9:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:                 %[[QUERY_MAP_9:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_9]], key:%[[VAL_8]]) : index
// CHECK-NEXT:                 dataflow.send %[[QUERY_MAP_9]], %[[VECTOR_LOAD_1]] : vector<64xf16>
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_10:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_10:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_10]], key:%[[VAL_8]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_11:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_11:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_11]], key:%[[VAL_8]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_12:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_8]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_12:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_12]], key:%[[VAL_8]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_10]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_11]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_12]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_10]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_11]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_12]] : index
// CHECK-NEXT:             %[[SUBI_1:.*]] = arith.subi %[[CONSTANT_13]], %[[VAL_12]] : index
// CHECK-NEXT:             scf.yield %[[SUBI_1]] : index
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           scf.yield %[[FOR_3]] : index
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_15:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         scf.for %[[VAL_16:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_8]] step %[[CONSTANT_10]] {
// CHECK-NEXT:           scf.for %[[VAL_17:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_9]] step %[[CONSTANT_10]] {
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_13:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_13:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_13]], key:%[[VAL_15]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_14:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_14:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_14]], key:%[[VAL_15]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_15:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_15:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_15]], key:%[[VAL_15]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_13]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_14]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_15]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_13]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_14]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_15]] : index
// CHECK-NEXT:             scf.for %[[VAL_18:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:               scf.for %[[VAL_19:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:                 %[[DEF_IMMUTABLE_MAPPING_16:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:                 %[[QUERY_MAP_16:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_16]], key:%[[VAL_15]]) : index
// CHECK-NEXT:                 %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_16]] : vector<64xf16>
// CHECK-NEXT:                 %[[DEF_IMMUTABLE_MAPPING_17:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:                 %[[QUERY_MAP_17:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_17]], key:%[[VAL_15]]) : index
// CHECK-NEXT:                 %[[RECEIVE_1:.*]] = dataflow.receive %[[QUERY_MAP_17]] : vector<64xf16>
// CHECK-NEXT:                 %[[BINARY_0:.*]] = vectorchain.binary %[[RECEIVE_0]], %[[RECEIVE_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_4]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:                 %[[DEF_IMMUTABLE_MAPPING_18:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:                 %[[QUERY_MAP_18:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_18]], key:%[[VAL_15]]) : index
// CHECK-NEXT:                 dataflow.send %[[QUERY_MAP_18]], %[[BINARY_0]] : vector<64xf16>
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_19:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_19:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_19]], key:%[[VAL_15]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_20:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_20:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_20]], key:%[[VAL_15]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_21:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_4]] -> %[[GET_UNIT_8]]], {{\[}}%[[GET_UNIT_5]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_21:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_21]], key:%[[VAL_15]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_19]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_20]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_21]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_19]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_20]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_21]] : index
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_20:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_22:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_22:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_22]], key:%[[VAL_20]]) : index
// CHECK-NEXT:         %[[FOR_4:.*]] = scf.for %[[VAL_21:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_8]] step %[[CONSTANT_10]] iter_args(%[[VAL_22:.*]] = %[[CONSTANT_1]]) -> (index) {
// CHECK-NEXT:           %[[FOR_5:.*]] = scf.for %[[VAL_23:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_9]] step %[[CONSTANT_10]] iter_args(%[[VAL_24:.*]] = %[[VAL_22]]) -> (index) {
// CHECK-NEXT:             %[[GET_LOGICAL_MEMORY_VIEW_6:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_22]], %[[VAL_24]] {layout_map = #[[$ATTR_10]]} : index, index, memref<2x2x1x1x1x1x64xf16>
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_23:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_23:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_23]], key:%[[VAL_20]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_24:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_24:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_24]], key:%[[VAL_20]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_25:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_25:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_25]], key:%[[VAL_20]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_23]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_24]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_25]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_23]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_24]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_25]] : index
// CHECK-NEXT:             scf.for %[[VAL_25:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:               scf.for %[[VAL_26:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:                 %[[DEF_IMMUTABLE_MAPPING_26:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:                 %[[QUERY_MAP_26:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_26]], key:%[[VAL_20]]) : index
// CHECK-NEXT:                 %[[RECEIVE_2:.*]] = dataflow.receive %[[QUERY_MAP_26]] : vector<64xf16>
// CHECK-NEXT:                 agen.vector_store %[[RECEIVE_2]], %[[GET_LOGICAL_MEMORY_VIEW_6]]{{\[}}%[[VAL_25]], %[[VAL_26]], 0, 0, 0, 0, 0] {store_order = #[[$ATTR_11]], store_set = #[[$ATTR_20]]} : memref<2x2x1x1x1x1x64xf16>, vector<64xf16>
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_27:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_27:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_27]], key:%[[VAL_20]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_28:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_28:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_28]], key:%[[VAL_20]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_29:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_6]] -> %[[GET_UNIT_8]]], {{\[}}%[[GET_UNIT_7]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_29:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_29]], key:%[[VAL_20]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_27]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_28]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_29]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_27]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_28]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_29]] : index
// CHECK-NEXT:             %[[SUBI_2:.*]] = arith.subi %[[CONSTANT_12]], %[[VAL_24]] : index
// CHECK-NEXT:             scf.yield %[[SUBI_2]] : index
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           scf.yield %[[FOR_5]] : index
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_27:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_30:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_30:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_30]], key:%[[VAL_27]]) : index
// CHECK-NEXT:         %[[GET_LOGICAL_MEMORY_VIEW_7:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_6]] {layout_map = #[[$ATTR_12]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:         %[[FOR_6:.*]] = scf.for %[[VAL_28:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_8]] step %[[CONSTANT_10]] iter_args(%[[VAL_29:.*]] = %[[CONSTANT_0]]) -> (index) {
// CHECK-NEXT:           %[[FOR_7:.*]] = scf.for %[[VAL_30:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_9]] step %[[CONSTANT_10]] iter_args(%[[VAL_31:.*]] = %[[VAL_29]]) -> (index) {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[CONSTANT_12]], %[[VAL_31]] : index
// CHECK-NEXT:             %[[GET_LOGICAL_MEMORY_VIEW_8:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_30]], %[[SUBI_3]] {layout_map = #[[$ATTR_10]]} : index, index, memref<2x2x1x1x1x1x64xf16>
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_31:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_31:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_31]], key:%[[VAL_27]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_32:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_4]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_32:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_32]], key:%[[VAL_27]]) : index
// CHECK-NEXT:             %[[DEF_IMMUTABLE_MAPPING_33:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_6]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:             %[[QUERY_MAP_33:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_33]], key:%[[VAL_27]]) : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_31]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_32]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_send %[[QUERY_MAP_33]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_31]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_32]] : index
// CHECK-NEXT:             dataflow.sync_recv %[[QUERY_MAP_33]] : index
// CHECK-NEXT:             scf.for %[[VAL_32:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:               scf.for %[[VAL_33:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_7]] step %[[CONSTANT_10]] {
// CHECK-NEXT:                 agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_8]]{{\[}}%[[VAL_32]], %[[VAL_33]], 0, 0, 0, 0, 0] dst:%[[GET_LOGICAL_MEMORY_VIEW_7]]{{\[}}%[[VAL_28]] * 2 + %[[VAL_32]], 0, 0, %[[VAL_30]] * 2 + %[[VAL_33]], 0]
// CHECK-NEXT:                  time_symbols(), load_iv(%[[VAL_34:.*]]:vector<64xf16>)
// CHECK-NEXT:                  {load_order = #[[$ATTR_11]], load_set = #[[$ATTR_20]], load_time_addr_map = #[[$ATTR_13]], store_order = #[[$ATTR_14]], store_set = #[[$ATTR_21]], store_time_addr_map = #[[$ATTR_15]], time_order = #[[$ATTR_4]], time_set = #[[$ATTR_17]]}
// CHECK-NEXT:                 {
// CHECK-NEXT:                   agen.yield
// CHECK-NEXT:                 } : memref<2x2x1x1x1x1x64xf16>, memref<12x1x1x64x64xf16>
// CHECK-NEXT:               } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:             scf.yield %[[SUBI_3]] : index
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           scf.yield %[[FOR_7]] : index
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @Add_1414() attributes {grid = [2]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %8 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
      %9 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
      %10 = ktdp.get_compute_tile_id : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %11 = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
      %12 = uniform.query_map(map:%11, key:%10) : index
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %13 = uniform.def_immutable_mapping([%c0_0 -> %2], [%c1_1 -> %3]):index
      %14 = uniform.query_map(map:%13, key:%10) : index
      %c0_2 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      %15 = uniform.def_immutable_mapping([%c0_2 -> %4], [%c1_3 -> %5]):index
      %16 = uniform.query_map(map:%15, key:%10) : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      %17 = uniform.def_immutable_mapping([%c0_4 -> %6], [%c1_5 -> %7]):index
      %18 = uniform.query_map(map:%17, key:%10) : index
      %c0_6 = arith.constant 0 : index
      %c1_7 = arith.constant 1 : index
      %19 = uniform.def_immutable_mapping([%c0_6 -> %8], [%c1_7 -> %9]):index
      %20 = uniform.query_map(map:%19, key:%10) : index
      %c32 = arith.constant 32 : index
      %c6 = arith.constant 6 : index
      %c2 = arith.constant 2 : index
      %c0_8 = arith.constant 0 : index
      %c1_9 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c113216 = arith.constant 113216 : index
      %c113152 = arith.constant 113152 : index
      %c64000 = arith.constant 64000 : index
      %c12 = arith.constant 12 : index
      %21 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
      %memspacecast = memref.memory_space_cast %21 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
      %22 = ktdp.construct_memory_view %c113152, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %memspacecast_10 = memref.memory_space_cast %22 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %reinterpret_cast_11 = memref.reinterpret_cast %memspacecast_10 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
      %cast_12 = memref.cast %reinterpret_cast_11 : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %23 = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
      %memspacecast_13 = memref.memory_space_cast %23 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
      %reinterpret_cast_14 = memref.reinterpret_cast %memspacecast_13 to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
      %cast_15 = memref.cast %reinterpret_cast_14 : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
      %c0_16 = arith.constant 0 : index
      %24 = builtin.unrealized_conversion_cast %c0_16 : index to memref<1x64xf16, "L1">
      ktdf_lowering.execute_on %12 {
        ktdf_lowering.execute_on %12 {
          ktdf.data_transfer from %cast_12[%c0_8, %c0_8] size [1, 64] to %24[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      }
      %c128 = arith.constant 128 : index
      %25 = builtin.unrealized_conversion_cast %c128 : index to memref<2x2x1x1x1x64xf16, "L1">
      %cast_17 = memref.cast %25 : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
      %c640 = arith.constant 640 : index
      %26 = builtin.unrealized_conversion_cast %c640 : index to memref<2x2x1x1x1x64xf16, "L1">
      %cast_18 = memref.cast %26 : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
      %c1152 = arith.constant 1152 : index
      %27 = builtin.unrealized_conversion_cast %c1152 : index to memref<2x2x1x1x1x1x64xf16, "L1">
      %cast_19 = memref.cast %27 : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
      %c1664 = arith.constant 1664 : index
      %28 = builtin.unrealized_conversion_cast %c1664 : index to memref<2x2x1x1x1x1x64xf16, "L1">
      %cast_20 = memref.cast %28 : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
      scf.for %arg0 = %c0_8 to %c6 step %c1_9 {
        %29 = ktdf.tiling.derive_size [%arg0 : %c2], total_size = %c12 : index
        scf.for %arg1 = %c0_8 to %c32 step %c1_9 {
          %30 = ktdf.tiling.derive_size [%arg1 : %c2], total_size = %c64 : index
          %31 = ktdf.buffer_phase(%arg0, %arg1) {num_phases = 2 : i64} : index
          %32 = ktdf.select_memref %31[%cast_17, %cast_18] : memref<?x?x1x1x1x64xf16, "L1">
          %33 = ktdf.select_memref %31[%cast_19, %cast_20] : memref<?x?x1x1x1x1x64xf16, "L1">
          ktdf_lowering.execute_on %12, %14, %16, %18, %20 {
            %c2176 = arith.constant 2176 : index
            %34 = builtin.unrealized_conversion_cast %c2176 : index to memref<1x64xf16, "L1">
            %35 = ktdf.create_token : !ktdf.token
            %36 = ktdf.create_token : !ktdf.token
            ktdf_lowering.execute_on %12 {
              scf.for %arg2 = %c0_8 to %29 step %c1_9 {
                scf.for %arg3 = %c0_8 to %30 step %c1_9 {
                  %37 = affine.apply #map(%arg0, %arg2)
                  %38 = affine.apply #map(%arg1, %arg3)
                  ktdf.data_transfer from %cast[%arg0 * 2 + %arg2, 0, %arg1 * 2 + %arg3, 0] size [1, 1, 1, 64] to %32[%arg2, %arg3, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            }
            ktdf_lowering.signal %12, %14, %16, %18
            ktdf_lowering.execute_on %14, %16, %18 {
              scf.for %arg2 = %c0_8 to %29 step %c1_9 {
                scf.for %arg3 = %c0_8 to %30 step %c1_9 {
                  ktdf_lowering.execute_on %14, %16, %18 {
                    %37:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    %38 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                    %39 = ktdf.create_token : !ktdf.token
                    %40 = ktdf.create_token : !ktdf.token
                    ktdf_lowering.execute_on %14 {
                      ktdf.data_transfer from %32[%arg2, %arg3, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %37#0 size [64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      ktdf.data_transfer from %24[0, 0] size [1, 64] to %37#1 size [64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    }
                    ktdf_lowering.execute_on %16 {
                      %41 = ktdf.read_from_fifo %37#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
                      %42 = ktdf.read_from_fifo %37#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                      %43 = tensor.empty() : tensor<1x1x1x1x64xf16>
                      %44 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%41, %42 : tensor<1x1x1x64xf16>, tensor<1x64xf16>) outs(%43 : tensor<1x1x1x1x64xf16>) {
                      ^bb0(%in: f16, %in_21: f16, %out: f16):
                        %45 = arith.addf %in, %in_21 : f16
                        linalg.yield %45 : f16
                      } -> tensor<1x1x1x1x64xf16>
                      ktdf.write_to_fifo %44, %38 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                    }
                    ktdf_lowering.execute_on %18 {
                      ktdf.data_transfer from %38 size [64] to %33[%arg2, %arg3, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, "L1">
                    }
                  }
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            }
            ktdf_lowering.signal %14, %16, %18, %20
            ktdf_lowering.execute_on %20 {
              scf.for %arg2 = %c0_8 to %29 step %c1_9 {
                scf.for %arg3 = %c0_8 to %30 step %c1_9 {
                  %37 = affine.apply #map(%arg0, %arg2)
                  %38 = affine.apply #map(%arg1, %arg3)
                  ktdf.data_transfer from %33[%arg2, %arg3, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %cast_15[%arg0 * 2 + %arg2, 0, 0, %arg1 * 2 + %arg3, 0] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            }
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      return
    }
  }
}
