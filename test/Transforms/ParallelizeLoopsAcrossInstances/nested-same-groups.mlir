// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// An outer pipeline whose only stage holds an inner pipeline. Both pipelines
// use only units a sub-core owns (L1LU, SFU, L1SU), so both would distribute
// over the same sub-core groups. Only the outer loop is distributed: splitting
// the inner loop across the same groups again would leave some iterations with
// no group to run on. The inner loop stays an scf.for.

// CHECK-LABEL:   func.func @nested_same_groups() {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:      ktdf.parallel (%[[I:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%[[C0]]) to (%[[C4]]) step (%[[C1]]) distribute(num_instances = 2) {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:            scf.for %[[J:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:              ktdf.pipeline {
// CHECK-NEXT:                ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:                  "test.load"(%[[I]], %[[J]]) : (index, index) -> ()
// CHECK-NEXT:                } {applicable_units = ["L1LU"]}
// CHECK-NEXT:                ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:                  "test.compute"(%[[I]], %[[J]]) : (index, index) -> ()
// CHECK-NEXT:                } {applicable_units = ["SFU"]}
// CHECK-NEXT:                ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:                  "test.store"(%[[I]], %[[J]]) : (index, index) -> ()
// CHECK-NEXT:                } {applicable_units = ["L1SU"]}
// CHECK-NEXT:              }
// CHECK-NEXT:            } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:          } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:        }
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @nested_same_groups() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          scf.for %j = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              ktdf.stage depends_in(none) depends_out(none) {
                "test.load"(%i, %j) : (index, index) -> ()
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(none) depends_out(none) {
                "test.compute"(%i, %j) : (index, index) -> ()
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(none) depends_out(none) {
                "test.store"(%i, %j) : (index, index) -> ()
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
