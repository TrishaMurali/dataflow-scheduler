// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s


// CHECK-LABEL:   func.func @units_basic() {
// CHECK-NEXT:     ktdf_lowering.execute_on {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     } {applicable_units = ["SFU"]}
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func @units_basic() {
    ktdf_lowering.execute_on {
      %c0 = arith.constant 0.0 : f32
    } {applicable_units = ["SFU"]}
    return
  }
}
