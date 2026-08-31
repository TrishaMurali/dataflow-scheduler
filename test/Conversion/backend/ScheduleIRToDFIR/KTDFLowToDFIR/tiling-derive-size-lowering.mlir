// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK-LABEL:   func.func @tiling.derive_size_with_remainder(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10x200xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 6 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 7 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 5 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_7]] step %[[CONSTANT_5]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_3]] step %[[CONSTANT_5]] {
// CHECK-NEXT:         %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:         %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONSTANT_6]], %[[CONSTANT_1]] : index
// CHECK-NEXT:         %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:         %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[CONSTANT_6]], %[[CONSTANT_5]] : index
// CHECK-NEXT:         scf.for %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[SELECT_0]] step %[[CONSTANT_5]] {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[SELECT_1]] step %[[CONSTANT_5]] {
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : index
// CHECK-NEXT:             memref.store %[[ADDI_0]], %[[ARG0]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : memref<10x200xindex>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// When total_size is an exact multiple of tile_size there is no epilogue, so
// tiling.derive_size lowers directly to the steady-state constant with no cmpi/select.
// CHECK-LABEL:   func.func @tiling.derive_size_no_remainder(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10x200xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 128 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] {
// CHECK-NOT:        arith.cmpi
// CHECK-NOT:        arith.select
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : index
// CHECK-NEXT:         memref.store %[[ADDI_0]], %[[ARG0]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] : memref<10x200xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }






ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @tiling.derive_size_with_remainder(%arg0: memref<10x200xindex>) {
  %c7 = arith.constant 7 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index

  scf.for %arg1 = %c0 to %c5 step %c1 {
    scf.for %arg2 = %c0 to %c7 step %c1 {
      %0 = ktdf.tiling.derive_size [%arg1 : %c3], total_size = %c5 : index
      %1 = ktdf.tiling.derive_size [%arg2 : %c3], total_size = %c7 : index
      scf.for %arg3 = %c0 to %0 step %c1 {
        scf.for %arg4 = %c0 to %1 step %c1 {
          %2 = arith.addi %arg3, %arg4 : index
          memref.store %2, %arg0[%arg3, %arg4] : memref<10x200xindex>
        }
      }
    }
  }
  return
}

func.func @tiling.derive_size_no_remainder(%arg0: memref<10x200xindex>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  scf.for %arg1 = %c0 to %c128 step %c1 {
    %0 = ktdf.tiling.derive_size [%arg1 : %c32], total_size = %c128 : index
    scf.for %arg2 = %c0 to %0 step %c1 {
      %1 = arith.addi %arg1, %arg2 : index
      memref.store %1, %arg0[%arg1, %c0] : memref<10x200xindex>
    }
  }
  return
}