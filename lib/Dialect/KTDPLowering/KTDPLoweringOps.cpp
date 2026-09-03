//===-- KTDPLoweringOps.cpp -------------------------------------*- C++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ktdp_lowering dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
// clang-format on

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

using namespace mlir;
using namespace mlir::ktdp_lowering;

//===----------------------------------------------------------------------===//
// KTDPLoweringDialect — op registration
//===----------------------------------------------------------------------===//

void KTDPLoweringDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructIndirectAccessTileOp::verify() {
  // ind_addr_buf_subscripts count must equal the rank of ind_addr_buf_memref.
  auto iab_type = mlir::cast<mlir::MemRefType>(getIndAddrBufMemref().getType());
  auto num_subscripts = static_cast<int64_t>(getIndAddrBufSubscripts().size());
  if (num_subscripts != iab_type.getRank())
    return emitOpError() << "ind_addr_buf_subscripts has " << num_subscripts
                         << " operand(s) but ind_addr_buf_memref has rank "
                         << iab_type.getRank() << "; they must be equal";

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructMemoryViewOp::verify() {
  unsigned nDims = getStaticSizes().size();
  if (getStaticStrides().size() != nDims)
    return emitOpError(
        "static_sizes and static_strides must have equal length");
  unsigned dynSizes = llvm::count(getStaticSizes(), mlir::ShapedType::kDynamic);
  unsigned dynStrides =
      llvm::count(getStaticStrides(), mlir::ShapedType::kDynamic);
  if (getSizes().size() != dynSizes)
    return emitOpError(
        "number of dynamic size operands does not match "
        "kDynamic entries in static_sizes");
  if (getStrides().size() != dynStrides)
    return emitOpError(
        "number of dynamic stride operands does not match "
        "kDynamic entries in static_strides");
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(getResult().getType());
  if (!memrefType) return emitOpError("result must be a memref type");
  if (memrefType.getRank() != static_cast<int64_t>(nDims))
    return emitOpError(
        "result memref rank does not match sizes/strides length");

  // Check 1: memory_space attribute must match the memref's memory space.
  if (getMemorySpace() != memrefType.getMemorySpace())
    return emitOpError(
        "memory_space attribute does not match result memref memory space");

  // Check 2: strides must be consistent with the result memref's layout.
  // If the memref has a StridedLayoutAttr, compare directly. If it has no
  // layout (identity / row-major), verify that the provided static strides
  // are strictly decreasing left-to-right and that each stride equals the
  // product of all static sizes to its right (i.e. they encode a row-major
  // layout).  Dynamic strides/sizes are skipped in both paths.
  llvm::ArrayRef<int64_t> opStrides = getStaticStrides();
  if (auto stridedLayout = mlir::dyn_cast_or_null<mlir::StridedLayoutAttr>(
          memrefType.getLayout())) {
    llvm::ArrayRef<int64_t> layoutStrides = stridedLayout.getStrides();
    for (unsigned i = 0; i < nDims; ++i) {
      int64_t opS = opStrides[i];
      int64_t lyS = layoutStrides[i];
      if (opS == mlir::ShapedType::kDynamic ||
          lyS == mlir::ShapedType::kDynamic)
        continue;
      if (opS != lyS)
        return emitOpError()
               << "stride " << i << " (" << opS
               << ") does not match the result memref layout stride (" << lyS
               << ")";
    }
  } else if (!memrefType.getLayout() ||
             mlir::isa<mlir::AffineMapAttr>(memrefType.getLayout())) {
    // Identity (no layout) or affine-map layout — check that the static
    // strides encode a row-major (decreasing) layout consistent with the
    // static sizes.
    llvm::ArrayRef<int64_t> sizes = getStaticSizes();
    // Walk from the rightmost dimension outward.  expectedStride accumulates
    // the product of static sizes seen so far (starting from 1 for dim N-1).
    int64_t expectedStride = 1;
    for (int i = static_cast<int>(nDims) - 1; i >= 0; --i) {
      int64_t s = opStrides[i];
      if (s != mlir::ShapedType::kDynamic) {
        if (s != expectedStride)
          return emitOpError()
                 << "stride " << i << " (" << s
                 << ") is inconsistent with a row-major layout for the given "
                    "static sizes (expected "
                 << expectedStride << ")";
      }
      // Advance the expected stride by this dimension's size (skip if dynamic).
      if (i > 0) {
        int64_t sz = sizes[i];
        if (sz == mlir::ShapedType::kDynamic)
          expectedStride = mlir::ShapedType::kDynamic;  // can't predict further
        else if (expectedStride != mlir::ShapedType::kDynamic)
          expectedStride *= sz;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp — ViewLikeOpInterface
//===----------------------------------------------------------------------===//

mlir::Value ConstructMemoryViewOp::getViewSource() { return getOffset(); }
