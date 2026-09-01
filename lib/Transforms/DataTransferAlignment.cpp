//===----------------------------------------------------------------------===//
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
// DataTransferAlignment: Widen illegal ktdf.data_transfer strides.
//
// A ktdf.data_transfer is legal iff its size is a multiple of the transfer
// granularity T and both source and destination addresses are multiples of
// the access alignment granularity A (both read from the ktdf_arch.device
// spec at pass construction time, never hard-coded).
//
// When a transfer has a non-unit innermost stride on its source, this pass
// widens the transfer and every peer transfer in the same enclosing scf.for to
// a shape whose size is a multiple of T and whose addresses are multiples of
//
// Ordering:      After stage-coarsening (and its following canonicalize).
//                Before double-buffering — must see single staging buffers.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "data-transfer-alignment"
#define DEBUG_TYPE PASS_NAME

namespace scheduler {
#define GEN_PASS_DEF_DATATRANSFERALIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

/// DataTransferLegality helper, walks through and tracks data movement
/// chains throughout the IR and identifies when data transfers need to
/// be adjusted. This class caches previously illegal data transfers as
/// it is often the case when correcting one data transfer will lead to
/// subsequent data transfers to then become invalid.
class DataTransferLegality {
 public:
  llvm::SmallVector<mlir::ktdf::DataTransferOp> step(mlir::Operation* root) {
    llvm::SmallVector<mlir::ktdf::DataTransferOp> result;
    root->walk([&](mlir::ktdf::DataTransferOp op) {
      // Only memref sources can have stride violations.
      if (!op.isSourceMemRef()) return;

      auto memrefType = mlir::cast<mlir::MemRefType>(op.getSource().getType());

      llvm::SmallVector<int64_t> strides;
      int64_t offset;
      if (mlir::failed(memrefType.getStridesAndOffset(strides, offset))) {
        LDBG(1) << "  getStridesAndOffset failed for: " << op;
        return;
      }

      if (strides.empty()) return;

      int64_t innermostStride = strides.back();
      LDBG(1) << "  checking transfer (innermost stride=" << innermostStride
              << "): " << op;

      // Dynamic stride (kDynamic) or unit stride — both are fine / unknowable.
      if (innermostStride == 1 ||
          innermostStride == mlir::ShapedType::kDynamic)
        return;

      LDBG(1) << "  illegal transfer (innermost stride=" << innermostStride
              << "): " << op;
      result.push_back(op);
    });
    return result;
  }
};

namespace {

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";

    auto toFix = legality_.step(getOperation());
    if (toFix.empty()) {
      LDBG(1) << "  no illegal transfers found";
      return;
    }

    for (auto op : toFix) {
      op.emitError("data-transfer-alignment: illegal innermost stride not yet fixed");
    }
    LDBG(1) << "  found " << toFix.size() << " illegal transfer(s) not yet fixed";
  }

  DataTransferLegality legality_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
