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
// ScalarBroadcastLegalization: legalize scalar broadcast inputs to linalg ops.
//
// Detects linalg.generic ops in compute stages whose input indexing maps have
// a constant in the vector dimension (scalar broadcast), and legalizes the
// compute side by:
//   1. Updating the read_from_fifo result type to the full vector width.
//   2. Replacing the constant-in-vector-dim indexing map with the identity.
//
// Memory-side transfer widening and FIFO splat setup are handled upstream by
// access-granularity-legalization and the front-end lowering respectively.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/WalkResult.h>

#include <memory>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "scalar-broadcast-legalization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_SCALARBROADCASTLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Scalar Broadcast Legalization pass"),
    llvm::cl::init(false));

namespace {

[[nodiscard]]
auto getResourceKind(mlir::ktdf::StageOp stage) -> ResourceType {
  if (const auto maybe_units = stage.getApplicableUnits();
      maybe_units && maybe_units->size() == 1) {
    return llvm::cast<ResourceType>(maybe_units->getValue().front());
  }
  return nullptr;
}

struct BroadcastLegalizationSite {
  mlir::linalg::GenericOp generic_op;
  unsigned operand_index;
  int64_t vector_width;
  mlir::AffineMap new_indexing_map;
};

static bool isBroadcastOverVectorDim(mlir::AffineMap m) {
  if (m.getNumResults() == 0) return false;
  return mlir::isa<mlir::AffineConstantExpr>(m.getResults().back());
}

static void collectSites(mlir::ktdf::PipelineOp pipeline,
                         llvm::SmallVector<BroadcastLegalizationSite>& sites,
                         mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  pipeline.walk([&](mlir::linalg::GenericOp generic_op) {
    auto stage = generic_op->getParentOfType<mlir::ktdf::StageOp>();
    if (!stage) return;
    auto compute_kind = getResourceKind(stage);
    if (!compute_kind) return;
    auto compute = resource_kinds.getInstance(compute_kind);
    if (!compute) return;

    auto inputs = generic_op.getDpsInputs();
    if (inputs.empty()) return;
    mlir::Type elem_type =
        mlir::cast<mlir::RankedTensorType>(inputs[0].getType())
            .getElementType();
    int64_t vector_width =
        compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(
            elem_type);
    if (vector_width == 0) {
      generic_op->emitWarning("unsupported vector scalar type ") << elem_type;
      return;
    }

    auto indexing_maps = generic_op.getIndexingMapsArray();
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (unsigned i = 0; i < num_inputs; ++i) {
      mlir::AffineMap m = indexing_maps[i];
      if (!isBroadcastOverVectorDim(m)) continue;

      llvm::SmallVector<mlir::AffineExpr> new_results(m.getResults().begin(),
                                                      m.getResults().end());
      new_results.back() =
          mlir::getAffineDimExpr(m.getNumDims() - 1, generic_op.getContext());
      mlir::AffineMap new_map =
          mlir::AffineMap::get(m.getNumDims(), m.getNumSymbols(), new_results,
                               generic_op.getContext());

      sites.push_back({generic_op, i, vector_width, new_map});
    }
  });
}

/// Update the read_from_fifo result type and the linalg.generic indexing map
/// for one BroadcastLegalizationSite.
static void applyTransformation(const BroadcastLegalizationSite& site,
                                mlir::RewriterBase& rewriter) {
  mlir::linalg::GenericOp generic_op = site.generic_op;
  mlir::Value input = generic_op.getDpsInputs()[site.operand_index];

  auto read_op = input.getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
  assert(read_op && "target must be read_from_fifo");

  auto old_result_type =
      mlir::cast<mlir::RankedTensorType>(read_op.getResult().getType());
  llvm::SmallVector<int64_t> new_shape(old_result_type.getShape());
  new_shape.back() = site.vector_width;
  auto new_result_type =
      mlir::RankedTensorType::get(new_shape, old_result_type.getElementType());

  rewriter.setInsertionPoint(read_op);
  auto new_read = mlir::ktdf::ReadFromFifoOp::create(
      rewriter, read_op.getLoc(), new_result_type, read_op.getFifoSlot());

  read_op.getResult().replaceAllUsesWith(new_read.getResult());
  read_op.erase();

  auto maps = generic_op.getIndexingMapsArray();
  maps[site.operand_index] = site.new_indexing_map;
  generic_op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
}

struct ScalarBroadcastLegalizationPass
    : public impl::ScalarBroadcastLegalizationPassBase<
          ScalarBroadcastLegalizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module_op = getOperation();
    llvm::SmallVector<BroadcastLegalizationSite> sites;

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    module_op.walk([&](mlir::ktdf::PipelineOp pipeline) {
      collectSites(pipeline, sites, resource_kinds);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (const auto& site : sites) {
      applyTransformation(site, rewriter);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createScalarBroadcastLegalizationPass() {
  return std::make_unique<ScalarBroadcastLegalizationPass>();
}
