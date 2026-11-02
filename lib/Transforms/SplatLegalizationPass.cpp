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
// SplatLegalization: validate splat data_transfer loads against the load
// unit's SIMD capabilities, widen the load to the arch's access granularity
// when required, and annotate consuming read_from_fifo ops with the sub-SIMD
// shuffle mode that closes the gap.
//
// Problem this pass addresses: A load unit that cannot splat a single element
// at its access granularity loads the whole granule instead, which leaves one
// live element per sub-SIMD group rather than one for the whole vector.
//
// This pass's responsibilities are:
//   1. Validate that the load unit's SIMD feature declares splat capability.
//   2. Widen the innermost static_source_sizes dimension to the hardware's
//      access granularity when it exceeds the requested element count.
//   3. Annotate each consuming ktdf.read_from_fifo with
//      #ktdf.splat<first_subsimd_lane_to_each_subsimd> when widening occurred,
//      naming the sub-SIMD broadcast shuffle the lowering to DFIR will emit.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <memory>
#include <numeric>

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "splat-legalization"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable the Splat Legalization pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_SPLATLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Resolve the load unit kind for an op that may live inside either a
/// ktdf.stage (pre-KTDFToKTDFLow IR) or a ktdf_lowering.execute_on
/// (post-KTDFToKTDFLow / direct test IR).  Returns nullptr when the kind is
/// ambiguous or cannot be resolved.
scheduler::ResourceType resolveLoadUnitKind(mlir::Operation* op) {
  // Pre-lowering: inside a ktdf.stage with applicable_units.
  if (auto stage = op->getParentOfType<mlir::ktdf::StageOp>()) {
    auto units = stage.getApplicableUnits();
    if (!units || units->size() != 1) return nullptr;
    return mlir::dyn_cast<scheduler::ResourceType>(units->getValue().front());
  }

  // Post-lowering: inside a ktdf_lowering.execute_on with unit operands.
  auto exec = op->getParentOfType<mlir::ktdf_lowering::ExecuteOnOp>();
  if (!exec || exec.getUnits().empty()) return nullptr;
  return scheduler::getUnitResourceType(exec.getUnits().front())
      .value_or(scheduler::ResourceType{});
}

/// Validate that @p transfer's load unit declares splat capability.
static mlir::LogicalResult validateSplatCapability(
    mlir::ktdf::DataTransferOp transfer,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  const auto simd = resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(
      resolveLoadUnitKind(transfer));
  if (!simd || !simd.canSplat())
    return transfer.emitError(
        "splat data_transfer requires the load unit to declare "
        "ktdf_arch.feature.simd = { splat, ... }");
  return mlir::success();
}

/// Legalize `transfer`'s load to the arch's access granularity and name, on
/// every read of it, the shuffle that finishes the splat.  A no-op when the
/// granularity is no wider than the source, i.e. when the hardware can splat
/// what was asked for as it stands.
mlir::LogicalResult legalizeSplatTransfer(
    mlir::RewriterBase& rewriter, mlir::ktdf::DataTransferOp transfer,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  scheduler::ResourceType load_unit_kind = resolveLoadUnitKind(transfer);

  auto static_src_sizes = transfer.getStaticSourceSizesArray();
  if (!static_src_sizes || static_src_sizes->empty())
    return transfer.emitError(
        "splat data_transfer requires static source sizes to derive the load "
        "width from");

  const int64_t src_elements =
      std::accumulate(static_src_sizes->begin(), static_src_sizes->end(),
                      int64_t{1}, std::multiplies<>());
  auto fifo_slot_type =
      mlir::cast<mlir::ktdf::FifoSlotType>(transfer.getDestination().getType());
  const int64_t load_elements = scheduler::computeSplatGranularityElements(
      src_elements, fifo_slot_type.getElementType(), load_unit_kind,
      resource_kinds);

  // The hardware can splat the source as it stands: nothing to record.
  if (load_elements <= src_elements) return mlir::success();

  LDBG(1) << "legalizing splat load from " << src_elements << " to "
          << load_elements << " elements for " << transfer;

  // The size operand carries the load width, so the lowering pass reads it
  // straight off the transfer and no extra attribute is needed for it.
  mlir::ktdf::setInnermostStaticSourceSize(rewriter, transfer, load_elements);

  // A legalized load leaves one live element per sub-SIMD group rather than
  // one for the whole vector.  Name the shuffle that closes that gap on every
  // read of the slot; the lowering to DFIR emits it right after the receive,
  // with the group width coming from the compute unit's sub_simd_lanes.
  mlir::Value fifo_slot = transfer.getDestination();
  for (mlir::Operation* user : fifo_slot.getUsers()) {
    auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(user);
    if (read_op && read_op.getFifoSlot() == fifo_slot) {
      rewriter.modifyOpInPlace(read_op, [&] {
        read_op.setSplat(mlir::ktdf::SplatMode::FirstSubSimdLaneToEachSubSimd);
      });
    }
  }
  return mlir::success();
}

struct SplatLegalizationPass
    : public scheduler::impl::SplatLegalizationPassBase<SplatLegalizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module_op = getOperation();

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

    llvm::SmallVector<mlir::ktdf::DataTransferOp> splats;
    module_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      if (transfer.isDestFifo() &&
          transfer.getTransferMode() == mlir::ktdf::TransferMode::Splat)
        splats.push_back(transfer);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (auto transfer : splats) {
      if (mlir::failed(validateSplatCapability(transfer, resource_kinds)) ||
          mlir::failed(
              legalizeSplatTransfer(rewriter, transfer, resource_kinds))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createSplatLegalizationPass() {
  return std::make_unique<SplatLegalizationPass>();
}
