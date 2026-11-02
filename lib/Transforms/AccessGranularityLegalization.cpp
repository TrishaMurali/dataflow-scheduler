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
// AccessGranularityLegalization: enforce hardware physical access_granularity
// constraints across all ktdf.data_transfer operations.
//
// Runs immediately after path-expansion and before splat-legalization, once
// all physical execution units and intermediate memory tiers are fully
// instantiated and concrete.
//
// Problem this pass addresses: a hardware load unit may only be able to issue
// transfers whose size is a multiple of some minimum bus granularity.  For
// example, an MNILU can only read 64-element chunks from DDR, so a transfer
// of [1, 1, 1, 1] must be widened to [1, 1, 1, 64] -- both source and
// destination sizes -- and the destination buffer must be resized to match.
//
// This pass's responsibilities are:
//   1. For each ktdf.data_transfer whose source is a memref (load), query the
//      load unit's access_granularity for the source memory space and compute
//      the smallest legal transfer size that covers all requested elements.
//   2. When widening is needed, update the innermost static_source_sizes and
//      static_dest_sizes dimensions, and resize the destination buffer or FIFO
//      slot accordingly.
//   3. Skips transfers with dynamic sizes, non-memref sources (FIFO-to-memref),
//      and stages with no unambiguous load unit (no applicable_units).
//
// Notes on scope:
//   - Only load (read) widening is handled here; write widening requires
//     hardware per-byte write-mask support and is not yet implemented.
//   - Source buffer bounds checking (verifying the source memref is large
//     enough to support the widened physical read) is a TODO.
//   - Integration with ZeroPadFront / ZeroPadBack for boundary tiles is a TODO.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <memory>
#include <numeric>
#include <optional>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "access-granularity-legalization"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable the Access Granularity Legalization pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_ACCESSGRANULARITYLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Resolve the load unit kind for a transfer that lives inside a ktdf.stage.
/// Returns nullptr when the stage has no unambiguous single applicable unit.
static scheduler::ResourceType resolveLoadUnitKind(
    mlir::ktdf::DataTransferOp transfer) {
  auto stage = transfer->getParentOfType<mlir::ktdf::StageOp>();
  if (!stage) return nullptr;
  auto units = stage.getApplicableUnits();
  if (!units || units->size() != 1) return nullptr;
  return mlir::dyn_cast<scheduler::ResourceType>(units->getValue().front());
}

/// Compute the number of elements that must be loaded to satisfy the load
/// unit's access granularity for @p src_elements of @p elem_type from
/// @p memory_space.  Returns @p src_elements when the transfer already
/// satisfies the granularity or when granularity information is unavailable.
///
/// Transfer sizes are in elements; the arch graph uses words (where
/// word_size is in bytes).  The conversion is:
///   bytes_requested  = src_elements × elem_bytes
///   words_requested  = ceil(bytes_requested / word_size)
///   granule_words    = fitAccess(words_requested).size_in_words
///   granule_elements = granule_words × word_size / elem_bytes
///
/// This correctly handles both small-word units (e.g. L1LU, word_size=1)
/// and large-word units (e.g. L3LU, word_size=128), where one bus word
/// spans many elements.
static int64_t computeRequiredLoadElements(
    int64_t src_elements, mlir::Type elem_type, mlir::Attribute memory_space,
    scheduler::ResourceType unit_kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  if (!unit_kind || !memory_space) return src_elements;

  auto load_feature =
      resource_kinds.getFeature<mlir::ktdf_arch::feature::Load>(unit_kind);
  if (!load_feature) return src_elements;

  auto granularity_list = load_feature.getAccessGranularity(memory_space);
  if (!granularity_list) return src_elements;

  const size_t word_size = load_feature.getWordSize(memory_space);
  if (word_size == 0) return src_elements;

  const size_t elem_bytes =
      (static_cast<size_t>(elem_type.getIntOrFloatBitWidth()) + 7) / 8;
  if (elem_bytes == 0) return src_elements;

  // How many bus words are needed to cover all requested bytes?
  const size_t bytes_requested = static_cast<size_t>(src_elements) * elem_bytes;
  const size_t words_requested = (bytes_requested + word_size - 1) / word_size;

  auto best = granularity_list.fitAccess(words_requested);
  if (!best) return src_elements;

  // Convert the granule back to elements.  word_size is expected to be a
  // multiple of elem_bytes (or vice versa) for any real hardware type.
  const size_t granule_bytes = best.getSizeInWords() * word_size;
  const int64_t granule_elements =
      static_cast<int64_t>(granule_bytes / elem_bytes);
  return std::max(granule_elements, src_elements);
}

/// Widen the innermost source and destination sizes of @p transfer from
/// @p src_elements to @p required_elements.  For a memref destination,
/// resizes the backing alloc.  For a FIFO destination, resizes the slot.
/// Returns failure and emits a diagnostic if resizing is not possible.
static mlir::LogicalResult widenTransfer(mlir::RewriterBase& rewriter,
                                         mlir::ktdf::DataTransferOp transfer,
                                         int64_t required_elements) {
  LDBG(1) << "widening transfer to " << required_elements
          << " elements: " << transfer;

  // Widen the source sizes (innermost dimension carries the element count).
  mlir::ktdf::setInnermostStaticSourceSize(rewriter, transfer,
                                           required_elements);

  // Widen the destination sizes and resize the backing storage.
  auto dest_sizes_opt = transfer.getStaticDestSizesArray();
  if (dest_sizes_opt) {
    llvm::SmallVector<int64_t> new_dest(*dest_sizes_opt);
    new_dest.back() = required_elements;
    transfer.setStaticDestSizes(llvm::ArrayRef<int64_t>(new_dest));
  }

  const mlir::Value dest = transfer.getDestination();
  if (transfer.isDestFifo()) {
    if (!mlir::ktdf::resizePrivateFifoSlot(rewriter, dest, required_elements))
      return transfer.emitError(
          "access-granularity-legalization: cannot resize FIFO slot");
  } else {
    // MemRef destination: resize to the new innermost dimension.
    auto dest_type = mlir::cast<mlir::MemRefType>(dest.getType());
    llvm::SmallVector<int64_t> new_shape(dest_type.getShape());
    new_shape.back() = required_elements;
    if (!mlir::ktdf::resizePrivateBuffer(rewriter, dest, new_shape))
      return transfer.emitError(
          "access-granularity-legalization: cannot resize destination buffer");
  }
  return mlir::success();
}

/// Inspect @p transfer and widen it to the load unit's access granularity if
/// required.  Non-load transfers (source is a FIFO) and transfers with dynamic
/// sizes are skipped silently.
static mlir::LogicalResult legalizeTransfer(
    mlir::RewriterBase& rewriter, mlir::ktdf::DataTransferOp transfer,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  // Only loads (memref source) are handled here.
  if (!transfer.isSourceMemRef()) return mlir::success();

  auto static_src_sizes = transfer.getStaticSourceSizesArray();
  if (!static_src_sizes || static_src_sizes->empty()) return mlir::success();

  const int64_t src_elements =
      std::accumulate(static_src_sizes->begin(), static_src_sizes->end(),
                      int64_t{1}, std::multiplies<>());

  auto src_memref_type =
      mlir::cast<mlir::MemRefType>(transfer.getSource().getType());
  mlir::Type elem_type = src_memref_type.getElementType();
  mlir::Attribute memory_space = src_memref_type.getMemorySpace();

  scheduler::ResourceType unit_kind = resolveLoadUnitKind(transfer);

  const int64_t required_elements = computeRequiredLoadElements(
      src_elements, elem_type, memory_space, unit_kind, resource_kinds);

  if (required_elements <= src_elements) return mlir::success();

  return widenTransfer(rewriter, transfer, required_elements);
}

struct AccessGranularityLegalizationPass
    : public scheduler::impl::AccessGranularityLegalizationPassBase<
          AccessGranularityLegalizationPass> {
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

    // Collect first to avoid invalidating the walk while rewriting.
    llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
    module_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      transfers.push_back(transfer);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (auto transfer : transfers) {
      if (mlir::failed(legalizeTransfer(rewriter, transfer, resource_kinds))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createAccessGranularityLegalizationPass() {
  return std::make_unique<AccessGranularityLegalizationPass>();
}
