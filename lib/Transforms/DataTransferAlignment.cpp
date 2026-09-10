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
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "data-transfer-alignment"
#define DEBUG_TYPE PASS_NAME

namespace scheduler {
#define GEN_PASS_DEF_DATATRANSFERALIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler


namespace {

/// Identifies illegal ktdf.data_transfer ops and builds a per-pipeline
/// analysis tree used by the corrective rewrite phase.
class DataTransferLegality {
 public:

  struct PipelineAnalysis; // forward declaration, mutually recursive with StageAnalysis

  /// Describes a single ktdf.data_transfer op. Flags are set at analysis time
  /// so the rewrite phase needs no re-analysis.
  struct TransferStep {
    mlir::ktdf::DataTransferOp    transfer;              // the ktdf.data_transfer op
    bool                          isIllegal    = false;  // source memref has non-contiguous stride
    bool                          isDisplaced  = false;  // peer of an illegal transfer in the same pipeline
    bool                          needsSplat   = false;  // local->FIFO: splat scalar to vector
    bool                          needsExtract = false;  // FIFO->local: extract lane 0
  };

  /// Per-stage node. Exactly one of nestedPipeline or transfers is populated.
  struct StageAnalysis {
    mlir::ktdf::StageOp                  stage;
    std::unique_ptr<PipelineAnalysis>    nestedPipeline; // non-null if stage wraps a nested pipeline
    llvm::SmallVector<TransferStep>      transfers;      // empty when nestedPipeline is set
    mlir::scf::ForOp                     innermostLoop;  // innermost scf.for enclosing the transfers
  };

  /// Root node of a per-pipeline analysis tree. Shared parameters (targetDim,
  /// requiredSize) are stored here so all child nodes can read them directly.
  struct PipelineAnalysis {
    mlir::ktdf::PipelineOp                   pipeline;
    llvm::SmallVector<mlir::memref::AllocOp> allocs;          // staging buffer allocs in ktdf.private
    int64_t                                  targetDim    = -1; // dimension to target the size change with
    int64_t                                  requiredSize =  0; // elements per hardware word (wordBytes / elemBytes)
    llvm::SmallVector<StageAnalysis>         stages;            // one entry per ktdf.stage
  };

  /// Build a PipelineAnalysis for `pipeline`. Stages containing a nested
  /// ktdf.pipeline are recorded as stubs (PipelineOp handle only); fixPipeline
  /// calls analyzePipeline on them after applying outer-level corrections.
  PipelineAnalysis analyzePipeline(
      mlir::ktdf::PipelineOp pipeline,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    PipelineAnalysis pa{};
    pa.pipeline = pipeline;

    // Collect staging buffer allocs from the ktdf.private region.
    if (auto privateOp = pipeline.getPrivateOp()) {
      privateOp->walk([&](mlir::memref::AllocOp alloc) {
        pa.allocs.push_back(alloc);
      });
    }

    // Visit every stage.
    bool hasIllegal = false;
    for (auto stage : pipeline.getStages()) {
      pa.stages.push_back(analyzeStage(stage, resourceKinds, hasIllegal));
    }

    if (!hasIllegal)
      return pa;

    // Mark every non-illegal transfer as displaced.
    for (auto& sa : pa.stages) {
      for (auto& ts : sa.transfers) {
        if (!ts.isIllegal)
          ts.isDisplaced = true;
      }
    }

    // Derive targetDim and requiredSize from the first illegal transfer.
    // Whether we widen or shrink is determined by the arch spec granularity
    // for the transfer's source memory space:
    for (auto& sa : pa.stages) {
      for (auto& ts : sa.transfers) {
        if (!ts.isIllegal) continue;
        auto srcMemref = mlir::cast<mlir::MemRefType>(
            ts.transfer.getSource().getType());
        auto space     = srcMemref.getMemorySpace();
        auto wordBytes = getWordSize(ts.transfer, space, resourceKinds);
        auto elemBytes = scheduler::tryGetSizeInBytes(srcMemref.getElementType());
        if (!wordBytes || !elemBytes || *elemBytes == 0) break;

        auto granularities = getAccessGranularity(ts.transfer, space, resourceKinds);
        bool isSingleElement = false;
        if (granularities) {
          for (auto entry : *granularities) {
            uint64_t granBytes = entry.getSizeInWords() * *wordBytes;
            if (granBytes == *elemBytes) { isSingleElement = true; break; }
          }
        }

        if (isSingleElement) {
          // Shrink: element-addressable memory, collapse to one element.
          pa.targetDim    = srcMemref.getRank() - 1;
          pa.requiredSize = 1;
        } else {
          // Widen: word-granular memory, expand to full word.
          pa.targetDim    = srcMemref.getRank() - 2;
          pa.requiredSize = static_cast<int64_t>(*wordBytes / *elemBytes);
        }
        break;
      }
      if (pa.targetDim != -1) break;
    }

    return pa;
  }

private:

  /// Returns the single applicable unit kind for the stage enclosing `op`,
  /// stopping at any PipelineOp boundary. Returns nullptr if the stage has
  /// zero or more than one unit.
  mlir::Attribute getUnitKind(mlir::Operation* op) {
    mlir::Operation* cursor = op->getParentOp();
    while (cursor && !mlir::isa<mlir::ktdf::StageOp>(cursor)) {
      if (mlir::isa<mlir::ktdf::PipelineOp>(cursor)) return nullptr;
      cursor = cursor->getParentOp();
    }
    auto stage = mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(cursor);
    if (!stage) return nullptr;
    auto units = stage.getApplicableUnits();
    if (!units || units->size() != 1) return nullptr;
    return (*units)[0];
  }

  /// Returns the Load feature for the unit enclosing `op`.
  std::optional<mlir::ktdf_arch::feature::Load> getLoad(
      mlir::Operation* op,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    auto feat = resourceKinds.getFeature<mlir::ktdf_arch::feature::Load>(kind);
    if (!feat) return std::nullopt;
    return feat;
  }

  /// Returns the Store feature for the unit enclosing `op`.
  std::optional<mlir::ktdf_arch::feature::Store> getStore(
      mlir::Operation* op,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    auto feat = resourceKinds.getFeature<mlir::ktdf_arch::feature::Store>(kind);
    if (!feat) return std::nullopt;
    return feat;
  }

  /// Returns the word size in bytes for `load` accessing `space`.
  std::optional<uint64_t> getWordSize(
      mlir::ktdf_arch::feature::Load load, mlir::Attribute space) {
    auto map = load.getWordSize();
    if (!map) return std::nullopt;
    return map.getValue(space);
  }

  /// Returns the word size in bytes for `store` accessing `space`.
  std::optional<uint64_t> getWordSize(
      mlir::ktdf_arch::feature::Store store, mlir::Attribute space) {
    auto map = store.getWordSize();
    if (!map) return std::nullopt;
    return map.getValue(space);
  }

  /// Returns the access granularity list for `load` and `space`.
  std::optional<mlir::ktdf_arch::AccessGranularityListAttr>
  getAccessGranularity(mlir::ktdf_arch::feature::Load load, mlir::Attribute space) {
    auto list = load.getAccessGranularity(space);
    if (!list) return std::nullopt;
    return list;
  }

  /// Returns the access granularity list for `store` and `space`.
  std::optional<mlir::ktdf_arch::AccessGranularityListAttr>
  getAccessGranularity(mlir::ktdf_arch::feature::Store store, mlir::Attribute space) {
    auto list = store.getAccessGranularity(space);
    if (!list) return std::nullopt;
    return list;
  }

  /// Returns the word size in bytes for `op`'s unit accessing `space`.
  /// Tries load feature first, then store.
  std::optional<uint64_t> getWordSize(
      mlir::ktdf::DataTransferOp op,
      mlir::Attribute space,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    if (auto load = getLoad(op, resourceKinds))
      return getWordSize(*load, space);
    if (auto store = getStore(op, resourceKinds))
      return getWordSize(*store, space);
    return std::nullopt;
  }

  /// Returns the access granularity list for `op`'s unit accessing `space`.
  /// Tries load feature first, then store.
  std::optional<mlir::ktdf_arch::AccessGranularityListAttr>
  getAccessGranularity(
      mlir::ktdf::DataTransferOp op,
      mlir::Attribute space,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    if (auto load = getLoad(op, resourceKinds))
      return getAccessGranularity(*load, space);
    if (auto store = getStore(op, resourceKinds))
      return getAccessGranularity(*store, space);
    return std::nullopt;
  }

  /// Returns the innermost stride of `memref`. Returns 1 for identity/default
  /// row-major layouts. Returns failure() if the innermost stride is dynamic.
  llvm::FailureOr<int64_t> innermostStride(mlir::MemRefType memref) {
    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (mlir::failed(memref.getStridesAndOffset(strides, offset)) || strides.empty())
      return 1;  // no explicit layout, implicit row-major
    int64_t s = strides.back();
    if (mlir::ShapedType::isDynamic(s))
      return mlir::failure();
    return s;
  }

  /// Checks whether the source memref of a data_transfer is legal against the
  /// arch spec. Only source contiguity and granularity are checked.
  ///
  /// Returns true if legal, false if illegal but correctable, nullopt on
  /// hard failure (error already emitted).
  std::optional<bool> checkSourceMemRef(
      mlir::ktdf::DataTransferOp dt,
      mlir::MemRefType srcMemref,
      llvm::ArrayRef<int64_t> transferSizes,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    auto space = srcMemref.getMemorySpace();

    auto wordBytes    = getWordSize(dt, space, resourceKinds);
    auto granularities = getAccessGranularity(dt, space, resourceKinds);

    if (!wordBytes) {
      dt->emitError(PASS_NAME ": no word-size entry in the arch spec for "
                    "memory space ")
          << space;
      return std::nullopt;
    }

    // Element byte size must be statically known.
    auto elemBytes = scheduler::tryGetSizeInBytes(srcMemref.getElementType());
    if (!elemBytes) {
      dt->emitError(PASS_NAME ": element type has unknown size: ")
          << srcMemref.getElementType();
      return std::nullopt;
    }

    // Every granularity entry must cover a whole number of elements:
    // entry.sizeInWords * wordBytes must be divisible by elemBytes.
    if (granularities) {
      for (auto entry : *granularities) {
        uint64_t granBytes = entry.getSizeInWords() * *wordBytes;
        if (granBytes % *elemBytes != 0) {
          dt->emitError(PASS_NAME ": granularity entry size (")
              << granBytes << "B) is not a multiple of element size ("
              << *elemBytes << "B) for memory space " << space;
          return std::nullopt;
        }
      }
    }

    // Innermost stride must not be dynamic.
    auto stride = innermostStride(srcMemref);
    if (mlir::failed(stride)) {
      dt->emitError(PASS_NAME ": source memref has a dynamic innermost stride");
      return std::nullopt;
    }

    // Total bytes being transferred (product of all sizes × element size).
    int64_t totalElems = 1;
    for (int64_t s : transferSizes) totalElems *= s;
    uint64_t transferBytes = static_cast<uint64_t>(totalElems) * *elemBytes;

    // Find the matching granularity entry for this transfer size.
    if (granularities) {
      bool matched = false;
      uint64_t matchedGranBytes = 0;
      for (auto entry : *granularities) {
        uint64_t granBytes = entry.getSizeInWords() * *wordBytes;
        if (granBytes == transferBytes) {
          matched = true;
          matchedGranBytes = granBytes;
          break;
        }
      }
      if (!matched) {
        dt->emitError(PASS_NAME ": transfer size (")
            << transferBytes << "B) does not match any granularity entry "
              "for memory space "
            << space;
        return std::nullopt;
      }

      // Single-element granularity: element-addressable, any stride is legal.
      if (matchedGranBytes == *elemBytes)
        return true;
    }

    // Multi-element transfer: requires contiguous stride.
    // A single element is always legal regardless of stride.
    if (totalElems == 1 || *stride == 1)
      return true;  // legal

    // Non-contiguous multi-element transfer, correctable.
    return false;
  }

  /// Classifies a single ktdf.data_transfer and returns a TransferStep.
  /// FIFO sources are returned with all flags false; they are handled once
  /// requiredSize is known. Returns nullopt on hard failure (error already emitted).
  std::optional<TransferStep> classifyTransferStep(
      mlir::ktdf::DataTransferOp dt,
      const scheduler::arch_view::ResourceKinds& resourceKinds) {
    TransferStep ts{};
    ts.transfer = dt;

    // If the source is a FIFO there is no memref to check.
    if (dt.isSourceFifo())
      return ts;

    // Source is a memref, check it.
    auto srcMemref = mlir::cast<mlir::MemRefType>(dt.getSource().getType());
    auto sizes     = dt.getStaticSourceSizesArray();
    if (!sizes) {
      dt->emitError(PASS_NAME ": source transfer has dynamic sizes; "
                    "static sizes are required");
      return std::nullopt;
    }

    auto result = checkSourceMemRef(dt, srcMemref, *sizes, resourceKinds);
    if (!result.has_value())
      return std::nullopt;

    ts.isIllegal = !*result;
    return ts;
  }

  /// Analyses a single stage. If it contains a nested ktdf.pipeline, stores a
  /// stub in sa.nestedPipeline for fixPipeline to resolve. Otherwise classifies
  /// every ktdf.data_transfer into sa.transfers. Sets hasIllegal if any
  /// transfer is illegal; isDisplaced/needsSplat/needsExtract are set later.
  StageAnalysis analyzeStage(
      mlir::ktdf::StageOp stage,
      const scheduler::arch_view::ResourceKinds& resourceKinds,
      bool& hasIllegal) {
    StageAnalysis sa{};
    sa.stage = stage;

    // Check if the stage contains a nested pipeline.
    mlir::ktdf::PipelineOp nestedPipeline;
    stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp p) {
      nestedPipeline = p;
      return mlir::WalkResult::interrupt();
    });

    if (nestedPipeline) {
      PipelineAnalysis stub{};
      stub.pipeline = nestedPipeline;
      sa.nestedPipeline = std::make_unique<PipelineAnalysis>(std::move(stub));
      return sa;
    }

    // Find the innermost scf.for enclosing the transfers in this stage.
    stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp forOp) {
      sa.innermostLoop = forOp;
    });

    // Leaf stage: collect every data_transfer op.
    stage->walk([&](mlir::ktdf::DataTransferOp dt) {
      auto ts = classifyTransferStep(dt, resourceKinds);
      if (ts) {
        if (ts->isIllegal)
          hasIllegal = true;
        sa.transfers.push_back(*ts);
      }
    });

    return sa;
  }
};

/// Prints a TransferStep prefixed by `indent`.
static llvm::raw_ostream& printTransferStep(llvm::raw_ostream& os,
                                            const DataTransferLegality::TransferStep& ts,
                                            llvm::StringRef indent) {
  os << indent << "TransferStep{\n";
  os << indent << "  loc=";
  if (ts.transfer) os << ts.transfer->getLoc(); else os << "<unset>";
  os << "\n"
     << indent << "  isIllegal="    << ts.isIllegal    << "\n"
     << indent << "  isDisplaced="  << ts.isDisplaced  << "\n"
     << indent << "  needsSplat="   << ts.needsSplat   << "\n"
     << indent << "  needsExtract=" << ts.needsExtract << "\n"
     << indent << "}";
  return os;
}

// Forward declaration; printStageAnalysis and printPipelineAnalysis are mutually recursive.
static llvm::raw_ostream& printPipelineAnalysis(llvm::raw_ostream& os,
                                                const DataTransferLegality::PipelineAnalysis& pa,
                                                llvm::StringRef indent);

/// Prints a StageAnalysis prefixed by `indent`.
static llvm::raw_ostream& printStageAnalysis(llvm::raw_ostream& os,
                                             const DataTransferLegality::StageAnalysis& sa,
                                             llvm::StringRef indent) {
  os << indent << "StageAnalysis{stage=";
  if (sa.stage) os << sa.stage->getLoc(); else os << "<unset>";
  if (sa.innermostLoop) {
    os << "\n" << indent << "  innermostLoop=";
    sa.innermostLoop->print(os, mlir::OpPrintingFlags().skipRegions());
  }
  if (sa.nestedPipeline) {
    os << ", nestedPipeline=\n";
    printPipelineAnalysis(os, *sa.nestedPipeline, (indent + "  ").str());
  } else if (sa.transfers.empty()) {
    os << ", transfers=<empty>";
  } else {
    os << ", transfers=[\n";
    std::string tsIndent = (indent + "    ").str();
    for (const auto& ts : sa.transfers)
      printTransferStep(os, ts, tsIndent);
    os << "\n" << indent << "]";
  }
  os << "}";
  return os;
}

/// Prints a PipelineAnalysis prefixed by `indent`.
static llvm::raw_ostream& printPipelineAnalysis(llvm::raw_ostream& os,
                                                const DataTransferLegality::PipelineAnalysis& pa,
                                                llvm::StringRef indent) {
  os << indent << "PipelineAnalysis{\n";
  os << indent << "  pipeline=";
  if (pa.pipeline) {
    auto pipeline = pa.pipeline;  // mutable copy, PipelineOp is a pointer wrapper
    os << pipeline->getLoc();
    os << ", stages=" << pipeline.getNumStages();
  } else {
    os << "<unset>";
  }
  os << "\n";
  os << indent << "  targetDim=" << pa.targetDim << "\n";
  os << indent << "  requiredSize=" << pa.requiredSize << "\n";
  if (pa.allocs.empty()) {
    os << indent << "  allocs=<none>\n";
  } else {
    for (const auto& alloc : pa.allocs)
      os << indent << "  alloc=" << alloc->getLoc() << "\n";
  }
  if (pa.stages.empty()) {
    os << indent << "  stages=<empty>\n";
  } else {
    os << indent << "  stages=[\n";
    std::string saIndent = (indent + "    ").str();
    for (const auto& sa : pa.stages) {
      printStageAnalysis(os, sa, saIndent);
      os << "\n";
    }
    os << indent << "  ]\n";
  }
  os << indent << "}";
  return os;
}

/// Prints a PipelineAnalysis at zero indent for LDBG.
static llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const DataTransferLegality::PipelineAnalysis& pa) {
  return printPipelineAnalysis(os, pa, "");
}

} // namespace

namespace {

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* device = device_manager.getOrImportDevice();
    if (!device) {
      getOperation()->emitError(PASS_NAME ": failed to import device spec");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<scheduler::arch_view::ResourceKinds>(*device);

    llvm::SmallVector<DataTransferLegality::PipelineAnalysis> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(legality_.analyzePipeline(pipeline, resource_kinds));
          return mlir::WalkResult::skip();
        });
    for (const auto& pa : pipelines)
      LDBG(1) << pa;
  }
  
private:
  DataTransferLegality legality_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
//===----------------------------------------------------------------------===//
// widenAlloc
//
// If ts.transfer writes into a ct_local memref, traces back to the defining
// AllocOp and widens pa.targetDim to E (pa.requiredSize). Non-ct_local
// destinations are skipped. Missing AllocOp on a ct_local destination is an
// error.
//===----------------------------------------------------------------------===//
static void widenAlloc(const TransferStep& ts, const PipelineAnalysis& pa,
                       mlir::OpBuilder& builder) {
  const int64_t E = pa.requiredSize;

  // Only act when the transfer destination is a ct_local memref.
  auto dest_type =
      mlir::dyn_cast<mlir::MemRefType>(ts.transfer.getDestination().getType());
  if (!dest_type) return;

  auto mem_space = mlir::dyn_cast_or_null<mlir::ktdp::MemorySpaceAttr>(
      dest_type.getMemorySpace());
  if (!mem_space || mem_space.getValue() != mlir::ktdp::MemorySpace::ct_local)
    return;

  // Destination is ct_local — trace back to the defining AllocOp.
  // A ct_local buffer must be backed by a memref.alloc.
  mlir::Value base = ts.transfer.getDestination();
  while (auto view_like =
             mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
                 base.getDefiningOp()))
    base = view_like.getViewSource();

  auto alloc = mlir::dyn_cast_or_null<mlir::memref::AllocOp>(
      base.getDefiningOp());
  if (!alloc) {
    ts.transfer->emitError(
        "widenAlloc: ct_local destination has no backing AllocOp");
    return;
  }

  mlir::MemRefType orig_type = alloc.getType();
  llvm::SmallVector<int64_t> new_shape(orig_type.getShape());

  // Already widened — nothing to do.
  if (new_shape[pa.targetDim] == E) {
    LDBG(1) << "  widenAlloc: already widened, skipping";
    return;
  }

  // Widen targetDim to E, keeping the original flat ct_local layout.
  new_shape[pa.targetDim] = E;

  mlir::MemRefType new_type =
      mlir::MemRefType::get(new_shape, orig_type.getElementType(),
                            orig_type.getLayout(),
                            orig_type.getMemorySpace());

  builder.setInsertionPoint(alloc);
  auto new_alloc = mlir::memref::AllocOp::create(
      builder, alloc.getLoc(), new_type, alloc.getDynamicSizes());

  LDBG(1) << "  widenAlloc: " << orig_type << " → " << new_type;
  alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
  alloc.erase();
}

//===----------------------------------------------------------------------===//
// adjustLoopBounds
//
// Divides the upper bound of sa.targetDimFor by E (pa.requiredSize). Only
// the loop iterating over the dimension for this stage is adjusted —
// outer tile loops and other stages' loops are left untouched.
// Called once per StageAnalysis before visiting its transfers.
//===----------------------------------------------------------------------===//
static void adjustLoopBounds(const StageAnalysis& sa,
                              const PipelineAnalysis& pa,
                              mlir::OpBuilder& builder) {
  if (!sa.targetDimFor) return;

  mlir::scf::ForOp loop = sa.targetDimFor;
  builder.setInsertionPoint(loop);

  mlir::Value e_val = mlir::arith::ConstantIndexOp::create(
      builder, loop.getLoc(), pa.requiredSize).getResult();

  // new_ub = original_ub/E
  mlir::Value new_ub = mlir::arith::DivUIOp::create(
      builder, loop.getLoc(),
      loop.getUpperBound(), e_val).getResult();
  loop.setUpperBound(new_ub);

  LDBG(1) << "  adjustLoopBounds: divided ub by " << pa.requiredSize
          << " on targetDimFor at " << loop.getLoc();
}

//===----------------------------------------------------------------------===//
// rewriteTransferShape
//
// Replace ts.transfer with a new DataTransferOp where pa.targetDim is widened
// to E. All other dimensions are preserved. No hardcoding — pa.targetDim and
// pa.requiredSize drive the change entirely.
//===----------------------------------------------------------------------===//
static void rewriteTransferShape(const TransferStep& ts,
                                 const PipelineAnalysis& pa,
                                 mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult e_ofr =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), pa.requiredSize);

  // Copy existing sizes and update only pa.targetDim.
  auto new_src_sizes = op.getMixedSourceSizes();
  new_src_sizes[pa.targetDim] = e_ofr;

  auto new_dst_sizes = op.getMixedDestSizes();
  new_dst_sizes[pa.targetDim] = e_ofr;

  mlir::AffineMap src_map =
      op.isSourceMemRef() ? op.getSourceMapAttr().getValue() : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(),
      op.getSource(), src_map, op.getSourceIndices(), new_src_sizes,
      op.getDestination(), dst_map, op.getDestIndices(), new_dst_sizes);

  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());

  LDBG(1) << "  rewriteTransferShape: widened dim " << pa.targetDim
          << " to " << pa.requiredSize << ": " << new_op;
  op.erase();
}

// Shrink pa.targetDim to 1 and stamp transfer_mode as mode ("splat" or
// "extract"). All other dimensions are preserved, mirroring rewriteTransferShape.
static void rewriteTransferShrink(const TransferStep& ts,
                                  const PipelineAnalysis& pa,
                                  llvm::StringRef mode,
                                  mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult one =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1);

  // Only shrink targetDim, leave all other dimensions as-is.
  auto new_src_sizes = op.getMixedSourceSizes();
  if (pa.targetDim < (int64_t)new_src_sizes.size())
    new_src_sizes[pa.targetDim] = one;

  auto new_dst_sizes = op.getMixedDestSizes();
  if (pa.targetDim < (int64_t)new_dst_sizes.size())
    new_dst_sizes[pa.targetDim] = one;

  mlir::AffineMap src_map =
      op.isSourceMemRef() ? op.getSourceMapAttr().getValue() : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(),
      op.getSource(), src_map, op.getSourceIndices(), new_src_sizes,
      op.getDestination(), dst_map, op.getDestIndices(), new_dst_sizes);

  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());

  new_op->setDiscardableAttr(
      mlir::StringAttr::get(ctx, "transfer_mode"),
      mlir::StringAttr::get(ctx, mode));

  LDBG(1) << "  rewriteTransferShrink(" << mode << "): shrunk dim "
          << pa.targetDim << " to 1: " << new_op;
  op.erase();
}

//===----------------------------------------------------------------------===//
// insertLoopAroundPipeline
//
// Wrap the nested ktdf.pipeline in a new scf.for %col = 0 to E step 1.
// This is the column-element loop that is required so that
// transfers can index each column of the widened alloc
// independently.
//===----------------------------------------------------------------------===//
static void insertLoopAroundPipeline(PipelineAnalysis* nestedPA, int64_t E,
                                     mlir::OpBuilder& builder) {
  mlir::ktdf::PipelineOp nested_pipeline = nestedPA->pipeline;
  mlir::Location loc = nested_pipeline.getLoc();

  builder.setInsertionPoint(nested_pipeline);

  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value cE =
      mlir::arith::ConstantIndexOp::create(builder, loc, E).getResult();
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

  auto col_loop = mlir::scf::ForOp::create(builder, loc, c0, cE, c1);

  // Move the nested pipeline into the new loop's body (before the terminator).
  mlir::Block* loop_body = col_loop.getBody();
  nested_pipeline->moveBefore(loop_body, loop_body->getTerminator()->getIterator());

  LDBG(1) << "  insertLoopAroundPipeline: inserted col loop (E=" << E
          << ") around nested pipeline at " << loc;
}

//===----------------------------------------------------------------------===//
// fixPipeline  (depth-first recursive rewrite)
//
// Consumes the PipelineAnalysis tree and applies the corrective actions:
//   StageAnalysis level: insert col loop around nested pipeline, or visit
//                         TransferSteps directly
//   TransferStep level: widen ct_local alloc at targetDim to E,
//                         divide targetDimFor loop bound by E,
//                         rewrite transfer shape at targetDim to E,
//                         or shrink targetDim to 1 + stamp transfer_mode
//                         for FIFO transfers (splat/extract)
//===----------------------------------------------------------------------===//
static mlir::LogicalResult fixPipeline(PipelineAnalysis& pa,
                                       mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " E=" << pa.requiredSize;

  for (StageAnalysis& sa : pa.stages) {

    if (sa.nestedPipeline != nullptr) {
      // This stage contains a nested pipeline — insert the column loop and
      // recurse.
      insertLoopAroundPipeline(sa.nestedPipeline.get(), pa.requiredSize, builder);
      if (mlir::failed(fixPipeline(*sa.nestedPipeline, builder)))
        return mlir::failure();
      continue;
    }

    // Adjust the dimension loop bound for this stage before visiting
    // its transfers.
    adjustLoopBounds(sa, pa, builder);

    for (TransferStep& ts : sa.transfers) {
      if (ts.isIllegal || ts.isDisplaced) {
        // 1. Widen the ct_local destination alloc at targetDim.
        widenAlloc(ts, pa, builder);
        // 2. Rewrite the transfer sizes at targetDim to E.
        rewriteTransferShape(ts, pa, builder);
        continue;
      }

      if (ts.needsSplat) {
        rewriteTransferShrink(ts, pa, "splat", builder);
        continue;
      }

      if (ts.needsExtract) {
        rewriteTransferShrink(ts, pa, "extract", builder);
      }
    }
  }
  return mlir::success();
}

}  // namespace
