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
// DataTransferAlignment: Align ktdf.data_transfer ops to hardware requirements.
//
// Reads word size and access alignment constraints from the ktdf_arch.device
// spec and rewrites any transfers that violate those constraints, resizing
// staging buffers and transfer shapes until all transfers are legal.
//
// Ordering: After stage-coarsening + canonicalize. Before double-buffering.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDP.h"
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
  };

  /// Per-stage node. Exactly one of nestedPipeline or transfers is populated.
  struct StageAnalysis {
    mlir::ktdf::StageOp                  stage;
    std::unique_ptr<PipelineAnalysis>    nestedPipeline; // non-null if stage wraps a nested pipeline
    llvm::SmallVector<TransferStep>      transfers;      // empty when nestedPipeline is set
    mlir::scf::ForOp                     innermostLoop;  // innermost scf.for enclosing the transfers
  };

  /// Whether the pass needs to widen transfers to cover a full contiguous block,
  /// or shrink them to a single element (element-addressable memory).
  enum class AlignmentKind { Widen, Shrink };

  /// Root node of a per-pipeline analysis tree. Shared parameters
  /// (alignmentKind, alignDim, alignmentFactor) are stored here so all child
  /// nodes can read them directly.
  struct PipelineAnalysis {
    mlir::ktdf::PipelineOp                   pipeline;
    llvm::SmallVector<mlir::memref::AllocOp> allocs;              // staging buffer allocs in ktdf.private
    AlignmentKind                            alignmentKind   = AlignmentKind::Widen;
    int64_t                                  alignDim        = -1; // which dim to align, offset from end
    int64_t                                  alignmentFactor =  0; // factor to multiply the data transfer size by on alignDim
    int64_t                                  strideElems     =  0; // full contiguous block extent in elements
    llvm::SmallVector<StageAnalysis>         stages;              // one entry per ktdf.stage
  };

  /// Build a PipelineAnalysis for `pipeline`. Stages containing a nested
  /// ktdf.pipeline are recorded as stubs (PipelineOp handle only); fixPipeline
  /// calls analyzePipeline on them after applying outer-level corrections.
  ///
  /// When `inheritedStrides` is non-empty the legality check uses those stride
  /// values instead of reading them from the actual memref layout. This lets the
  /// outer pipeline's non-contiguous stride be visible to nested-pipeline
  /// analysis without touching any IR types.
  PipelineAnalysis analyzePipeline(
      mlir::ktdf::PipelineOp pipeline,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds,
      llvm::ArrayRef<int64_t> inheritedStrides = {}) {
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
      pa.stages.push_back(analyzeStage(stage, resourceKinds, hasIllegal,
                                       inheritedStrides));
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

    // Derive alignDim, alignmentFactor, and alignmentKind from the first illegal
    // transfer.
    //
    // The required transfer size is determined by the non-contiguous stride on
    // the source memref: to move the data the transfer requests, we must pull
    // a contiguous block of (innermostStride) elements from the source memory.
    // The arch spec granularity is then a validity check that this stride-derived
    // size is actually a legal transfer size for the hardware — it is not the
    // source of truth for the size.
    for (auto& sa : pa.stages) {
      for (auto& ts : sa.transfers) {
        if (!ts.isIllegal) continue;

        // Select the strided operand: prefer source if it is a non-unit-stride
        // memref; fall back to destination (store-back to strided global memory).
        // Only switch to destination when it is actually a memref.
        mlir::ktdf::DataTransferOp dt = ts.transfer;
        bool useDest = false;
        if (dt.isSourceMemRef()) {
          auto srcMrt = mlir::cast<mlir::MemRefType>(dt.getSource().getType());
          auto stride = innermostStride(srcMrt);
          if (!mlir::failed(stride) && *stride == 1 && dt.isDestMemRef())
            useDest = true;  // source is contiguous — illegality is on dest
        } else if (dt.isDestMemRef()) {
          useDest = true;  // source is a FIFO, destination carries the stride
        } else {
          continue;  // neither operand is a memref; nothing to derive here
        }

        auto stridedMemref = useDest
            ? mlir::cast<mlir::MemRefType>(dt.getDestination().getType())
            : mlir::cast<mlir::MemRefType>(dt.getSource().getType());
        auto space     = stridedMemref.getMemorySpace();
        auto elemBytes = scheduler::tryGetSizeInBytes(stridedMemref.getElementType());
        if (!elemBytes || *elemBytes == 0) break;

        auto granularities = getAccessGranularity(ts.transfer, space, resourceKinds);
        bool isSingleElement = false;
        if (granularities) {
          auto wordBytes = getWordSize(ts.transfer, space, resourceKinds);
          if (wordBytes) {
            for (auto entry : *granularities) {
              uint64_t granBytes = entry.getSizeInWords() * *wordBytes;
              if (granBytes == *elemBytes) { isSingleElement = true; break; }
            }
          }
        }

        if (isSingleElement) {
          // Shrink: element-addressable memory, collapse to one element.
          // alignDim=0 means the innermost dimension (offset 0 from end).
          pa.alignmentKind   = AlignmentKind::Shrink;
          pa.alignDim        = 0;
          pa.alignmentFactor = 1;
        } else {
          // Widen: the non-contiguous innermost stride tells us the total span
          // of the strided operand in elements. Each transfer iteration already
          // covers elemsPerIter (the innermost transfer size), so the number of
          // loop iterations — and hence the loop bound E — is stride / elemsPerIter.
          // Use inheritedStrides when set (nested pipeline case), otherwise
          // read the stride directly from the strided memref layout.
          int64_t strideElems = 1;
          if (!inheritedStrides.empty()) {
            strideElems = inheritedStrides.back();
          } else {
            auto stride = innermostStride(stridedMemref);
            if (!mlir::failed(stride))
              strideElems = *stride;
          }

          // Elements covered per loop iteration = the transfer size at alignDim
          // (offset 1 from the end, i.e. sizes[rank-2]).
          // E = strideElems / elemsPerIter gives the number of iterations needed
          // to cover one full contiguous block.
          int64_t elemsPerIter = 1;
          auto sizes = useDest ? dt.getStaticDestSizesArray()
                               : dt.getStaticSourceSizesArray();
          if (sizes) {
            // alignDim=1 → index from end = 1 → absolute index = rank-2
            int64_t rank = (int64_t)sizes->size();
            int64_t idx  = rank - 1 - 1;  // rank - 1 - alignDim(=1)
            if (idx >= 0 && idx < rank)
              elemsPerIter = std::max<int64_t>(1, (*sizes)[idx]);
          }

          // alignDim=1 means the second-to-last dimension (offset 1 from end).
          pa.alignmentKind   = AlignmentKind::Widen;
          pa.alignDim        = 1;
          pa.strideElems     = strideElems;
          pa.alignmentFactor = strideElems / elemsPerIter;
        }
        break;
      }
      if (pa.alignDim != -1) break;
    }

    // Post-analysis fixup: set needsSplat on memref→FIFO transfers now that
    // alignmentFactor is known. Only units with ktdf_arch.feature.simd { splat }
    // can legally broadcast a scalar element across a FIFO slot. If the unit
    // lacks that capability the transfer is unresolvable — emit an error.
    // classifyTransferStep returns early for FIFO sources so this flag must
    // be applied here.
    if (pa.alignmentFactor > 0) {
      for (auto& sa : pa.stages) {
        for (auto& ts : sa.transfers) {
          mlir::ktdf::DataTransferOp dt = ts.transfer;
          if (dt.isSourceMemRef() && dt.isDestFifo()) {
            if (canSplat(dt, resourceKinds)) {
              ts.needsSplat = true;
            } else {
              dt->emitError(PASS_NAME
                  ": memref→FIFO transfer requires splat but the enclosing "
                  "unit does not declare ktdf_arch.feature.simd { splat }");
            }
          }
        }
      }
    }

    return pa;
  }

private:

  /// Returns the single applicable unit kind for the stage enclosing `op`,
  /// stopping at any PipelineOp boundary. Returns nullptr if the stage has
  /// zero or more than one unit.
  mlir::ktdf_arch::KindAttr getUnitKind(mlir::Operation* op) {
    mlir::Operation* cursor = op->getParentOp();
    while (cursor && !mlir::isa<mlir::ktdf::StageOp>(cursor)) {
      if (mlir::isa<mlir::ktdf::PipelineOp>(cursor)) return nullptr;
      cursor = cursor->getParentOp();
    }
    auto stage = mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(cursor);
    if (!stage) return nullptr;
    auto units = stage.getApplicableUnits();
    if (!units || units->size() != 1) return nullptr;
    return mlir::dyn_cast<mlir::ktdf_arch::KindAttr>((*units)[0]);
  }

  /// Returns the Load feature for the unit enclosing `op`.
  std::optional<mlir::ktdf_arch::feature::Load> getLoad(
      mlir::Operation* op,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    auto feat = resourceKinds.getFeature<mlir::ktdf_arch::feature::Load>(kind);
    if (!feat) return std::nullopt;
    return feat;
  }

  /// Returns the Store feature for the unit enclosing `op`.
  std::optional<mlir::ktdf_arch::feature::Store> getStore(
      mlir::Operation* op,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    auto feat = resourceKinds.getFeature<mlir::ktdf_arch::feature::Store>(kind);
    if (!feat) return std::nullopt;
    return feat;
  }

  /// Returns true if the unit enclosing `op` has a SIMD feature with splat
  /// capability declared in the arch spec.
  bool canSplat(mlir::Operation* op,
                const mlir::ktdf_arch::ResourceKinds& resourceKinds) {
    auto kind = getUnitKind(op);
    if (!kind) return false;
    auto simd = resourceKinds.getFeature<mlir::ktdf_arch::feature::SIMD>(kind);
    return simd && simd.canSplat();
  }

  /// Returns the word size in bytes for the load unit accessing `space`.
  uint64_t getWordSize(
      mlir::ktdf_arch::feature::Load load, mlir::Attribute space) {
    return load.getWordSize(space);
  }

  /// Returns the word size in bytes for the store unit accessing `space`.
  uint64_t getWordSize(
      mlir::ktdf_arch::feature::Store store, mlir::Attribute space) {
    return store.getWordSize(space);
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
  /// Returns nullopt only when the unit has no Load or Store feature at all.
  /// Tries load feature first, then store.
  std::optional<uint64_t> getWordSize(
      mlir::ktdf::DataTransferOp op,
      mlir::Attribute space,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds) {
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
      const mlir::ktdf_arch::ResourceKinds& resourceKinds) {
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

  /// Checks whether a memref operand of a data_transfer is legal against the
  /// arch spec. Checks contiguity and granularity.
  ///
  /// When `inheritedStrides` is non-empty those values are used in place of the
  /// memref's actual layout strides for the contiguity check.
  ///
  /// Returns true if legal, false if illegal but correctable, nullopt on
  /// hard failure (error already emitted).
  std::optional<bool> checkMemRef(
      mlir::ktdf::DataTransferOp dt,
      mlir::MemRefType memref,
      llvm::ArrayRef<int64_t> transferSizes,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds,
      llvm::ArrayRef<int64_t> inheritedStrides = {}) {
    auto space = memref.getMemorySpace();

    auto wordBytes     = getWordSize(dt, space, resourceKinds);
    auto granularities = getAccessGranularity(dt, space, resourceKinds);

    // Element byte size must be statically known.
    auto elemBytes = scheduler::tryGetSizeInBytes(memref.getElementType());
    if (!elemBytes) {
      dt->emitError(PASS_NAME ": element type has unknown size: ")
          << memref.getElementType();
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

    // Use inherited strides (from an outer pipeline) when provided; otherwise
    // read the innermost stride from the memref layout.
    int64_t strideVal = 1;
    if (!inheritedStrides.empty()) {
      strideVal = inheritedStrides.back();
    } else {
      auto stride = innermostStride(memref);
      if (mlir::failed(stride)) {
        dt->emitError(PASS_NAME ": memref has a dynamic innermost stride");
        return std::nullopt;
      }
      strideVal = *stride;
    }

    // Total bytes being transferred (product of all sizes × element size).
    int64_t totalElems = 1;
    for (int64_t s : transferSizes) totalElems *= s;
    uint64_t transferBytes = static_cast<uint64_t>(totalElems) * *elemBytes;

    // Find a granularity entry that the transfer size is a multiple of.
    // A transfer of N bytes is legal if N is a positive integer multiple of
    // some entry's granularity.
    if (granularities) {
      bool matched = false;
      uint64_t matchedGranBytes = 0;
      for (auto entry : *granularities) {
        uint64_t granBytes = entry.getSizeInWords() * *wordBytes;
        if (granBytes != 0 && transferBytes % granBytes == 0) {
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
    if (totalElems == 1 || strideVal == 1)
      return true;  // legal

    // Non-contiguous multi-element transfer, correctable.
    return false;
  }

  /// Classifies a single ktdf.data_transfer and returns a TransferStep.
  /// FIFO sources are returned with all flags false; they are handled once
  /// alignmentFactor is known. Returns nullopt on hard failure (error already emitted).
  /// When `inheritedStrides` is non-empty it is forwarded to checkMemRef.
  std::optional<TransferStep> classifyTransferStep(
      mlir::ktdf::DataTransferOp dt,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds,
      llvm::ArrayRef<int64_t> inheritedStrides = {}) {
    TransferStep ts{};
    ts.transfer = dt;

    // Check a memref operand; returns false on hard failure (error emitted).
    auto checkOperand = [&](mlir::Value operand,
                            llvm::StringRef side,
                            std::optional<llvm::SmallVector<int64_t>> sizes)
        -> bool {
      if (!sizes) {
        dt->emitError(PASS_NAME ": ") << side << " transfer has dynamic sizes; "
                      "static sizes are required";
        return false;
      }
      auto memref = mlir::cast<mlir::MemRefType>(operand.getType());
      auto result = checkMemRef(dt, memref, *sizes, resourceKinds,
                                inheritedStrides);
      if (!result.has_value())
        return false;
      if (!*result)
        ts.isIllegal = true;
      return true;
    };

    if (dt.isSourceMemRef() &&
        !checkOperand(dt.getSource(), "source", dt.getStaticSourceSizesArray()))
      return std::nullopt;

    if (dt.isDestMemRef() &&
        !checkOperand(dt.getDestination(), "destination", dt.getStaticDestSizesArray()))
      return std::nullopt;

    return ts;
  }

  /// Analyses a single stage. If it contains a nested ktdf.pipeline, stores a
  /// stub in sa.nestedPipeline for fixPipeline to resolve. Otherwise classifies
  /// every ktdf.data_transfer into sa.transfers. Sets hasIllegal if any
  /// transfer is illegal; isDisplaced/needsSplat are set later.
  /// When `inheritedStrides` is non-empty it is forwarded to classifyTransferStep.
  StageAnalysis analyzeStage(
      mlir::ktdf::StageOp stage,
      const mlir::ktdf_arch::ResourceKinds& resourceKinds,
      bool& hasIllegal,
      llvm::ArrayRef<int64_t> inheritedStrides = {}) {
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
      // Also find the innermost loop enclosing the nested pipeline — it needs
      // to be collapsed to ub=1 just like leaf stage loops, so that the only
      // column iteration is the loop inserted by insertLoopAroundPipeline.
      stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp forOp) {
        if (!nestedPipeline->isAncestor(forOp))
          sa.innermostLoop = forOp;
      });
      return sa;
    }

    // Find the innermost scf.for enclosing the transfers in this stage.
    // PreOrder visits outer-to-inner; no interrupt() means the last
    // assignment wins, leaving innermostLoop as the innermost ForOp.
    stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp forOp) {
      sa.innermostLoop = forOp;
    });

    // Leaf stage: collect every data_transfer op.
    stage->walk([&](mlir::ktdf::DataTransferOp dt) {
      auto ts = classifyTransferStep(dt, resourceKinds, inheritedStrides);
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
  os << indent << "  alignmentKind="
     << (pa.alignmentKind == DataTransferLegality::AlignmentKind::Widen
             ? "Widen" : "Shrink") << "\n";
  os << indent << "  alignDim=" << pa.alignDim << "\n";
  os << indent << "  alignmentFactor=" << pa.alignmentFactor << "\n";
  os << indent << "  strideElems=" << pa.strideElems << "\n";
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

/// Returns true if `val` is a memref located in per-core ct_local memory.
static bool isPerCoreScratchpad(
    mlir::Value val, const scheduler::arch_view::MemoryTree& memory_tree) {
  auto mrt = mlir::dyn_cast<mlir::MemRefType>(val.getType());
  if (!mrt) return false;
  auto space = mrt.getMemorySpace();
  if (!space) return false;
  auto kind = mlir::dyn_cast<mlir::ktdf_arch::KindAttr>(space);
  if (!kind) return false;
  return memory_tree.isPerCoreScratchPadMemory(kind);
}

/// Widens any ct_local alloc backing a source or destination of `ts` at
/// pa.alignDim by pa.alignmentFactor. Non-ct_local sides are skipped.
static void widenAlloc(const DataTransferLegality::TransferStep& ts,
                       const DataTransferLegality::PipelineAnalysis& pa,
                       const scheduler::arch_view::MemoryTree& memory_tree,
                       mlir::OpBuilder& builder) {
  const int64_t alignmentFactor = pa.alignmentFactor;

  // DataTransferOp is a pointer wrapper — copy it so we can call non-const
  // accessors without needing to drop the const on the TransferStep.
  mlir::ktdf::DataTransferOp dt = ts.transfer;

  // Tries to widen the alloc backing `val` if it is a ct_local memref owned
  // by this pipeline. Returns early silently for non-ct_local or foreign allocs.
  auto tryWiden = [&](mlir::Value val) {
    if (!isPerCoreScratchpad(val, memory_tree))
      return;

    // ct_local buffers are exposed as ktdf.private results; follow the result
    // index into the private_yield operands to reach the backing memref.alloc.
    // Track private_op so its declared result type can be updated to match.
    mlir::Value base = val;
    mlir::ktdf::PrivateOp private_op;
    unsigned private_result_number = 0;
    if (auto pop = mlir::dyn_cast_or_null<mlir::ktdf::PrivateOp>(
            base.getDefiningOp())) {
      private_result_number =
          mlir::cast<mlir::OpResult>(base).getResultNumber();
      base = pop.getYieldOp().getOperand(private_result_number);
      private_op = pop;
    }

    auto alloc = mlir::dyn_cast_or_null<mlir::memref::AllocOp>(
        base.getDefiningOp());
    if (!alloc) {
      ts.transfer->emitError(
          "widenAlloc: ct_local operand has no backing AllocOp");
      return;
    }

    // Only widen allocs that belong to this pipeline's ktdf.private region.
    // Allocs from an outer pipeline are already handled and must not be touched.
    if (!llvm::is_contained(pa.allocs, alloc))
      return;

    mlir::MemRefType orig_type = alloc.getType();
    llvm::SmallVector<int64_t> new_shape(orig_type.getShape());

    // Convert offset-from-end to an absolute index into the alloc shape.
    int64_t alloc_dim = (int64_t)orig_type.getRank() - 1 - pa.alignDim;
    if (alloc_dim < 0 || alloc_dim >= (int64_t)orig_type.getRank()) {
      ts.transfer->emitError("widenAlloc: alignDim ")
          << alloc_dim << " is out of range for alloc rank "
          << orig_type.getRank();
      return;
    }

    llvm::SmallVector<mlir::Value> new_dynamic_sizes(alloc.getDynamicSizes());

    if (orig_type.getNumDynamicDims() > 0 && !new_dynamic_sizes.empty()) {
      builder.setInsertionPoint(alloc);
      mlir::Value c1 =
          mlir::arith::ConstantIndexOp::create(builder, alloc.getLoc(), 1)
              .getResult();
      new_dynamic_sizes.back() = c1;
    } else {
      if (new_shape[alloc_dim] == new_shape[alloc_dim] * alignmentFactor)
        return;

      if (new_shape[0] % alignmentFactor != 0) {
        ts.transfer->emitError("widenAlloc: outermost dimension (")
            << new_shape[0] << ") is not divisible by alignment factor "
            << alignmentFactor;
        return;
      }
      new_shape[0] = new_shape[0] / alignmentFactor;

      int64_t widened_dim = new_shape[alloc_dim] * alignmentFactor;
      int64_t innermost_dim = new_shape[(int64_t)orig_type.getRank() - 1];
      if (innermost_dim > 0 && widened_dim % innermost_dim != 0) {
        ts.transfer->emitError("widenAlloc: widened dimension (")
            << widened_dim << ") is not a multiple of innermost dimension ("
            << innermost_dim << ")";
        return;
      }
      new_shape[alloc_dim] = widened_dim;
    }

    // Use the default identity layout - the affine maps on the data_transfer
    // ops encode all access patterns.
    mlir::MemRefType new_type =
        mlir::MemRefType::get(new_shape, orig_type.getElementType(),
                              mlir::MemRefLayoutAttrInterface{},
                              orig_type.getMemorySpace());

    builder.setInsertionPoint(alloc);
    auto new_alloc = mlir::memref::AllocOp::create(
        builder, alloc.getLoc(), new_type, new_dynamic_sizes);

    // Keep the ktdf.private result type in sync; the verifier requires it to
    // match the private_yield operand type and the alloc type.
    if (private_op)
      private_op.getResult(private_result_number).setType(new_type);

    LDBG(1) << "  widenAlloc: " << orig_type << " → " << new_type;
    alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
    alloc.erase();
  };

  // Try both sides: dest-side ct_local (e.g. load into staging buffer) and
  // source-side ct_local (e.g. store out of staging buffer).
  tryWiden(dt.getDestination());
  tryWiden(dt.getSource());
}

/// Reduces sa.innermostLoop upper bound by pa.alignmentFactor after verifying
/// that the loop's total_size operand is divisible by pa.alignmentFactor.
static mlir::LogicalResult adjustLoopBounds(
    const DataTransferLegality::StageAnalysis& sa,
    const DataTransferLegality::PipelineAnalysis& pa,
    mlir::OpBuilder& builder) {
  if (!sa.innermostLoop) return mlir::success();

  mlir::scf::ForOp loop = sa.innermostLoop;

  // The upper bound is normally a ktdf.tiling.derive_size result (produced by
  // StageCoarseningPass). When the IR was not tiled it may instead be a bare
  // arith.constant.
  int64_t total_val = -1;
  auto derive = mlir::dyn_cast_or_null<mlir::ktdf::TilingDeriveSizeOp>(
      loop.getUpperBound().getDefiningOp());
  if (derive) {
    // total_size is the constant full trip count for this dimension.
    // tile_sizes are still symbolic reserve_size placeholders at this stage.
    auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
        derive.getTotalSize().getDefiningOp());
    if (!cst) {
      return loop->emitError(
            PASS_NAME ": ktdf.tiling.derive_size total_size is not a constant "
            "— cannot verify dimension size against alignmentFactor=")
            << pa.alignmentFactor;
    }
    total_val = cst.value();
  } else {
    auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
        loop.getUpperBound().getDefiningOp());
    if (!cst) {
      return loop->emitError(
            PASS_NAME ": innermost stage loop upper bound is neither a "
            "ktdf.tiling.derive_size nor a constant index "
            "— cannot verify dimension size against alignmentFactor=")
            << pa.alignmentFactor;
    }
    total_val = cst.value();
  }

  if (pa.alignmentFactor <= 0 || total_val % pa.alignmentFactor != 0) {
    return loop->emitError(PASS_NAME ": dimension total size (")
        << total_val << ") is not divisible by alignment factor "
        << pa.alignmentFactor
        << "; tiling and alignment constraints are inconsistent";
  }

  // Each widened transfer now covers alignmentFactor times as much data per
  // iteration, so the loop only needs total_val / alignmentFactor iterations.
  int64_t new_ub = total_val / pa.alignmentFactor;
  builder.setInsertionPoint(loop);
  mlir::Value new_ub_val =
      mlir::arith::ConstantIndexOp::create(builder, loop.getLoc(), new_ub)
          .getResult();
  loop.setUpperBound(new_ub_val);

  LDBG(1) << "  adjustLoopBounds: total_size=" << total_val
          << " / alignmentFactor=" << pa.alignmentFactor
          << " → new ub=" << new_ub << " at " << loop.getLoc();
  return mlir::success();
}

/// Replaces ts.transfer with a new DataTransferOp with pa.alignDim widened
/// Any explicit StridedLayoutAttr on a global (non-ct_local) memref operand
/// is stripped via a memref.cast to identity layout — the strided layout was
/// used by the analysis to detect illegality but must not appear on the
/// rewritten transfer.
static void rewriteTransferShape(const DataTransferLegality::TransferStep& ts,
                                 const DataTransferLegality::PipelineAnalysis& pa,
                                 const scheduler::arch_view::MemoryTree& memory_tree,
                                 mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  // Convert offset-from-end to absolute indices for src and dst size vectors.
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  int64_t src_rank  = (int64_t)new_src_sizes.size();
  int64_t dst_rank  = (int64_t)new_dst_sizes.size();
  int64_t src_dim   = src_rank - 1 - pa.alignDim;
  int64_t dst_dim   = dst_rank - 1 - pa.alignDim;

  auto getScaled = [&](mlir::OpFoldResult ofr) -> mlir::OpFoldResult {
    int64_t existing = mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(ofr)).getInt();
    return mlir::IntegerAttr::get(mlir::IndexType::get(ctx), existing * pa.alignmentFactor);
  };

  if (src_dim >= 0 && src_dim < src_rank)
    new_src_sizes[src_dim] = getScaled(new_src_sizes[src_dim]);
  if (dst_dim >= 0 && dst_dim < dst_rank)
    new_dst_sizes[dst_dim] = getScaled(new_dst_sizes[dst_dim]);

  mlir::AffineMap src_map =
      op.isSourceMemRef() ? op.getSourceMapAttr().getValue() : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  // Fix non-row-major strides on any global memref operand by
  // walking back to the ktdp.construct_memory_view and rewriting its
  // static_strides attribute to natural row-major values in place. The
  // strided layout was used during analysis to detect illegality;
  auto fixConstructStrides = [&](mlir::Value val) {
    auto mrt = mlir::dyn_cast<mlir::MemRefType>(val.getType());
    if (!mrt) return;
    if (isPerCoreScratchpad(val, memory_tree))
      return;
    if (!mlir::isa<mlir::StridedLayoutAttr>(mrt.getLayout()))
      return;

    // Walk up view-like ops to find the construct_memory_view.
    mlir::Value cursor = val;
    mlir::ktdp::ConstructMemoryViewOp construct;
    while (auto defOp = cursor.getDefiningOp()) {
      if (auto c = mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(defOp)) {
        construct = c;
        break;
      }
      if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(defOp))
        cursor = castOp.getSource();
      else if (auto mscastOp = mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(defOp))
        cursor = mscastOp.getSource();
      else if (auto ricastOp = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(defOp))
        cursor = ricastOp.getSource();
      else
        break;
    }
    if (!construct) return;

    // Compute row-major strides from the construct op's result shape.
    auto orig_mrt = mlir::cast<mlir::MemRefType>(construct.getResult().getType());
    auto shape = orig_mrt.getShape();
    int64_t rank = (int64_t)shape.size();
    llvm::SmallVector<int64_t> row_major(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i)
      row_major[i] = row_major[i + 1] * shape[i + 1];

    LDBG(1) << "  fixConstructStrides: rewriting strides on "
            << construct->getLoc() << " to row-major";
    construct.setStaticStridesAttr(
        mlir::DenseI64ArrayAttr::get(ctx, row_major));

    // Update the construct op's result type to carry the new row-major layout,
    // then propagate the updated type forward through all downstream view-like
    // ops (memory_space_cast, cast, reinterpret_cast) so the strided layout
    // doesn't linger in their result types.
    auto new_layout = mlir::StridedLayoutAttr::get(
        ctx, /*offset=*/mlir::ShapedType::kDynamic, row_major);
    auto new_construct_type = mlir::MemRefType::get(
        shape, orig_mrt.getElementType(), new_layout,
        orig_mrt.getMemorySpace());
    construct.getResult().setType(new_construct_type);

    // Walk forward through uses, updating each view-like op's result type.
    llvm::SmallVector<mlir::Value> worklist = {construct.getResult()};
    while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      for (mlir::Operation* user : v.getUsers()) {
        mlir::Value new_val;
        mlir::MemRefType src_type =
            mlir::dyn_cast<mlir::MemRefType>(v.getType());
        if (!src_type) continue;

        if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(user)) {
          // Propagate the new layout into the cast result type.
          auto res_type = mlir::dyn_cast<mlir::MemRefType>(
              castOp.getResult().getType());
          if (!res_type) continue;
          auto updated = mlir::MemRefType::get(
              res_type.getShape(), res_type.getElementType(),
              src_type.getLayout(), res_type.getMemorySpace());
          castOp.getResult().setType(updated);
          new_val = castOp.getResult();
        } else if (auto mscastOp =
                       mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(user)) {
          auto res_type = mlir::dyn_cast<mlir::MemRefType>(
              mscastOp.getResult().getType());
          if (!res_type) continue;
          auto updated = mlir::MemRefType::get(
              res_type.getShape(), res_type.getElementType(),
              src_type.getLayout(), res_type.getMemorySpace());
          mscastOp.getResult().setType(updated);
          new_val = mscastOp.getResult();
        } else {
          continue;
        }
        worklist.push_back(new_val);
      }
    }
  };

  fixConstructStrides(op.getSource());
  fixConstructStrides(op.getDestination());

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(),
      op.getSource(), src_map, op.getSourceIndices(), new_src_sizes,
      op.getDestination(), dst_map, op.getDestIndices(), new_dst_sizes);

  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());

  LDBG(1) << "  rewriteTransferShape: scaled dim " << pa.alignDim
          << " by factor " << pa.alignmentFactor << ": " << new_op;
  op.erase();
}

/// Replaces ts.transfer with a new DataTransferOp sized [1,1,1,1] with
/// transfer_mode set to `mode` ("splat") or empty for FIFO→memref.
/// The two innermost indices on any ct_local memref operand are replaced with
/// the row and column IVs from the nested scf.for loops inserted by
/// insertLoopAroundPipeline, giving direct [0, 0, %row, %col] addressing.
static void rewriteTransferShrink(const DataTransferLegality::TransferStep& ts,
                                  const DataTransferLegality::PipelineAnalysis& pa,
                                  const scheduler::arch_view::MemoryTree& memory_tree,
                                  llvm::StringRef mode,
                                  mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult one =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1);

  // Splat: 1 source element broadcast to fill the FIFO (E elements).
  // Collapse the source to [1,1,1,1] — one scalar element —
  // and keep the FIFO destination at its natural size [E] so the hardware
  // knows to broadcast that scalar across the full slot.
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  for (auto& s : new_src_sizes) s = one;
  // Leave new_dst_sizes unchanged — the FIFO slot size is already correct.

  // Find the two loop IVs inserted by insertLoopAroundPipeline. Walking out
  // past the enclosing ktdf.pipeline(s) we expect to hit the inner scf.for
  // (col) first, then the outer scf.for (row).
  mlir::Value row_iv, col_iv;
  mlir::Operation* cursor = op->getParentOp();
  while (cursor) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(cursor)) {
      cursor = cursor->getParentOp();
      continue;
    }
    if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(cursor)) {
      if (!col_iv) {
        col_iv = for_op.getInductionVar();
        LDBG(1) << "  rewriteTransferShrink: found col_iv at " << for_op->getLoc();
      } else {
        row_iv = for_op.getInductionVar();
        LDBG(1) << "  rewriteTransferShrink: found row_iv at " << for_op->getLoc();
        break;
      }
    }
    cursor = cursor->getParentOp();
  }
  if (!row_iv || !col_iv)
    LDBG(1) << "  rewriteTransferShrink: row_iv=" << (bool)row_iv
            << " col_iv=" << (bool)col_iv << " — applyRowCol will no-op";

  // For ct_local memref operands, build a fresh affine map and index list.
  // The load (source) buffer is row-major: index [0, 0, %row, %col].
  // The store (destination) buffer is transposed (column-major): index
  // [0, 0, %col, %row] — row and col are swapped to match the word layout.
  auto applyRowCol = [&](mlir::Value memref_val,
                         mlir::AffineMap& map,
                         llvm::SmallVector<mlir::Value>& indices,
                         bool transpose) {
    if (!row_iv || !col_iv) return;
    if (!isPerCoreScratchpad(memref_val, memory_tree))
      return;
    auto mrt = mlir::cast<mlir::MemRefType>(memref_val.getType());

    int64_t rank = (int64_t)mrt.getRank();
    if (rank < 2) return;

    // d0 = first IV, d1 = second IV.
    // For load: (row, col) -> [0,0,row,col].
    // For store (transposed): (col, row) -> [0,0,col,row] i.e. swap the IVs.
    mlir::Value first_iv  = transpose ? col_iv : row_iv;
    mlir::Value second_iv = transpose ? row_iv : col_iv;

    llvm::SmallVector<mlir::AffineExpr> results(rank,
        mlir::getAffineConstantExpr(0, ctx));
    results[rank - 2] = mlir::getAffineDimExpr(0, ctx);
    results[rank - 1] = mlir::getAffineDimExpr(1, ctx);
    map     = mlir::AffineMap::get(/*dims=*/2, /*syms=*/0, results, ctx);
    indices = {first_iv, second_iv};
  };

  auto new_src_indices =
      llvm::SmallVector<mlir::Value>(op.getSourceIndices());
  auto new_dst_indices =
      llvm::SmallVector<mlir::Value>(op.getDestIndices());

  mlir::AffineMap src_map =
      op.isSourceMemRef() ? op.getSourceMapAttr().getValue() : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  applyRowCol(op.getSource(),      src_map, new_src_indices, /*transpose=*/false);
  applyRowCol(op.getDestination(), dst_map, new_dst_indices, /*transpose=*/true);

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(),
      op.getSource(), src_map, new_src_indices, new_src_sizes,
      op.getDestination(), dst_map, new_dst_indices, new_dst_sizes);

  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());

  if (!mode.empty())
    new_op->setDiscardableAttr(
        mlir::StringAttr::get(ctx, "transfer_mode"),
        mlir::StringAttr::get(ctx, mode));

  LDBG(1) << "  rewriteTransferShrink(" << mode << "): shrunk dim "
          << pa.alignDim << " to 1: " << new_op;
  op.erase();
}

/// Wraps the nested ktdf.pipeline in two nested scf.for loops over [0, bound)
/// so that each element of the widened alloc block can be addressed
/// directly via %row and %col IVs without any index arithmetic.
static void insertLoopAroundPipeline(
    DataTransferLegality::PipelineAnalysis* nestedPA, int64_t bound,
    mlir::OpBuilder& builder) {
  mlir::ktdf::PipelineOp nested_pipeline = nestedPA->pipeline;
  mlir::Location loc = nested_pipeline.getLoc();

  builder.setInsertionPoint(nested_pipeline);

  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value cBound =
      mlir::arith::ConstantIndexOp::create(builder, loc, bound).getResult();
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

  // Outer loop: rows [0, bound)
  auto row_loop = mlir::scf::ForOp::create(builder, loc, c0, cBound, c1);
  mlir::Block* row_body = row_loop.getBody();

  // Inner loop: cols [0, bound) — inserted inside the row loop body.
  builder.setInsertionPoint(row_body, row_body->getTerminator()->getIterator());
  auto col_loop = mlir::scf::ForOp::create(builder, loc, c0, cBound, c1);

  // Move the nested pipeline into the col loop body.
  mlir::Block* col_body = col_loop.getBody();
  nested_pipeline->moveBefore(col_body, col_body->getTerminator()->getIterator());

  LDBG(1) << "  insertLoopAroundPipeline: inserted row/col loops (bound=" << bound
          << ") around nested pipeline at " << loc;
}

/// Applies corrective rewrites to the PipelineAnalysis tree: inserts a loop
/// around nested pipelines, then for each leaf stage adjusts the loop bound
/// and rewrites every transfer shape (illegal/displaced → widen; FIFO →
/// splat shrink).
static mlir::LogicalResult fixPipeline(
    DataTransferLegality& legality,
    DataTransferLegality::PipelineAnalysis& pa,
    const mlir::ktdf_arch::ResourceKinds& resourceKinds,
    const scheduler::arch_view::MemoryTree& memoryTree,
    mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " alignmentFactor=" << pa.alignmentFactor
          << " strideElems=" << pa.strideElems;

  for (DataTransferLegality::StageAnalysis& sa : pa.stages) {

    if (sa.nestedPipeline != nullptr) {
      // Collapse the tile loop enclosing the nested pipeline to ub=1 so the
      // only column iteration is the element loop inserted below.
      if (mlir::failed(adjustLoopBounds(sa, pa, builder)))
        return mlir::failure();

      // Pass the outer pipeline's strideElems as an inherited stride so that
      // the nested pipeline's transfers see the full non-contiguous stride and are
      // classified correctly (isIllegal / isDisplaced / needsSplat) without
      // any IR types being modified.
      llvm::SmallVector<int64_t, 1> inheritedStrides = {pa.strideElems};
      *sa.nestedPipeline = legality.analyzePipeline(
          sa.nestedPipeline->pipeline, resourceKinds, inheritedStrides);
      LDBG(1) << "  nested PipelineAnalysis:\n" << *sa.nestedPipeline;

      insertLoopAroundPipeline(sa.nestedPipeline.get(), pa.strideElems, builder);
      if (mlir::failed(fixPipeline(legality, *sa.nestedPipeline, resourceKinds,
                                   memoryTree, builder)))
        return mlir::failure();
      continue;
    }

    // Adjust the dimension loop bound for this stage before visiting
    // its transfers.
    if (mlir::failed(adjustLoopBounds(sa, pa, builder)))
      return mlir::failure();

    for (DataTransferLegality::TransferStep& ts : sa.transfers) {
      if (pa.alignmentKind == DataTransferLegality::AlignmentKind::Widen) {
        // Outer pipeline: widen staging allocs and bulk transfer shapes
        if (ts.isIllegal || ts.isDisplaced) {
          widenAlloc(ts, pa, memoryTree, builder);
          rewriteTransferShape(ts, pa, memoryTree, builder);
        }
      } else {
        // Nested / inner pipeline: shrink transfers to scalar access with IV indexing
        if (ts.needsSplat) {
          rewriteTransferShrink(ts, pa, memoryTree, "splat", builder);
        } else if (ts.transfer.isSourceFifo() && ts.transfer.isDestMemRef()) {
          rewriteTransferShrink(ts, pa, memoryTree, "", builder);
        }
      }
    }
  }
  return mlir::success();
}

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    llvm::errs() << "[" PASS_NAME "] running on: "
                 << getOperation()->getName() << "\n";
    LDBG(1) << "========= " PASS_NAME " =========";

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* device = device_manager.getOrImportDevice();
    if (!device) {
      getOperation()->emitError(PASS_NAME ": failed to import device spec");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);
    auto& memory_tree =
        device_manager.getOrCreateView<scheduler::arch_view::MemoryTree>(*device);

    llvm::SmallVector<DataTransferLegality::PipelineAnalysis> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(legality_.analyzePipeline(pipeline, resource_kinds));
          return mlir::WalkResult::skip();
        });

    mlir::OpBuilder builder(getOperation()->getContext());
    for (auto& pa : pipelines) {
      LDBG(1) << pa;
      if (pa.alignmentFactor == 0) continue;  // nothing to fix for this pipeline
      if (mlir::failed(fixPipeline(legality_, pa, resource_kinds, memory_tree,
                                   builder))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  DataTransferLegality legality_;
};

} // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
