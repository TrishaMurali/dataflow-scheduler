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
    int64_t                                  targetDim    = -1; // offset from end (0 = innermost, 1 = second-from-end)
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
          // targetDim=0 means the innermost dimension (offset 0 from end).
          pa.targetDim    = 0;
          pa.requiredSize = 1;
        } else {
          // Widen: word-granular memory, expand to full word.
          // targetDim=1 means the second-to-last dimension (offset 1 from end).
          pa.targetDim    = 1;
          pa.requiredSize = static_cast<int64_t>(*wordBytes / *elemBytes);
        }
        break;
      }
      if (pa.targetDim != -1) break;
    }

    // Post-analysis fixup: set needsSplat on memref→FIFO transfers now that
    // requiredSize is known. classifyTransferStep returns early for FIFO
    // sources so this flag must be applied here.
    if (pa.requiredSize > 0) {
      for (auto& sa : pa.stages) {
        for (auto& ts : sa.transfers) {
          mlir::ktdf::DataTransferOp dt = ts.transfer;
          if (dt.isSourceMemRef() && dt.isDestFifo())
            ts.needsSplat = true;
        }
      }
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
  /// transfer is illegal; isDisplaced/needsSplat are set later.
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
    // PreOrder visits outer-to-inner; no interrupt() means the last
    // assignment wins, leaving innermostLoop as the innermost ForOp.
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

/// Widens any ct_local alloc backing a source or destination of `ts` at
/// pa.targetDim to E (pa.requiredSize). Non-ct_local sides are skipped.
static void widenAlloc(const DataTransferLegality::TransferStep& ts,
                       const DataTransferLegality::PipelineAnalysis& pa,
                       mlir::OpBuilder& builder) {
  const int64_t E = pa.requiredSize;

  // DataTransferOp is a pointer wrapper — copy it so we can call non-const
  // accessors without needing to drop the const on the TransferStep.
  mlir::ktdf::DataTransferOp dt = ts.transfer;

  // Tries to widen the alloc backing `val` if it is a ct_local memref owned
  // by this pipeline. Returns early silently for non-ct_local or foreign allocs.
  auto tryWiden = [&](mlir::Value val, mlir::Value stride_source) {
    auto memref_type = mlir::dyn_cast<mlir::MemRefType>(val.getType());
    if (!memref_type) return;

    auto mem_space = mlir::dyn_cast_or_null<mlir::ktdp::MemorySpaceAttr>(
        memref_type.getMemorySpace());
    if (!mem_space || mem_space.getKind() != mlir::ktdp::MemorySpaceKind::ct_local)
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
    int64_t alloc_dim = (int64_t)orig_type.getRank() - 1 - pa.targetDim;
    if (alloc_dim < 0 || alloc_dim >= (int64_t)orig_type.getRank()) {
      ts.transfer->emitError("widenAlloc: targetDim ")
          << alloc_dim << " is out of range for alloc rank "
          << orig_type.getRank();
      return;
    }

    // Already widened — nothing to do.
    if (new_shape[alloc_dim] == E) {
      LDBG(1) << "  widenAlloc: already widened, skipping";
      return;
    }

    // Widen the mapped alloc dim to E.
    new_shape[alloc_dim] = E;

    // Build a strided layout from the global-side (non-ct_local) operand's
    // strides. Skip if that operand is a FIFO or has no layout.
    auto global_type = mlir::dyn_cast<mlir::MemRefType>(stride_source.getType());
    mlir::MemRefLayoutAttrInterface new_layout = orig_type.getLayout();
    if (global_type) {
      llvm::SmallVector<int64_t, 4> src_strides;
      int64_t src_offset;
      if (mlir::succeeded(global_type.getStridesAndOffset(src_strides, src_offset))
          && (int64_t)src_strides.size() >= 2) {
        int64_t alloc_rank = (int64_t)new_shape.size();
        llvm::SmallVector<int64_t> new_strides(alloc_rank, 1);
        new_strides[alloc_rank - 1] = src_strides[src_strides.size() - 1];
        new_strides[alloc_rank - 2] = src_strides[src_strides.size() - 2];
        new_layout = mlir::StridedLayoutAttr::get(
            orig_type.getContext(), /*offset=*/0, new_strides);
      }
    }

    mlir::MemRefType new_type =
        mlir::MemRefType::get(new_shape, orig_type.getElementType(),
                              new_layout,
                              orig_type.getMemorySpace());

    // Rebuild dynamic-size operands, dropping the one for alloc_dim if it
    // was dynamic (it is now a constant).
    auto orig_dynamic = alloc.getDynamicSizes();
    llvm::SmallVector<mlir::Value> new_dynamic;
    unsigned dyn_idx = 0;
    for (int64_t i = 0; i < (int64_t)orig_type.getRank(); ++i) {
      if (mlir::ShapedType::isDynamic(orig_type.getShape()[i])) {
        if (mlir::ShapedType::isDynamic(new_shape[i]))
          new_dynamic.push_back(orig_dynamic[dyn_idx]);
        ++dyn_idx;
      }
    }

    builder.setInsertionPoint(alloc);
    auto new_alloc = mlir::memref::AllocOp::create(
        builder, alloc.getLoc(), new_type, new_dynamic);

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
  tryWiden(dt.getDestination(), dt.getSource());
  tryWiden(dt.getSource(), dt.getDestination());
}

/// Collapses sa.innermostLoop to a single iteration (ub=1) after verifying
/// that the loop's total_size operand equals E (pa.requiredSize). The upper
/// bound must be a ktdf.tiling.derive_size result; any other form is an error.
static mlir::LogicalResult adjustLoopBounds(
    const DataTransferLegality::StageAnalysis& sa,
    const DataTransferLegality::PipelineAnalysis& pa,
    mlir::OpBuilder& builder) {
  if (!sa.innermostLoop) return mlir::success();

  mlir::scf::ForOp loop = sa.innermostLoop;

  // After StageCoarseningPass the upper bound is always a
  // ktdf.tiling.derive_size result, not a bare constant.
  auto derive = mlir::dyn_cast_or_null<mlir::ktdf::TilingDeriveSizeOp>(
      loop.getUpperBound().getDefiningOp());
  if (!derive) {
    return loop->emitError(
        PASS_NAME ": innermost stage loop upper bound is not a "
        "ktdf.tiling.derive_size — cannot verify dimension size against E=")
        << pa.requiredSize;
  }

  // total_size is the constant full trip count for this dimension.
  // tile_sizes are still symbolic reserve_size placeholders at this stage.
  auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
      derive.getTotalSize().getDefiningOp());
  if (!cst) {
    return loop->emitError(
        PASS_NAME ": ktdf.tiling.derive_size total_size is not a constant "
        "— cannot verify dimension size against E=")
        << pa.requiredSize;
  }

  int64_t total_val = cst.value();
  if (total_val != pa.requiredSize) {
    return loop->emitError(PASS_NAME ": dimension total size (")
        << total_val << ") does not match required alignment size E="
        << pa.requiredSize
        << "; tiling and alignment constraints are inconsistent";
  }

  // Total size confirmed == E; collapse to one iteration.
  builder.setInsertionPoint(loop);
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loop.getLoc(), 1)
          .getResult();
  loop.setUpperBound(c1);

  LDBG(1) << "  adjustLoopBounds: total_size=" << total_val
          << " == E=" << pa.requiredSize
          << ", collapsed loop to ub=1 at " << loop.getLoc();
  return mlir::success();
}

/// Replaces ts.transfer with a new DataTransferOp with pa.targetDim widened
/// to E (pa.requiredSize). All other dimensions are preserved unchanged.
static void rewriteTransferShape(const DataTransferLegality::TransferStep& ts,
                                 const DataTransferLegality::PipelineAnalysis& pa,
                                 mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult e_ofr =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), pa.requiredSize);

  // Convert offset-from-end to absolute indices for src and dst size vectors.
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  int64_t src_rank  = (int64_t)new_src_sizes.size();
  int64_t dst_rank  = (int64_t)new_dst_sizes.size();
  int64_t src_dim   = src_rank - 1 - pa.targetDim;
  int64_t dst_dim   = dst_rank - 1 - pa.targetDim;
  if (src_dim >= 0 && src_dim < src_rank)
    new_src_sizes[src_dim] = e_ofr;
  if (dst_dim >= 0 && dst_dim < dst_rank)
    new_dst_sizes[dst_dim] = e_ofr;

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

/// Replaces ts.transfer with a new DataTransferOp with pa.targetDim shrunk to
/// 1 and transfer_mode set to `mode` ("splat" or "extract").
static void rewriteTransferShrink(const DataTransferLegality::TransferStep& ts,
                                  const DataTransferLegality::PipelineAnalysis& pa,
                                  llvm::StringRef mode,
                                  mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult one =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1);

  // Convert offset-from-end to absolute indices and shrink those dimensions.
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  int64_t src_rank  = (int64_t)new_src_sizes.size();
  int64_t dst_rank  = (int64_t)new_dst_sizes.size();
  int64_t src_dim   = src_rank - 1 - pa.targetDim;
  int64_t dst_dim   = dst_rank - 1 - pa.targetDim;
  if (src_dim >= 0 && src_dim < src_rank)
    new_src_sizes[src_dim] = one;
  if (dst_dim >= 0 && dst_dim < dst_rank)
    new_dst_sizes[dst_dim] = one;

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

  if (!mode.empty())
    new_op->setDiscardableAttr(
        mlir::StringAttr::get(ctx, "transfer_mode"),
        mlir::StringAttr::get(ctx, mode));

  LDBG(1) << "  rewriteTransferShrink(" << mode << "): shrunk dim "
          << pa.targetDim << " to 1: " << new_op;
  op.erase();
}

/// Wraps the nested ktdf.pipeline in a new scf.for loop from 0 to E (step 1)
/// so that transfers can index each element of the widened alloc independently.
static void insertLoopAroundPipeline(
    DataTransferLegality::PipelineAnalysis* nestedPA, int64_t E,
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

/// Applies corrective rewrites to the PipelineAnalysis tree: inserts a loop
/// around nested pipelines, then for each leaf stage adjusts the loop bound
/// and rewrites every transfer shape (illegal/displaced → widen; FIFO →
/// splat/extract shrink).
static mlir::LogicalResult fixPipeline(
    DataTransferLegality& legality,
    DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::ResourceKinds& resourceKinds,
    mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " E=" << pa.requiredSize;

  for (DataTransferLegality::StageAnalysis& sa : pa.stages) {

    if (sa.nestedPipeline != nullptr) {
      *sa.nestedPipeline = legality.analyzePipeline(
          sa.nestedPipeline->pipeline, resourceKinds);
      LDBG(1) << "  nested PipelineAnalysis:\n" << *sa.nestedPipeline;

      insertLoopAroundPipeline(sa.nestedPipeline.get(), pa.requiredSize, builder);
      if (mlir::failed(fixPipeline(legality, *sa.nestedPipeline, resourceKinds, builder)))
        return mlir::failure();
      continue;
    }

    // Adjust the dimension loop bound for this stage before visiting
    // its transfers. Fails if the tile size != E.
    if (mlir::failed(adjustLoopBounds(sa, pa, builder)))
      return mlir::failure();

    for (DataTransferLegality::TransferStep& ts : sa.transfers) {
      // FIFO-side transfers are checked first — splat/extract describe the
      // fundamental transfer kind and take priority over stride illegality.
      if (ts.needsSplat) {
        rewriteTransferShrink(ts, pa, "splat", builder);
        continue;
      }

      // FIFO→memref transfers are shrunk to size [1] with no transfer_mode
      // attr — the lowering handles single-element receive + store implicitly.
      if (ts.transfer.isSourceFifo() && ts.transfer.isDestMemRef()) {
        rewriteTransferShrink(ts, pa, "", builder);
        continue;
      }

      if (ts.isIllegal || ts.isDisplaced) {
        widenAlloc(ts, pa, builder);
        rewriteTransferShape(ts, pa, builder);
      }
    }
  }
  return mlir::success();
}

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

    mlir::OpBuilder builder(getOperation()->getContext());
    for (auto& pa : pipelines) {
      LDBG(1) << pa;
      if (pa.requiredSize == 0) continue;  // nothing to fix for this pipeline
      if (mlir::failed(fixPipeline(legality_, pa, resource_kinds, builder))) {
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