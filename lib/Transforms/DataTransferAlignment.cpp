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

#include <algorithm>
#include <memory>
#include <numeric>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
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

static bool isIdentity(llvm::ArrayRef<int64_t> perm) {
  return llvm::all_of(llvm::enumerate(perm), [](auto p) {
    return p.value() == static_cast<int64_t>(p.index());
  });
}

static llvm::SmallVector<int64_t> invertPermutation(
    llvm::ArrayRef<int64_t> perm) {
  llvm::SmallVector<int64_t> inverse(perm.size());
  for (auto [i, p] : llvm::enumerate(perm)) inverse[p] = i;
  return inverse;
}

/// Size of the contiguous (unit-stride) dim of `memref`; failure if there is
/// none or its size is dynamic. Size-1 dims are skipped: their stride is
/// arbitrary.
static mlir::FailureOr<int64_t> contiguousDimSize(mlir::MemRefType memref) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(memref.getStridesAndOffset(strides, offset)))
    return mlir::failure();
  for (auto [stride, size] : llvm::zip(strides, memref.getShape()))
    if (stride == 1 && size != 1 && !mlir::ShapedType::isDynamic(size))
      return size;
  return mlir::failure();
}

/// Index of the contiguous (unit-stride) dim of `memref`; failure if there is
/// none or its size is dynamic. Size-1 dims are skipped: their stride is
/// arbitrary.
static mlir::FailureOr<int64_t> contiguousDimIndex(mlir::MemRefType memref) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(memref.getStridesAndOffset(strides, offset)))
    return mlir::failure();
  for (auto [idx, p] : llvm::enumerate(llvm::zip(strides, memref.getShape()))) {
    auto [stride, size] = p;
    if (stride == 1 && size != 1 && !mlir::ShapedType::isDynamic(size))
      return static_cast<int64_t>(idx);
  }
  return mlir::failure();
}

/// Logical dims sorted by descending stride; empty for identity, rank 0,
/// dynamic strides or a failed stride query.
static llvm::SmallVector<int64_t> derivePermutation(mlir::MemRefType memref) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(memref.getStridesAndOffset(strides, offset)) ||
      strides.empty() || llvm::any_of(strides, mlir::ShapedType::isDynamic))
    return {};

  llvm::SmallVector<int64_t> perm(strides.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    return strides[a] > strides[b];
  });
  if (isIdentity(perm)) return {};
  return perm;
}

/// Composes permutation, returns empty vector if result is identity.
static llvm::SmallVector<int64_t> composePermutations(
    llvm::ArrayRef<int64_t> src_perm, llvm::ArrayRef<int64_t> dst_perm) {
  if (dst_perm.empty()) return llvm::to_vector(src_perm);
  llvm::SmallVector<int64_t> inv_dst = invertPermutation(dst_perm);
  llvm::SmallVector<int64_t> result = inv_dst;
  if (!src_perm.empty()) {
    if (src_perm.size() != dst_perm.size()) return llvm::to_vector(src_perm);
    for (auto [i, s] : llvm::enumerate(src_perm)) result[i] = inv_dst[s];
  }
  if (isIdentity(result)) return {};
  return result;
}

/// result[i] = values[perm[i]]; an empty `perm` returns `values` unchanged.
template <typename T>
static llvm::SmallVector<T> applyPermutation(llvm::ArrayRef<int64_t> perm,
                                             llvm::ArrayRef<T> values) {
  if (perm.empty()) return llvm::SmallVector<T>(values.begin(), values.end());
  llvm::SmallVector<T> result;
  result.reserve(perm.size());
  for (int64_t src : perm) result.push_back(values[src]);
  return result;
}

/// Dim of a transfer operand of rank `rank` that is widened by the alignment
/// factor, given the pipeline's stride-derived permutation `perm` (logical dims
/// sorted by descending stride). The widened
/// logical dim is the one that is contiguous in the source, `perm.back()`. A
/// global source is described in memory order, so there it is wherever the last
/// logical dim sits in that order. Staging buffers carry extra leading
/// loop dims, so `rank` may exceed `perm.size()`.
/// `inv_perm` must be the precomputed inverse of `perm` (from
/// PipelineAnalysis).
static int64_t widenedDim(llvm::ArrayRef<int64_t> perm,
                          llvm::ArrayRef<int64_t> inv_perm, int64_t rank,
                          bool in_memory_order) {
  int64_t leading = rank - static_cast<int64_t>(perm.size());
  return leading + (in_memory_order ? inv_perm.back() : perm.back());
}

/// Identifies illegal ktdf.data_transfer ops and builds a per-pipeline
/// analysis tree used by the corrective rewrite phase.
class DataTransferLegality {
 public:
  struct PipelineAnalysis;  // forward declaration, mutually recursive with
                            // StageAnalysis

  /// Per-stage node. Exactly one of nested_pipeline or transfers is populated.
  struct StageAnalysis {
    mlir::ktdf::StageOp stage;
    mlir::scf::ForOp innermost_loop;  // loop to shrink; null if stage has none
    std::unique_ptr<PipelineAnalysis>
        nested_pipeline;  // non-null if stage wraps a nested pipeline
    llvm::SmallVector<mlir::ktdf::DataTransferOp>
        transfers;  // empty when nested_pipeline is set
  };

  /// Root node of a per-pipeline analysis tree. Shared parameters
  /// (alignment_factor, innermost_size, permutation) are stored here so all
  /// child nodes can read them directly.
  struct PipelineAnalysis {
    mlir::ktdf::PipelineOp pipeline;
    llvm::DenseSet<mlir::memref::AllocOp>
        allocs;  // staging buffer allocs in ktdf.private
    int64_t alignment_factor =
        0;  // factor to multiply the data transfer size by on the widened dim
    int64_t innermost_size = 0;  // size of the contiguous dim, in elements
    llvm::SmallVector<int64_t> permutation;  // stride-derived index permutation
    llvm::SmallVector<int64_t>
        inv_permutation;                      // cached inverse of permutation
    llvm::SmallVector<StageAnalysis> stages;  // one entry per ktdf.stage
  };

  /// Build a PipelineAnalysis tree for `pipeline`, recursing into nested
  /// ktdf.pipeline ops once the outer alignment is known. Read-only.
  ///
  /// When `inherited_strides` is non-empty the legality check uses those stride
  /// values instead of reading them from the actual memref layout. This lets
  /// the outer pipeline's non-contiguous stride be visible to nested-pipeline
  /// analysis without touching any IR types.
  PipelineAnalysis analyzePipeline(
      mlir::ktdf::PipelineOp pipeline,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      llvm::ArrayRef<int64_t> inherited_strides = {},
      llvm::ArrayRef<int64_t> inherited_permutation = {}) {
    PipelineAnalysis pa{};
    pa.pipeline = pipeline;

    // Collect staging buffer allocs from the ktdf.private region.
    if (auto private_op = pipeline.getPrivateOp()) {
      private_op->walk(
          [&](mlir::memref::AllocOp alloc) { pa.allocs.insert(alloc); });
    }

    // Visit every stage.
    mlir::ktdf::DataTransferOp first_illegal;
    for (auto stage : pipeline.getStages()) {
      pa.stages.push_back(analyzeStage(stage, resource_kinds, first_illegal,
                                       inherited_strides));
    }

    if (!first_illegal) return pa;

    deriveAlignment(pa, first_illegal, resource_kinds, inherited_strides,
                    inherited_permutation);
    if (pa.alignment_factor == 0) return pa;

    // Cache the inverse permutation once so callers don't recompute it
    // per-transfer.
    if (!pa.permutation.empty())
      pa.inv_permutation = invertPermutation(pa.permutation);

    // Now that innermost_size is known, pick the correct loop per stage.
    fixupStageLoops(pa);

    if (mlir::failed(validateFifoTransfers(pa, resource_kinds))) return pa;

    // Nested pipelines see the outer non-contiguous stride and permutation.
    for (auto& sa : pa.stages) {
      if (!sa.nested_pipeline) continue;
      *sa.nested_pipeline =
          analyzePipeline(sa.nested_pipeline->pipeline, resource_kinds,
                          {pa.innermost_size}, pa.permutation);
    }

    return pa;
  }

  /// Returns true if the unit enclosing `op` has a SIMD feature with splat
  /// capability declared in the arch spec.
  bool canSplat(mlir::Operation* op,
                const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto kind = getUnitKind(op);
    if (!kind) return false;
    auto simd = resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(kind);
    return simd && simd.canSplat();
  }

  /// Returns true if the unit enclosing `op` has a Store feature with a 2-byte
  /// access_granularity entry for ct_local, meaning it supports extract stores.
  bool canExtract(mlir::Operation* op,
                  const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto kind = getUnitKind(op);
    if (!kind) return false;
    auto space =
        mlir::cast<mlir::MemRefType>(mlir::cast<mlir::ktdf::DataTransferOp>(op)
                                         .getDestination()
                                         .getType())
            .getMemorySpace();
    auto info = accessInfoFrom<mlir::ktdf_arch::feature::Store>(kind, space,
                                                                resource_kinds);
    if (!info || !info->granularities) return false;
    return llvm::any_of(info->granularities, [&](auto entry) {
      return entry.getSizeInWords() * info->word_bytes == 2;
    });
  }

 private:
  /// Word size and access granularities of a unit for one memory space.
  struct AccessInfo {
    uint64_t word_bytes;
    mlir::ktdf_arch::AccessGranularityListAttr granularities;  // may be null
  };

  /// Derives alignment_factor, innermost_size and permutation on `pa` from the
  /// first illegal transfer `dt`.
  ///
  /// The required transfer size is determined by the non-contiguous stride on
  /// the strided memref: to move the data the transfer requests, we must pull
  /// a contiguous block of (innermostStride) elements from the source memory.
  /// The arch spec granularity only decides Shrink (single element) vs Widen.
  void deriveAlignment(PipelineAnalysis& pa, mlir::ktdf::DataTransferOp dt,
                       const mlir::ktdf_arch::ResourceKinds& resource_kinds,
                       llvm::ArrayRef<int64_t> inherited_strides,
                       llvm::ArrayRef<int64_t> inherited_permutation) {
    auto operand = selectStridedOperand(dt);
    if (!operand) return;  // neither operand is a memref
    auto [strided_memref, use_dest] = *operand;
    auto space = strided_memref.getMemorySpace();
    auto elem_bytes =
        scheduler::tryGetSizeInBytes(strided_memref.getElementType());
    if (!elem_bytes || *elem_bytes == 0) return;

    auto info = getAccessInfo(dt, space, resource_kinds);
    bool is_single_element = false;
    if (info && info->granularities) {
      is_single_element = llvm::any_of(info->granularities, [&](auto entry) {
        return entry.getSizeInWords() * info->word_bytes == *elem_bytes;
      });
    } else if (!inherited_strides.empty()) {
      is_single_element = true;
    }

    // Shrink: element-addressable memory, collapse to one element.
    if (is_single_element) {
      pa.alignment_factor = 1;
      pa.permutation = inherited_permutation.empty()
                           ? derivePipelinePermutation(pa, dt)
                           : llvm::to_vector(inherited_permutation);
      return;
    }

    // Widen: the size of the memref's contiguous dim is the block to cover;
    // each iteration covers elems_per_iter, so the factor is
    // innermost_size / elems_per_iter.
    auto contiguous_size = contiguousDimSize(strided_memref);
    if (mlir::failed(contiguous_size)) return;
    int64_t innermost_size = *contiguous_size;

    // Elements covered per iteration = transfer size at the contiguous dim.
    int64_t elems_per_iter = 1;
    auto sizes = use_dest ? dt.getStaticDestSizesArray()
                          : dt.getStaticSourceSizesArray();
    auto c_idx = contiguousDimIndex(strided_memref);
    if (sizes && !mlir::failed(c_idx) &&
        static_cast<size_t>(*c_idx) < sizes->size())
      elems_per_iter = std::max<int64_t>(1, (*sizes)[*c_idx]);

    pa.innermost_size = innermost_size;
    pa.alignment_factor = innermost_size / elems_per_iter;

    pa.permutation = inherited_permutation.empty()
                         ? derivePipelinePermutation(pa, dt)
                         : llvm::to_vector(inherited_permutation);
  }

  /// Picks the memref operand carrying the stride: the source unless it is
  /// contiguous and the destination is also a memref. Returns the memref and
  /// whether it is the destination; nullopt if neither operand is a memref.
  std::optional<std::pair<mlir::MemRefType, bool>> selectStridedOperand(
      mlir::ktdf::DataTransferOp dt) {
    if (!dt.isSourceMemRef() && !dt.isDestMemRef()) return std::nullopt;
    bool use_dest = !dt.isSourceMemRef();
    if (!use_dest && dt.isDestMemRef()) {
      auto stride = innermostStride(
          mlir::cast<mlir::MemRefType>(dt.getSource().getType()));
      use_dest = !mlir::failed(stride) && *stride == 1;
    }
    mlir::Value operand = use_dest ? dt.getDestination() : dt.getSource();
    return std::pair{mlir::cast<mlir::MemRefType>(operand.getType()), use_dest};
  }

  /// Permutation from the first strided global memref seen on each side of
  /// the pipeline's transfers, falling back to `dt`'s own operands.
  llvm::SmallVector<int64_t> derivePipelinePermutation(
      const PipelineAnalysis& pa, mlir::ktdf::DataTransferOp dt) {
    auto strided_or_null = [](mlir::Value v) -> mlir::MemRefType {
      auto mrt = mlir::cast<mlir::MemRefType>(v.getType());
      return mlir::isa<mlir::StridedLayoutAttr>(mrt.getLayout())
                 ? mrt
                 : mlir::MemRefType{};
    };
    mlir::MemRefType global_src, global_dst;
    for (const StageAnalysis& sa : pa.stages) {
      for (mlir::ktdf::DataTransferOp op : sa.transfers) {
        if (!global_src && op.isSourceMemRef())
          global_src = strided_or_null(op.getSource());
        if (!global_dst && op.isDestMemRef())
          global_dst = strided_or_null(op.getDestination());
      }
    }
    if (!global_src && dt.isSourceMemRef())
      global_src = mlir::cast<mlir::MemRefType>(dt.getSource().getType());
    if (!global_dst && dt.isDestMemRef())
      global_dst = mlir::cast<mlir::MemRefType>(dt.getDestination().getType());

    auto src_perm = global_src ? derivePermutation(global_src)
                               : llvm::SmallVector<int64_t>{};
    auto dst_perm = global_dst ? derivePermutation(global_dst)
                               : llvm::SmallVector<int64_t>{};
    return composePermutations(src_perm, dst_perm);
  }

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

  /// Reads word size and granularities from feature `Feature` of `kind`.
  template <typename Feature>
  std::optional<AccessInfo> accessInfoFrom(
      mlir::ktdf_arch::KindAttr kind, mlir::Attribute space,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto feat = resource_kinds.getFeature<Feature>(kind);
    if (!feat) return std::nullopt;
    return AccessInfo{feat.getWordSize(space),
                      feat.getAccessGranularity(space)};
  }

  /// Access info for `op`'s unit and `space`; Load feature first, then Store.
  /// Returns nullopt only when the unit has neither feature.
  std::optional<AccessInfo> getAccessInfo(
      mlir::Operation* op, mlir::Attribute space,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    if (auto load = accessInfoFrom<mlir::ktdf_arch::feature::Load>(
            kind, space, resource_kinds))
      return load;
    return accessInfoFrom<mlir::ktdf_arch::feature::Store>(kind, space,
                                                           resource_kinds);
  }

  /// Returns the innermost stride of `memref`. Returns 1 for identity/default
  /// row-major layouts. Returns failure() if the innermost stride is dynamic.
  llvm::FailureOr<int64_t> innermostStride(mlir::MemRefType memref) {
    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (mlir::failed(memref.getStridesAndOffset(strides, offset)) ||
        strides.empty())
      return 1;  // no explicit layout, implicit row-major
    int64_t s = strides.back();
    if (mlir::ShapedType::isDynamic(s)) return mlir::failure();
    return s;
  }

  /// Bytes of the first granularity entry that `transfer_bytes` is a positive
  /// multiple of, or nullopt if none match.
  std::optional<uint64_t> matchGranularity(const AccessInfo& info,
                                           uint64_t transfer_bytes) {
    for (auto entry : info.granularities) {
      uint64_t gran_bytes = entry.getSizeInWords() * info.word_bytes;
      if (gran_bytes != 0 && transfer_bytes % gran_bytes == 0)
        return gran_bytes;
    }
    return std::nullopt;
  }

  /// Checks whether a memref operand of a data_transfer is legal against the
  /// arch spec. Checks contiguity and granularity.
  ///
  /// When `inherited_strides` is non-empty those values are used in place of
  /// the memref's actual layout strides for the contiguity check.
  ///
  /// Returns true if legal, false if illegal but correctable, nullopt on
  /// hard failure (error already emitted).
  std::optional<bool> checkMemRef(
      mlir::ktdf::DataTransferOp dt, mlir::MemRefType memref,
      llvm::ArrayRef<int64_t> transfer_sizes,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    auto space = memref.getMemorySpace();
    auto info = getAccessInfo(dt, space, resource_kinds);
    bool has_granularities = info && info->granularities;

    // Element byte size must be statically known.
    auto elem_bytes = scheduler::tryGetSizeInBytes(memref.getElementType());
    if (!elem_bytes) {
      dt->emitError(PASS_NAME ": element type has unknown size: ")
          << memref.getElementType();
      return std::nullopt;
    }

    // Use inherited strides (from an outer pipeline) when provided; otherwise
    // read the innermost stride from the memref layout.
    int64_t stride_val = 1;
    if (!inherited_strides.empty()) {
      stride_val = inherited_strides.back();
    } else {
      auto stride = innermostStride(memref);
      if (mlir::failed(stride)) {
        dt->emitError(PASS_NAME ": memref has a dynamic innermost stride");
        return std::nullopt;
      }
      stride_val = *stride;
    }

    // Total bytes being transferred (product of all sizes × element size).
    int64_t total_elems = llvm::product_of(transfer_sizes);
    uint64_t transfer_bytes = static_cast<uint64_t>(total_elems) * *elem_bytes;

    if (has_granularities) {
      auto matched = matchGranularity(*info, transfer_bytes);
      if (!matched) {
        dt->emitError(PASS_NAME ": transfer size (")
            << transfer_bytes
            << "B) does not match any granularity entry "
               "for memory space "
            << space;
        return std::nullopt;
      }

      // Single-element granularity: element-addressable, any stride is legal.
      if (*matched == *elem_bytes) return true;
    }

    // Multi-element transfer: requires contiguous stride.
    // A single element is always legal regardless of stride.
    if (total_elems == 1 || stride_val == 1) return true;  // legal

    // Non-contiguous multi-element transfer, correctable.
    return false;
  }

  /// Returns whether `dt` is illegal (has a correctable, non-contiguous memref
  /// operand), or nullopt on hard failure (error already emitted). When
  /// `inherited_strides` is non-empty it is forwarded to checkMemRef.
  std::optional<bool> classifyTransfer(
      mlir::ktdf::DataTransferOp dt,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    bool is_illegal = false;

    // Check a memref operand; returns false on hard failure (error emitted).
    auto check_operand =
        [&](mlir::Value operand, llvm::StringRef side,
            std::optional<llvm::SmallVector<int64_t>> sizes) -> bool {
      if (!sizes) {
        dt->emitError(PASS_NAME ": ") << side
                                      << " transfer has dynamic sizes; "
                                         "static sizes are required";
        return false;
      }
      auto memref = mlir::cast<mlir::MemRefType>(operand.getType());
      auto result =
          checkMemRef(dt, memref, *sizes, resource_kinds, inherited_strides);
      if (!result.has_value()) return false;
      if (!*result) is_illegal = true;
      return true;
    };

    if (dt.isSourceMemRef() && !check_operand(dt.getSource(), "source",
                                              dt.getStaticSourceSizesArray()))
      return std::nullopt;

    if (dt.isDestMemRef() && !check_operand(dt.getDestination(), "destination",
                                            dt.getStaticDestSizesArray()))
      return std::nullopt;

    return is_illegal;
  }

  /// Reports FIFO transfers whose unit cannot splat / extract. Read-only.
  /// Returns failure() if any diagnostic was emitted so the caller can
  /// propagate pass failure.
  mlir::LogicalResult validateFifoTransfers(
      const PipelineAnalysis& pa,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    mlir::LogicalResult result = mlir::success();
    for (const auto& sa : pa.stages) {
      for (mlir::ktdf::DataTransferOp dt : sa.transfers) {
        if (dt.isSourceMemRef() && dt.isDestFifo() &&
            !canSplat(dt, resource_kinds)) {
          dt->emitError(
              PASS_NAME
              ": memref→FIFO transfer requires splat but the enclosing "
              "unit does not declare ktdf_arch.feature.simd { splat }");
          result = mlir::failure();
        } else if (dt.isSourceFifo() && dt.isDestMemRef() &&
                   !canExtract(dt, resource_kinds)) {
          dt->emitError(PASS_NAME
                        ": FIFO→memref transfer requires (extract) store "
                        "but the enclosing unit does not declare a sub-element "
                        "access_granularity entry for ct_local");
          result = mlir::failure();
        }
      }
    }
    return result;
  }

  /// Returns all scf.for ops inside `stage` that do not lie inside a nested
  /// ktdf.pipeline, in walk order.
  static llvm::SmallVector<mlir::scf::ForOp> collectStageLoops(
      mlir::ktdf::StageOp stage) {
    llvm::SmallVector<mlir::scf::ForOp> loops;
    stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
      // Don't descend into a nested pipeline (but do visit the stage itself).
      if (mlir::isa<mlir::ktdf::PipelineOp>(op) && op != stage.getOperation())
        return mlir::WalkResult::skip();
      if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(op))
        loops.push_back(for_op);
      return mlir::WalkResult::advance();
    });
    return loops;
  }

  /// Returns the loop from `loops` whose total_size equals `target_size`,
  /// or the last loop as a fallback.
  static mlir::scf::ForOp pickLoopByTotalSize(
      llvm::ArrayRef<mlir::scf::ForOp> loops, int64_t target_size) {
    if (loops.empty()) return {};
    for (mlir::scf::ForOp loop : loops) {
      mlir::Value bound = loop.getUpperBound();
      auto derive = bound.getDefiningOp<mlir::ktdf::TilingDeriveSizeOp>();
      auto cst = (derive ? derive.getTotalSize() : bound)
                     .getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (cst && cst.value() == target_size) return loop;
    }
    return loops.back();
  }

  /// Analyses a single stage. If it contains a nested ktdf.pipeline, stores a
  /// stub in sa.nested_pipeline for analyzePipeline to fill in. Otherwise
  /// records every ktdf.data_transfer in sa.transfers and sets `first_illegal`
  /// to the first illegal one if still unset. When `inherited_strides` is
  /// non-empty it is forwarded to classifyTransfer.
  /// sa.innermost_loop is filled in by a second pass in analyzePipeline once
  /// innermost_size is known.
  StageAnalysis analyzeStage(
      mlir::ktdf::StageOp stage,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      mlir::ktdf::DataTransferOp& first_illegal,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    StageAnalysis sa{};
    sa.stage = stage;

    // Check if the stage contains a nested pipeline.
    mlir::ktdf::PipelineOp nested_pipeline;
    stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp p) {
      nested_pipeline = p;
      return mlir::WalkResult::interrupt();
    });

    if (nested_pipeline) {
      PipelineAnalysis stub{};
      stub.pipeline = nested_pipeline;
      sa.nested_pipeline = std::make_unique<PipelineAnalysis>(std::move(stub));
      return sa;
    }

    // Leaf stage: collect every data_transfer op.
    stage->walk([&](mlir::ktdf::DataTransferOp dt) {
      auto is_illegal = classifyTransfer(dt, resource_kinds, inherited_strides);
      if (!is_illegal) return;
      if (*is_illegal && !first_illegal) first_illegal = dt;
      sa.transfers.push_back(dt);
    });

    return sa;
  }

  /// Second pass: once innermost_size is known, pick the stage loop whose
  /// total_size matches it. Called from analyzePipeline after deriveAlignment.
  static void fixupStageLoops(PipelineAnalysis& pa) {
    for (StageAnalysis& sa : pa.stages) {
      if (sa.nested_pipeline) continue;
      auto loops = collectStageLoops(sa.stage);
      sa.innermost_loop = pickLoopByTotalSize(loops, pa.innermost_size);
    }
  }
};

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

/// A ct_local buffer: the backing alloc and, when exposed through a
/// ktdf.private result, the op and result index whose type must track it.
struct StagingAlloc {
  mlir::memref::AllocOp alloc;
  mlir::ktdf::PrivateOp private_op;
  unsigned result_number = 0;
};

/// Follows a ktdf.private result into its private_yield operand to reach the
/// backing memref.alloc. Returns null alloc if there is none.
static StagingAlloc resolveStagingAlloc(mlir::Value val) {
  StagingAlloc staging;
  if (auto pop =
          mlir::dyn_cast_or_null<mlir::ktdf::PrivateOp>(val.getDefiningOp())) {
    staging.private_op = pop;
    staging.result_number = mlir::cast<mlir::OpResult>(val).getResultNumber();
    val = pop.getYieldOp().getOperand(staging.result_number);
  }
  staging.alloc =
      mlir::dyn_cast_or_null<mlir::memref::AllocOp>(val.getDefiningOp());
  return staging;
}

/// Shape of `type` with `widen_dim` scaled by `factor`. The outermost dim is
/// divided by `factor` unless the alloc has dynamic sizes.
static mlir::FailureOr<llvm::SmallVector<int64_t>> widenedShape(
    mlir::ktdf::DataTransferOp dt, mlir::MemRefType type, int64_t factor,
    int64_t widen_dim, bool has_dynamic_sizes) {
  int64_t rank = type.getRank();
  if (rank < 2 || widen_dim < 0 || widen_dim >= rank)
    return dt->emitError("widenAlloc: invalid widen_dim ")
           << widen_dim << " for rank " << rank;

  llvm::SmallVector<int64_t> shape(type.getShape());
  if (!has_dynamic_sizes) {
    if (shape[0] % factor != 0)
      return dt->emitError("widenAlloc: outermost dimension (")
             << shape[0] << ") is not divisible by alignment factor " << factor;
    shape[0] /= factor;
  }

  shape[widen_dim] *= factor;
  return shape;
}

/// Widens any ct_local alloc backing a source or destination of `dt` by
/// pa.alignment_factor on the dim derived from the pipeline permutation.
/// Non-ct_local sides are skipped.
static void widenAlloc(mlir::ktdf::DataTransferOp dt,
                       const DataTransferLegality::PipelineAnalysis& pa,
                       const scheduler::arch_view::MemoryTree& memory_tree,
                       mlir::OpBuilder& builder) {
  // Widens the alloc backing `val` if it is a ct_local memref owned by this
  // pipeline; silently skips non-ct_local and foreign (already handled) allocs.
  auto try_widen = [&](mlir::Value val) {
    if (!isPerCoreScratchpad(val, memory_tree)) return;

    StagingAlloc staging = resolveStagingAlloc(val);
    mlir::memref::AllocOp alloc = staging.alloc;
    if (!alloc) {
      dt->emitError("widenAlloc: ct_local operand has no backing AllocOp");
      return;
    }
    if (!pa.allocs.contains(alloc)) return;

    mlir::MemRefType orig_type = alloc.getType();
    llvm::SmallVector<mlir::Value> new_dynamic_sizes(alloc.getDynamicSizes());
    bool has_dynamic_sizes =
        orig_type.getNumDynamicDims() > 0 && !new_dynamic_sizes.empty();

    // The widened dim in the alloc is derived from the pipeline permutation
    // in logical (non-memory) order; leading dynamic dims shift the index.
    int64_t wdim = widenedDim(pa.permutation, pa.inv_permutation,
                              orig_type.getRank(), false);
    auto new_shape = widenedShape(dt, orig_type, pa.alignment_factor, wdim,
                                  has_dynamic_sizes);
    if (mlir::failed(new_shape)) return;

    builder.setInsertionPoint(alloc);
    if (has_dynamic_sizes) {
      // Quick fix: the dim to collapse is the one whose tile was pinned to
      // alignment_factor by fixReserveSize (a constant of that value), not
      // necessarily the last one. Fall back to the last dynamic size.
      size_t collapse_idx = new_dynamic_sizes.size() - 1;
      for (size_t i = new_dynamic_sizes.size(); i-- > 0;) {
        auto cst =
            new_dynamic_sizes[i].getDefiningOp<mlir::arith::ConstantIndexOp>();
        if (cst && cst.value() == pa.alignment_factor) {
          collapse_idx = i;
          break;
        }
      }
      new_dynamic_sizes[collapse_idx] =
          mlir::arith::ConstantIndexOp::create(builder, alloc.getLoc(), 1)
              .getResult();
    }

    // Use the default identity layout - the affine maps on the data_transfer
    // ops encode all access patterns.
    mlir::MemRefType new_type = mlir::MemRefType::get(
        *new_shape, orig_type.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, orig_type.getMemorySpace());
    auto new_alloc = mlir::memref::AllocOp::create(builder, alloc.getLoc(),
                                                   new_type, new_dynamic_sizes);

    // Keep the ktdf.private result type in sync; the verifier requires it to
    // match the private_yield operand type and the alloc type.
    if (staging.private_op)
      staging.private_op.getResult(staging.result_number).setType(new_type);

    LDBG(1) << "  widenAlloc: " << orig_type << " → " << new_type;
    alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
    alloc.erase();
  };

  // Try both sides: dest-side ct_local (e.g. load into staging buffer) and
  // source-side ct_local (e.g. store out of staging buffer).
  try_widen(dt.getDestination());
  try_widen(dt.getSource());
}

/// Constant trip count of `loop`. The upper bound is normally a
/// ktdf.tiling.derive_size result (produced by StageCoarseningPass) whose
/// total_size is the full trip count; untiled IR has a bare arith.constant.
static mlir::FailureOr<int64_t> getLoopTotalSize(mlir::scf::ForOp loop,
                                                 int64_t alignment_factor) {
  mlir::Value bound = loop.getUpperBound();
  auto derive = bound.getDefiningOp<mlir::ktdf::TilingDeriveSizeOp>();
  auto cst = (derive ? derive.getTotalSize() : bound)
                 .getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!cst)
    return loop->emitError(PASS_NAME
                           ": innermost stage loop upper bound is "
                           "neither a ktdf.tiling.derive_size "
                           "with constant total_size nor a "
                           "constant index — cannot verify "
                           "dimension size against "
                           "alignment_factor=")
           << alignment_factor;
  return cst.value();
}

/// Replaces the symbolic reserve_size tile placeholder of a derive_size bound
/// with the constant `alignment_factor`.
static void fixReserveSize(mlir::scf::ForOp loop, int64_t alignment_factor,
                           mlir::OpBuilder& builder) {
  auto derive =
      loop.getUpperBound().getDefiningOp<mlir::ktdf::TilingDeriveSizeOp>();
  if (!derive || derive.getTileSizes().empty()) return;
  auto reserve_op = derive.getTileSizes()
                        .back()
                        .getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
  if (!reserve_op) return;
  builder.setInsertionPoint(reserve_op);
  auto cst_tile = mlir::arith::ConstantIndexOp::create(
      builder, reserve_op.getLoc(), alignment_factor);
  reserve_op.replaceAllUsesWith(cst_tile.getResult());
  reserve_op.erase();
}

/// Reduces the upper bound of `loop` by pa.alignment_factor after verifying
/// that the loop's total_size operand is divisible by it.
static mlir::LogicalResult adjustLoopBound(
    mlir::scf::ForOp loop, const DataTransferLegality::PipelineAnalysis& pa,
    mlir::OpBuilder& builder) {
  auto total = getLoopTotalSize(loop, pa.alignment_factor);
  if (mlir::failed(total)) return mlir::failure();
  int64_t total_val = *total;

  if (pa.alignment_factor <= 0 || total_val % pa.alignment_factor != 0) {
    return loop->emitError(PASS_NAME ": dimension total size (")
           << total_val << ") is not divisible by alignment factor "
           << pa.alignment_factor
           << "; tiling and alignment constraints are inconsistent";
  }

  fixReserveSize(loop, pa.alignment_factor, builder);

  // Each widened transfer covers alignment_factor times as much data per
  // iteration, so the loop only needs total_val / alignment_factor iterations.
  int64_t new_ub = total_val / pa.alignment_factor;
  builder.setInsertionPoint(loop);
  mlir::Value new_ub_val =
      mlir::arith::ConstantIndexOp::create(builder, loop.getLoc(), new_ub)
          .getResult();
  loop.setUpperBound(new_ub_val);

  LDBG(1) << "  adjustLoopBounds: total_size=" << total_val
          << " / alignment_factor=" << pa.alignment_factor
          << " → new ub=" << new_ub << " at " << loop.getLoc();
  return mlir::success();
}

/// Adjusts every stage's innermost loop recorded on `pa` exactly once.
static mlir::LogicalResult adjustLoopBounds(
    const DataTransferLegality::PipelineAnalysis& pa,
    mlir::OpBuilder& builder) {
  for (const DataTransferLegality::StageAnalysis& sa : pa.stages) {
    if (!sa.innermost_loop) continue;
    if (mlir::failed(adjustLoopBound(sa.innermost_loop, pa, builder)))
      return mlir::failure();
  }
  return mlir::success();
}

/// `map` with its results reordered so result j is the old result perm[j].
/// Used to turn logical coordinates into the coordinates of a view whose dims
/// were permuted into memory order. Unchanged if `perm` is empty or does not
/// match the result count.
static mlir::AffineMap permuteMapResults(mlir::AffineMap map,
                                         llvm::ArrayRef<int64_t> perm) {
  if (perm.empty() || map.getNumResults() != perm.size()) return map;
  llvm::SmallVector<mlir::AffineExpr> results;
  for (int64_t src : perm) results.push_back(map.getResult(src));
  return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                              map.getContext());
}

/// Mutable operand state of a data_transfer, used to build its replacement.
struct TransferOperands {
  mlir::AffineMap src_map, dst_map;
  llvm::SmallVector<mlir::Value> src_indices, dst_indices;
  llvm::SmallVector<mlir::OpFoldResult> src_sizes, dst_sizes;

  explicit TransferOperands(mlir::ktdf::DataTransferOp op)
      : src_map(op.isSourceMemRef() ? op.getSourceMapAttr().getValue()
                                    : mlir::AffineMap{}),
        dst_map(op.isDestMemRef() ? op.getDestMapAttr().getValue()
                                  : mlir::AffineMap{}),
        src_indices(op.getSourceIndices()),
        dst_indices(op.getDestIndices()),
        src_sizes(op.getMixedSourceSizes()),
        dst_sizes(op.getMixedDestSizes()) {}
};

/// Replaces `op` with a transfer built from `t`, keeping its discardable
/// attributes. Returns the new op.
static mlir::ktdf::DataTransferOp replaceTransfer(mlir::ktdf::DataTransferOp op,
                                                  const TransferOperands& t,
                                                  mlir::OpBuilder& builder) {
  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(), op.getSource(), t.src_map, t.src_indices,
      t.src_sizes, op.getDestination(), t.dst_map, t.dst_indices, t.dst_sizes);
  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());
  op.erase();
  return new_op;
}

/// Multiplies the static size at `dim` by `factor`; no-op if out of range.
static void scaleDim(llvm::SmallVectorImpl<mlir::OpFoldResult>& sizes,
                     int64_t dim, int64_t factor, mlir::MLIRContext* ctx) {
  if (dim < 0 || static_cast<size_t>(dim) >= sizes.size()) return;
  mlir::OpFoldResult& size = sizes[dim];
  int64_t existing =
      mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(size)).getInt();
  size = mlir::IntegerAttr::get(mlir::IndexType::get(ctx), existing * factor);
}

/// Walks back through cast-like ops to the defining construct_memory_view.
static mlir::ktdp::ConstructMemoryViewOp findConstructMemoryView(
    mlir::Value val) {
  while (auto* def = val.getDefiningOp()) {
    if (auto construct = mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(def))
      return construct;
    if (!mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp,
                   mlir::memref::ReinterpretCastOp>(def))
      return {};
    val = def->getOperand(0);
  }
  return {};
}

/// Physical shape of `construct`: its shape with all dims (inner ones too)
/// permuted into descending-stride order, as laid out in memory. Unchanged if
/// the strides are already descending.
static llvm::SmallVector<int64_t> viewShape(
    mlir::ktdp::ConstructMemoryViewOp construct) {
  auto shape =
      mlir::cast<mlir::MemRefType>(construct.getResult().getType()).getShape();
  auto strides = construct.getStaticStrides();
  if (strides.size() != shape.size() ||
      llvm::any_of(strides, mlir::ShapedType::isDynamic))
    return llvm::SmallVector<int64_t>(shape.begin(), shape.end());

  llvm::SmallVector<int64_t> perm(strides.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    return strides[a] > strides[b];
  });
  return applyPermutation<int64_t>(perm, shape);
}

static llvm::SmallVector<int64_t> rowMajorStrides(
    llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

/// Pushes `construct`'s result layout forward through downstream cast-like
/// ops so the old strided layout does not linger in their result types.
static void propagateLayout(mlir::ktdp::ConstructMemoryViewOp construct) {
  llvm::SmallVector<mlir::Value> worklist = {construct.getResult()};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    auto src_type = mlir::dyn_cast<mlir::MemRefType>(v.getType());
    if (!src_type) continue;
    for (mlir::Operation* user : v.getUsers()) {
      if (!mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp>(
              user))
        continue;
      mlir::Value result = user->getResult(0);
      auto res_type = mlir::dyn_cast<mlir::MemRefType>(result.getType());
      if (!res_type) continue;
      result.setType(mlir::MemRefType::get(
          src_type.getShape(), res_type.getElementType(), src_type.getLayout(),
          res_type.getMemorySpace()));
      worklist.push_back(result);
    }
  }
}

/// Rewrites the construct_memory_view behind a strided global memref `val` to
/// natural row-major strides in place. The strided layout was only used by the
/// analysis to detect illegality; it must not survive the rewrite.
static void rewriteConstructStrides(
    mlir::Value val, const scheduler::arch_view::MemoryTree& memory_tree) {
  auto mrt = mlir::dyn_cast<mlir::MemRefType>(val.getType());
  if (!mrt || isPerCoreScratchpad(val, memory_tree) ||
      !mlir::isa<mlir::StridedLayoutAttr>(mrt.getLayout()))
    return;
  auto construct = findConstructMemoryView(val);
  if (!construct) return;

  mlir::MLIRContext* ctx = construct->getContext();
  auto orig_type =
      mlir::cast<mlir::MemRefType>(construct.getResult().getType());
  auto new_shape = viewShape(construct);
  auto row_major = rowMajorStrides(new_shape);

  LDBG(1) << "  rewriteConstructStrides: rewriting strides on "
          << construct->getLoc() << " to row-major";
  construct.setStaticStridesAttr(mlir::DenseI64ArrayAttr::get(ctx, row_major));
  construct.setStaticSizesAttr(mlir::DenseI64ArrayAttr::get(ctx, new_shape));
  construct.getResult().setType(mlir::MemRefType::get(
      new_shape, orig_type.getElementType(),
      mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic, row_major),
      orig_type.getMemorySpace()));
  propagateLayout(construct);
}

/// Replaces `op` with a new DataTransferOp with the appropriate dim widened by
/// pa.alignment_factor, outer source indices permuted, and any strided global
/// memref rewritten to row-major. The widened dim is derived per operand from
/// the pipeline permutation: in-memory-order for global sources, logical order
/// for global destinations and ct_local allocs.
static void rewriteTransferShape(
    mlir::ktdf::DataTransferOp op,
    const DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  mlir::MLIRContext* ctx = op->getContext();

  TransferOperands t(op);

  // Scale the correct dim for each operand. A global source with a strided
  // layout is described in memory order; everything else uses logical order.
  auto src_rank = static_cast<int64_t>(t.src_sizes.size());
  auto dst_rank = static_cast<int64_t>(t.dst_sizes.size());
  bool src_is_global =
      op.isSourceMemRef() && !isPerCoreScratchpad(op.getSource(), memory_tree);
  bool src_strided =
      src_is_global &&
      mlir::isa<mlir::StridedLayoutAttr>(
          mlir::cast<mlir::MemRefType>(op.getSource().getType()).getLayout());
  int64_t src_wdim =
      widenedDim(pa.permutation, pa.inv_permutation, src_rank, src_strided);
  int64_t dst_wdim =
      widenedDim(pa.permutation, pa.inv_permutation, dst_rank, false);
  scaleDim(t.src_sizes, src_wdim, pa.alignment_factor, ctx);
  scaleDim(t.dst_sizes, dst_wdim, pa.alignment_factor, ctx);

  // A strided global source view is rewritten to memory order (see
  // rewriteConstructStrides), so its coordinates must be permuted the same way.
  if (src_strided)
    t.src_map = permuteMapResults(
        t.src_map, derivePermutation(
                       mlir::cast<mlir::MemRefType>(op.getSource().getType())));

  rewriteConstructStrides(op.getSource(), memory_tree);
  rewriteConstructStrides(op.getDestination(), memory_tree);

  auto new_op = replaceTransfer(op, t, builder);
  LDBG(1) << "  rewriteTransferShape: widened src dim " << src_wdim
          << " dst dim " << dst_wdim << " by factor " << pa.alignment_factor
          << ": " << new_op;
}

/// {row, col} induction variables of the two nearest enclosing scf.for loops
/// of `op` (inner loop = col). Both null if fewer than two loops enclose it.
static std::pair<mlir::Value, mlir::Value> findRowColIVs(mlir::Operation* op) {
  llvm::SmallVector<mlir::Value, 2> ivs;
  for (mlir::Operation* cursor = op->getParentOp(); cursor && ivs.size() < 2;
       cursor = cursor->getParentOp())
    if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(cursor))
      ivs.push_back(for_op.getInductionVar());
  if (ivs.size() < 2) return {};
  return {ivs[1], ivs[0]};
}

/// Replaces `op` with a new DataTransferOp sized [1,1,1,1] with
/// transfer_mode set to `mode` ("splat" / "extract") or empty for plain
/// FIFO→memref. The two innermost indices on any ct_local memref operand are
/// replaced with the row and column IVs from the nested scf.for loops inserted
/// by insertLoopAroundPipeline, giving direct [0, 0, %row, %col] addressing.
static void rewriteTransferShrink(
    mlir::ktdf::DataTransferOp op,
    const DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::MemoryTree& memory_tree, llvm::StringRef mode,
    mlir::OpBuilder& builder) {
  mlir::MLIRContext* ctx = op->getContext();
  TransferOperands t(op);

  // The memref side collapses to one scalar element; the FIFO side keeps its
  // natural slot size so the hardware broadcasts / extracts across it.
  mlir::OpFoldResult one = mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1);
  llvm::fill(mode == "extract" ? t.dst_sizes : t.src_sizes, one);

  auto ivs = findRowColIVs(op);
  mlir::Value row_iv = ivs.first, col_iv = ivs.second;
  if (!row_iv || !col_iv)
    LDBG(1) << "  rewriteTransferShrink: missing row/col IV — apply_row_col "
               "will no-op";

  // For ct_local memref operands, build a fresh affine map and index list.
  // The widened dim holds the row (or col when transposed); the last dim holds
  // the col (or row). The widened dim is derived from the pipeline permutation
  // in logical order, consistent with how the alloc was widened.
  auto apply_row_col = [&](mlir::Value memref_val, mlir::AffineMap& map,
                           llvm::SmallVector<mlir::Value>& indices,
                           bool transpose) {
    if (!row_iv || !col_iv) return;
    if (!isPerCoreScratchpad(memref_val, memory_tree)) return;
    int64_t rank = mlir::cast<mlir::MemRefType>(memref_val.getType()).getRank();
    if (rank < 2) return;

    // Pad indices with zeros if needed so it has length `rank`.
    if (indices.size() < static_cast<size_t>(rank)) {
      builder.setInsertionPoint(op);
      indices.resize(
          rank, mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0)
                    .getResult());
    }

    // Place row/col IVs at the widened dim and the last dim.
    int64_t wdim = widenedDim(pa.permutation, pa.inv_permutation, rank, false);
    indices[wdim] = transpose ? col_iv : row_iv;
    indices[rank - 1] = transpose ? row_iv : col_iv;
    map = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
  };

  apply_row_col(op.getSource(), t.src_map, t.src_indices, false);
  apply_row_col(op.getDestination(), t.dst_map, t.dst_indices, true);

  auto new_op = replaceTransfer(op, t, builder);
  if (!mode.empty())
    new_op->setDiscardableAttr(mlir::StringAttr::get(ctx, "transfer_mode"),
                               mlir::StringAttr::get(ctx, mode));

  LDBG(1) << "  rewriteTransferShrink(" << mode
          << "): alignment_factor=" << pa.alignment_factor << ": " << new_op;
}

/// Wraps the nested ktdf.pipeline in two nested scf.for loops over [0, bound)
/// so that each element of the widened alloc block can be addressed
/// directly via %row and %col IVs without any index arithmetic.
static void insertLoopAroundPipeline(
    DataTransferLegality::PipelineAnalysis* nested_pa, int64_t bound,
    mlir::OpBuilder& builder) {
  mlir::ktdf::PipelineOp nested_pipeline = nested_pa->pipeline;
  mlir::Location loc = nested_pipeline.getLoc();

  builder.setInsertionPoint(nested_pipeline);

  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value c_bound =
      mlir::arith::ConstantIndexOp::create(builder, loc, bound).getResult();
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

  // Outer loop: rows [0, bound)
  auto row_loop = mlir::scf::ForOp::create(builder, loc, c0, c_bound, c1);
  mlir::Block* row_body = row_loop.getBody();

  // Inner loop: cols [0, bound) — inserted inside the row loop body.
  builder.setInsertionPoint(row_body, row_body->getTerminator()->getIterator());
  auto col_loop = mlir::scf::ForOp::create(builder, loc, c0, c_bound, c1);

  // Move the nested pipeline into the col loop body.
  mlir::Block* col_body = col_loop.getBody();
  nested_pipeline->moveBefore(col_body,
                              col_body->getTerminator()->getIterator());

  LDBG(1) << "  insertLoopAroundPipeline: inserted row/col loops (bound="
          << bound << ") around nested pipeline at " << loc;
}

/// Applies corrective rewrites to the PipelineAnalysis tree: adjusts the
/// stage loop bounds once, inserts loops around nested pipelines, then for
/// each leaf stage rewrites every transfer (factor > 1 → widen; otherwise
/// FIFO transfers shrink to scalar splat / extract).
static mlir::LogicalResult fixPipeline(
    DataTransferLegality::PipelineAnalysis& pa, DataTransferLegality& legality,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " alignment_factor=" << pa.alignment_factor
          << " innermost_size=" << pa.innermost_size;

  if (mlir::failed(adjustLoopBounds(pa, builder))) return mlir::failure();

  for (DataTransferLegality::StageAnalysis& sa : pa.stages) {
    if (sa.nested_pipeline) {
      insertLoopAroundPipeline(sa.nested_pipeline.get(), pa.innermost_size,
                               builder);
      if (mlir::failed(fixPipeline(*sa.nested_pipeline, legality,
                                   resource_kinds, memory_tree, builder)))
        return mlir::failure();
      continue;
    }

    // The alignment factor is a pipeline invariant: hoist the branch out of
    // the per-transfer loop so it is evaluated once per stage.
    if (pa.alignment_factor > 1) {
      // Outer pipeline: widen the staging buffer and the transfer shape.
      for (mlir::ktdf::DataTransferOp dt : sa.transfers) {
        widenAlloc(dt, pa, memory_tree, builder);
        rewriteTransferShape(dt, pa, memory_tree, builder);
      }
    } else {
      // Nested / inner pipeline: shrink FIFO transfers to scalar access with
      // IV indexing.
      for (mlir::ktdf::DataTransferOp dt : sa.transfers) {
        if (dt.isSourceMemRef() && dt.isDestFifo()) {
          if (legality.canSplat(dt, resource_kinds))
            rewriteTransferShrink(dt, pa, memory_tree, "splat", builder);
        } else if (dt.isSourceFifo() && dt.isDestMemRef()) {
          llvm::StringRef mode =
              legality.canExtract(dt, resource_kinds) ? "extract" : "";
          rewriteTransferShrink(dt, pa, memory_tree, mode, builder);
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
    llvm::errs() << "[" PASS_NAME "] running on: " << getOperation()->getName()
                 << "\n";
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
        device_manager.getOrCreateView<scheduler::arch_view::MemoryTree>(
            *device);

    llvm::SmallVector<DataTransferLegality::PipelineAnalysis, 2> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(
              legality_.analyzePipeline(pipeline, resource_kinds));
          return mlir::WalkResult::skip();
        });

    mlir::OpBuilder builder(getOperation()->getContext());
    for (auto& pa : pipelines) {
      if (pa.alignment_factor == 0)
        continue;  // nothing to fix for this pipeline
      if (mlir::failed(fixPipeline(pa, legality_, resource_kinds, memory_tree,
                                   builder))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  DataTransferLegality legality_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
