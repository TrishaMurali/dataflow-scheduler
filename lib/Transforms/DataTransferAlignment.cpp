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
#include <optional>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Transfer path through a pipeline anchored at one global memref. An input
/// chain runs global source -> staging buffer(s) -> compute FIFO; an output
/// chain runs compute FIFO -> staging buffer(s) -> global destination. A
/// pipeline with several global inputs has one input chain per input.
struct TransferChain {
  mlir::Value global;
  bool is_input = true;
  llvm::SmallVector<int64_t> permutation;  // descending-stride order of
                                           // `global`; empty if row-major
  llvm::SmallVector<mlir::ktdf::DataTransferOp> hops;  // in discovery order

  /// True for an input whose global is not in memory order, i.e. whose
  /// elements are swapped on the way into the compute FIFO.
  bool needsStrideSwap() const { return is_input && !permutation.empty(); }
};

/// Describes the memory indexing and slice parameters for one endpoint (source
/// or destination) of a `ktdf.data_transfer`.
///
/// Bundles the coordinate transformation `map`, the base coordinate `indices`,
/// and the transfer slice `sizes`. Provides helpers to extract these
/// descriptors from an existing transfer op for transformation (e.g. widening
/// or scalarizing).
struct Access {
  mlir::AffineMap map;
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<mlir::OpFoldResult> sizes;

  static Access source(mlir::ktdf::DataTransferOp op) {
    return {op.isSourceMemRef() ? op.getSourceMapAttr().getValue()
                                : mlir::AffineMap{},
            llvm::to_vector(op.getSourceIndices()), op.getMixedSourceSizes()};
  }

  static Access dest(mlir::ktdf::DataTransferOp op) {
    return {
        op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{},
        llvm::to_vector(op.getDestIndices()), op.getMixedDestSizes()};
  }
};

/// The contiguous (unit-stride) dim of a memref.
struct ContiguousDim {
  int64_t dim;   // logical dim with stride 1
  int64_t size;  // its extent
};

/// The contiguous (unit-stride) dim of `memref`; failure if there is none or
/// its size is dynamic. Size-1 dims are skipped.
mlir::FailureOr<ContiguousDim> contiguousDim(mlir::MemRefType memref) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(memref.getStridesAndOffset(strides, offset))) {
    return mlir::failure();
  }
  for (auto [idx, p] : llvm::enumerate(llvm::zip(strides, memref.getShape()))) {
    auto [stride, size] = p;
    if (stride == 1 && size != 1 && !mlir::ShapedType::isDynamic(size)) {
      return ContiguousDim{static_cast<int64_t>(idx), size};
    }
  }
  return mlir::failure();
}

/// Logical dims sorted by descending stride; empty for identity, rank 0 or
/// dynamic strides.
llvm::SmallVector<int64_t> derivePermutation(llvm::ArrayRef<int64_t> strides) {
  if (strides.empty() || llvm::any_of(strides, mlir::ShapedType::isDynamic)) {
    return {};
  }

  llvm::SmallVector<int64_t> perm(strides.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    return strides[a] > strides[b];
  });
  if (llvm::all_of(llvm::enumerate(perm), [](auto p) {
        return p.value() == static_cast<int64_t>(p.index());
      })) {
    return {};
  }
  return perm;
}

/// Stride-derived permutation of a memref value; empty if none.
llvm::SmallVector<int64_t> memrefPermutation(mlir::Value memref) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  auto type = mlir::cast<mlir::MemRefType>(memref.getType());
  if (mlir::failed(type.getStridesAndOffset(strides, offset))) {
    return {};
  }
  return derivePermutation(strides);
}

/// result[i] = values[perm[i]]; an empty `perm` returns `values` unchanged.
template <typename T>
llvm::SmallVector<T> applyPermutation(llvm::ArrayRef<int64_t> perm,
                                      llvm::ArrayRef<T> values) {
  if (perm.empty()) {
    return llvm::SmallVector<T>(values.begin(), values.end());
  }
  llvm::SmallVector<T> result;
  result.reserve(perm.size());
  for (int64_t src : perm) {
    result.push_back(values[src]);
  }
  return result;
}

/// Dim of a rank-`rank` operand that holds the contiguous logical dim
/// (`perm.back()`) of the strided input driving the pipeline's alignment.
/// Staging buffers carry extra leading loop dims.
int64_t widenedDim(llvm::ArrayRef<int64_t> perm, int64_t rank) {
  return rank - static_cast<int64_t>(perm.size()) + perm.back();
}

/// Constant trip count of `loop`: the total_size of a ktdf.tiling.derive_size
/// upper bound, or a bare constant bound for untiled IR. nullopt otherwise.
std::optional<int64_t> constantTotalSize(mlir::scf::ForOp loop) {
  mlir::Value bound = loop.getUpperBound();
  auto derive = bound.getDefiningOp<mlir::ktdf::TilingDeriveSizeOp>();
  auto cst = (derive ? derive.getTotalSize() : bound)
                 .getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!cst) {
    return std::nullopt;
  }
  return cst.value();
}

/// Returns the single applicable unit kind for the stage enclosing `op`,
/// stopping at any PipelineOp boundary. Returns nullptr if the stage has
/// zero or more than one unit.
mlir::ktdf_arch::KindAttr getUnitKind(mlir::Operation* const op) {
  for (mlir::Operation* cursor = op->getParentOp(); cursor;
       cursor = cursor->getParentOp()) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(cursor)) {
      return nullptr;
    }
    if (auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(cursor)) {
      auto units = stage.getApplicableUnits();
      if (!units || units->size() != 1) {
        return nullptr;
      }
      return mlir::dyn_cast<mlir::ktdf_arch::KindAttr>((*units)[0]);
    }
  }
  return nullptr;
}

/// Returns true if `Feature` of `kind` supports element-level granularity.
template <typename Feature>
bool hasElementGranularity(
    mlir::ktdf_arch::KindAttr kind, mlir::Attribute space, uint64_t elem_bytes,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  auto feat = resource_kinds.getFeature<Feature>(kind);
  if (!feat) {
    return false;
  }
  auto granularities = feat.getAccessGranularity(space);
  if (!granularities) {
    return false;
  }
  uint64_t word_bytes = feat.getWordSize(space);
  return llvm::any_of(granularities, [&](auto entry) {
    return entry.getSizeInWords() * word_bytes == elem_bytes;
  });
}

/// True if the unit running `transfer` can address single elements of its
/// memref operand, i.e. it can swap strides itself. DMA-style block units
/// cannot.
bool isElementAddressable(
    mlir::ktdf::DataTransferOp transfer,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  auto kind = getUnitKind(transfer);
  if (!kind) {
    return false;
  }
  mlir::Value memref = transfer.isSourceMemRef() ? transfer.getSource()
                                                 : transfer.getDestination();
  auto type = mlir::cast<mlir::MemRefType>(memref.getType());
  auto elem_bytes = scheduler::tryGetSizeInBytes(type.getElementType());
  if (!elem_bytes) {
    return false;
  }
  auto space = type.getMemorySpace();
  return hasElementGranularity<mlir::ktdf_arch::feature::Load>(
             kind, space, *elem_bytes, resource_kinds) ||
         hasElementGranularity<mlir::ktdf_arch::feature::Store>(
             kind, space, *elem_bytes, resource_kinds);
}

/// Returns true if `val` is a memref located in per-core ct_local memory.
bool isPerCoreScratchpad(mlir::Value val,
                         const scheduler::arch_view::MemoryTree& memory_tree) {
  auto mrt = mlir::dyn_cast<mlir::MemRefType>(val.getType());
  if (!mrt) {
    return false;
  }
  auto kind = mlir::dyn_cast_if_present<mlir::ktdf_arch::KindAttr>(
      mrt.getMemorySpace());
  return kind && memory_tree.isPerCoreScratchPadMemory(kind);
}

/// Splits the transfers of `pipeline` (nested pipelines included) into one
/// chain per global memref. An input chain follows a global source forward
/// through per-core staging buffers; an output chain follows a global
/// destination backward. Each transfer belongs to at most one chain.
llvm::SmallVector<TransferChain> collectTransferChains(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
  pipeline->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::ktdf::DataTransferOp transfer) {
        transfers.push_back(transfer);
      });

  llvm::SmallVector<TransferChain> chains;
  llvm::DenseSet<mlir::Operation*> claimed;
  const auto new_chain = [&](mlir::Value global, bool is_input) {
    TransferChain chain;
    chain.global = global;
    chain.is_input = is_input;
    chain.permutation = memrefPermutation(global);
    chains.push_back(std::move(chain));
    return chains.size() - 1;
  };

  // Input chains: global source -> staging buffer(s) -> compute FIFO.
  llvm::DenseMap<mlir::Value, size_t> input_of;
  for (mlir::ktdf::DataTransferOp transfer : transfers) {
    if (!transfer.isSourceMemRef()) {
      continue;
    }
    mlir::Value src = transfer.getSource();
    if (!isPerCoreScratchpad(src, memory_tree) && !input_of.count(src)) {
      input_of[src] = new_chain(src, true);
    }
    auto it = input_of.find(src);
    if (it == input_of.end()) {
      continue;
    }
    size_t idx = it->second;
    chains[idx].hops.push_back(transfer);
    claimed.insert(transfer.getOperation());
    if (transfer.isDestMemRef() &&
        isPerCoreScratchpad(transfer.getDestination(), memory_tree)) {
      input_of.try_emplace(transfer.getDestination(), idx);
    }
  }

  // Output chains: compute FIFO -> staging buffer(s) -> global destination.
  llvm::DenseMap<mlir::Value, size_t> output_of;
  for (mlir::ktdf::DataTransferOp transfer : llvm::reverse(transfers)) {
    if (!transfer.isDestMemRef()) {
      continue;
    }
    mlir::Value dst = transfer.getDestination();
    if (!isPerCoreScratchpad(dst, memory_tree) && !output_of.count(dst)) {
      output_of[dst] = new_chain(dst, false);
    }
    auto it = output_of.find(dst);
    if (it == output_of.end()) {
      continue;
    }
    size_t idx = it->second;
    if (claimed.insert(transfer.getOperation()).second) {
      chains[idx].hops.push_back(transfer);
    }
    if (transfer.isSourceMemRef() &&
        isPerCoreScratchpad(transfer.getSource(), memory_tree)) {
      output_of.try_emplace(transfer.getSource(), idx);
    }
  }
  return chains;
}

/// Tile size of the tiled loop that indexes source dim `dim` of `transfer`,
/// read from the linearize_index feeding that dim. Null if the dim is not
/// tiled.
mlir::Value tileSizeOfDim(mlir::ktdf::DataTransferOp transfer, int64_t dim) {
  mlir::AffineMap map = transfer.getSourceMapAttr().getValue();
  auto indices = transfer.getSourceIndices();
  if (dim < 0 || static_cast<unsigned>(dim) >= map.getNumResults()) {
    return {};
  }
  mlir::AffineExpr expr = map.getResult(dim);
  for (unsigned idx = 0; idx < map.getNumDims() && idx < indices.size();
       ++idx) {
    if (!expr.isFunctionOfDim(idx)) {
      continue;
    }
    auto lin = indices[idx].getDefiningOp<mlir::ktdf::TilingLinearizeIndexOp>();
    if (!lin || lin.getStrides().size() < 2) {
      continue;
    }
    // Strides are outermost-first and end with the point loop's 1.
    return lin.getStrides()[lin.getStrides().size() - 2];
  }
  return {};
}

/// True if `loop` walks the contiguous dim: its derive_size uses `tile` as the
/// innermost tile size when known, else its total_size equals `size`.
bool isTargetLoop(mlir::scf::ForOp loop, int64_t size, mlir::Value tile) {
  if (!tile) {
    return constantTotalSize(loop).value_or(0) == size;
  }
  auto derive =
      loop.getUpperBound().getDefiningOp<mlir::ktdf::TilingDeriveSizeOp>();
  return derive && !derive.getTileSizes().empty() &&
         derive.getTileSizes().back() == tile;
}

/// Outermost target loops inside `pipeline` (not descending into nested
/// pipelines), or the nearest enclosing target loop if there are none.
llvm::SmallVector<mlir::scf::ForOp> collectTargetLoops(
    mlir::ktdf::PipelineOp pipeline, int64_t size, mlir::Value tile) {
  llvm::SmallVector<mlir::scf::ForOp> loops;
  pipeline->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* inner_op) {
    if (inner_op != pipeline.getOperation() &&
        mlir::isa<mlir::ktdf::PipelineOp>(inner_op)) {
      return mlir::WalkResult::skip();
    }
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(inner_op);
    if (!loop || !isTargetLoop(loop, size, tile)) {
      return mlir::WalkResult::advance();
    }
    loops.push_back(loop);
    return mlir::WalkResult::skip();
  });
  if (!loops.empty()) {
    return loops;
  }

  for (mlir::Operation* cursor = pipeline->getParentOp(); cursor;
       cursor = cursor->getParentOp()) {
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(cursor);
    if (loop && isTargetLoop(loop, size, tile)) {
      return {loop};
    }
  }
  return loops;
}

/// Replaces the symbolic reserve_size tile placeholder of a derive_size bound
/// with the constant `factor`.
void fixReserveSize(mlir::scf::ForOp loop, int64_t factor,
                    mlir::OpBuilder& builder) {
  auto derive =
      loop.getUpperBound().getDefiningOp<mlir::ktdf::TilingDeriveSizeOp>();
  if (!derive || derive.getTileSizes().empty()) {
    return;
  }
  auto reserve_op = derive.getTileSizes()
                        .back()
                        .getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
  if (!reserve_op) {
    return;
  }
  builder.setInsertionPoint(reserve_op);
  auto cst_tile = mlir::arith::ConstantIndexOp::create(
      builder, reserve_op.getLoc(), factor);
  reserve_op.replaceAllUsesWith(cst_tile.getResult());
  reserve_op.erase();
}

/// Divides the trip count of `loop` by `factor`, pinning its tile to `factor`.
mlir::LogicalResult shrinkLoop(mlir::scf::ForOp loop, int64_t factor,
                               mlir::OpBuilder& builder) {
  // Ensure constant trip count is divisible by the alignment factor.
  std::optional<int64_t> total = constantTotalSize(loop);
  if (!total || *total % factor != 0) {
    return loop->emitError(PASS_NAME
                           ": loop trip count is not a constant "
                           "divisible by alignment factor ")
           << factor;
  }

  // Pin any symbolic reserve tile size to factor, then divide upper bound.
  fixReserveSize(loop, factor, builder);
  builder.setInsertionPoint(loop);
  loop.setUpperBound(mlir::arith::ConstantIndexOp::create(
      builder, loop.getLoc(), *total / factor));
  return mlir::success();
}

/// Widens the alloc behind `val` by `factor` on the contiguous dim, trading
/// the outermost (loop) dim for it. Allocs outside `pipeline` are left alone.
mlir::LogicalResult widenStagingAlloc(mlir::Value val,
                                      mlir::ktdf::PipelineOp pipeline,
                                      llvm::ArrayRef<int64_t> perm,
                                      int64_t factor,
                                      mlir::OpBuilder& builder) {
  // Resolve backing alloc and ensure it belongs to this pipeline.
  auto pop = mlir::dyn_cast_or_null<mlir::ktdf::PrivateOp>(val.getDefiningOp());
  mlir::Value alloc_val =
      pop ? pop.getYieldOp().getOperand(
                mlir::cast<mlir::OpResult>(val).getResultNumber())
          : val;
  auto alloc =
      mlir::dyn_cast_or_null<mlir::memref::AllocOp>(alloc_val.getDefiningOp());
  if (!alloc) {
    return mlir::emitError(val.getLoc())
           << PASS_NAME ": ct_local operand has no backing memref.alloc";
  }
  if (alloc->getParentOp() != pipeline.getPrivateOp().getOperation()) {
    return mlir::success();
  }

  // Find the contiguous dimension on the staging buffer.
  mlir::MemRefType type = alloc.getType();
  llvm::SmallVector<int64_t> shape(type.getShape());
  llvm::SmallVector<mlir::Value> dynamic_sizes(alloc.getDynamicSizes());
  int64_t dim = widenedDim(perm, type.getRank());
  if (type.getRank() < 2 || mlir::ShapedType::isDynamic(shape[dim])) {
    return alloc.emitError(PASS_NAME ": cannot widen staging buffer ") << type;
  }

  // Trade the outermost loop dimension for the contiguous dimension.
  builder.setInsertionPoint(alloc);
  if (dynamic_sizes.empty()) {
    if (shape[0] % factor != 0) {
      return alloc.emitError(PASS_NAME ": outermost dimension (")
             << shape[0] << ") is not divisible by alignment factor " << factor;
    }
    shape[0] /= factor;
  } else {
    // The loop tile was pinned to `factor`; that dynamic size collapses to 1.
    for (mlir::Value& size : llvm::reverse(dynamic_sizes)) {
      auto cst = size.getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!cst || cst.value() != factor) {
        continue;
      }
      size = mlir::arith::ConstantIndexOp::create(builder, alloc.getLoc(), 1);
      break;
    }
  }
  shape[dim] *= factor;

  // Reallocate the staging buffer with the updated shape and update the
  // private result type.
  auto new_type = mlir::MemRefType::get(shape, type.getElementType(),
                                        mlir::MemRefLayoutAttrInterface{},
                                        type.getMemorySpace());
  auto new_alloc = mlir::memref::AllocOp::create(builder, alloc.getLoc(),
                                                 new_type, dynamic_sizes);
  // The ktdf.private verifier requires its result type to match the alloc.
  if (pop) {
    val.setType(new_type);
  }
  alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
  alloc.erase();
  return mlir::success();
}

/// Walks back through cast-like ops to the defining construct_memory_view.
mlir::ktdp::ConstructMemoryViewOp findConstructMemoryView(mlir::Value val) {
  while (auto* def = val.getDefiningOp()) {
    if (auto construct =
            mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(def)) {
      return construct;
    }
    if (!mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp,
                   mlir::memref::ReinterpretCastOp>(def)) {
      return {};
    }
    val = def->getOperand(0);
  }
  return {};
}

/// Pushes `construct`'s result layout forward through downstream cast-like
/// ops so the old strided layout does not linger in their result types.
void propagateLayout(mlir::ktdp::ConstructMemoryViewOp construct) {
  llvm::SmallVector<mlir::Value> worklist = {construct.getResult()};
  while (!worklist.empty()) {
    mlir::Value val = worklist.pop_back_val();
    auto src_type = mlir::cast<mlir::MemRefType>(val.getType());
    for (mlir::Operation* user : val.getUsers()) {
      if (!mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp>(
              user)) {
        continue;
      }
      mlir::Value result = user->getResult(0);
      auto res_type = mlir::cast<mlir::MemRefType>(result.getType());
      result.setType(mlir::MemRefType::get(
          src_type.getShape(), res_type.getElementType(), src_type.getLayout(),
          res_type.getMemorySpace()));
      worklist.push_back(result);
    }
  }
}

/// Rewrites the construct_memory_view behind the strided global `val` so its
/// dims are laid out in memory order with row-major strides.
void rewriteToMemoryOrder(mlir::Value val) {
  auto construct = findConstructMemoryView(val);
  if (!construct) {
    return;
  }

  // Permute shape to memory order and compute contiguous row-major strides.
  mlir::MLIRContext* ctx = construct->getContext();
  auto type = mlir::cast<mlir::MemRefType>(construct.getResult().getType());
  llvm::SmallVector<int64_t> shape = applyPermutation<int64_t>(
      derivePermutation(construct.getStaticStrides()), type.getShape());
  llvm::SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // Update construct_memory_view attributes and propagate the new layout
  // downstream.
  construct.setStaticStridesAttr(mlir::DenseI64ArrayAttr::get(ctx, strides));
  construct.setStaticSizesAttr(mlir::DenseI64ArrayAttr::get(ctx, shape));
  construct.getResult().setType(mlir::MemRefType::get(
      shape, type.getElementType(),
      mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic, strides),
      type.getMemorySpace()));
  propagateLayout(construct);
}

/// Replaces `op` with a new data_transfer having the updated `src`/`dst` access
/// parameters and optional `mode`, preserving any discardable attributes.
void rewriteTransfer(mlir::ktdf::DataTransferOp op, const Access& src,
                     const Access& dst,
                     std::optional<mlir::ktdf::TransferMode> mode,
                     mlir::OpBuilder& builder) {
  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(), op.getSource(), src.map, src.indices, src.sizes,
      op.getDestination(), dst.map, dst.indices, dst.sizes,
      mode.value_or(op.getTransferMode()));
  for (mlir::NamedAttribute attr : op->getDiscardableAttrs()) {
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());
  }
  LDBG(1) << "  rewriteTransfer: " << new_op;
  op.erase();
}

/// Multiplies the static size at `dim` by `factor`.
void scaleDim(llvm::SmallVectorImpl<mlir::OpFoldResult>& sizes, int64_t dim,
              int64_t factor, mlir::Builder& builder) {
  if (dim < 0 || static_cast<size_t>(dim) >= sizes.size()) {
    return;
  }
  sizes[dim] =
      builder.getIndexAttr(*mlir::getConstantIntValue(sizes[dim]) * factor);
}

/// Widens one memref side of a block transfer by `factor` on the contiguous
/// dim. A strided global is described in memory order, where that dim is last.
void widenAccess(mlir::Value memref, Access& access,
                 llvm::ArrayRef<int64_t> perm, int64_t factor,
                 mlir::OpBuilder& builder) {
  int64_t rank = access.sizes.size();
  llvm::SmallVector<int64_t> own = memrefPermutation(memref);
  // Unstrided / scratchpad operand: scale contiguous dimension directly.
  if (own.empty()) {
    scaleDim(access.sizes, widenedDim(perm, rank), factor, builder);
    return;
  }

  // Strided global operand: reorder access map and sizes into memory order
  // before scaling.
  llvm::SmallVector<unsigned> order(own.begin(), own.end());
  access.map = mlir::AffineMap::getPermutationMap(order, builder.getContext())
                   .compose(access.map);
  access.sizes = applyPermutation<mlir::OpFoldResult>(own, access.sizes);
  scaleDim(access.sizes, rank - 1, factor, builder);
}

/// Widens a block transfer (e.g. MNILU / MNISU) to cover the contiguous block.
void widenTransfer(mlir::ktdf::DataTransferOp op, llvm::ArrayRef<int64_t> perm,
                   int64_t factor, mlir::OpBuilder& builder) {
  Access src = Access::source(op), dst = Access::dest(op);
  if (op.isSourceMemRef()) {
    widenAccess(op.getSource(), src, perm, factor, builder);
  }
  if (op.isDestMemRef()) {
    widenAccess(op.getDestination(), dst, perm, factor, builder);
  }
  rewriteTransfer(op, src, dst, std::nullopt, builder);
}

/// Turns an element-addressable FIFO transfer into a scalar splat / extract of
/// the widened staging buffer. A splat load of a strided input (`swap_strides`)
/// applies the stride swap, reading at [.., col, .., row]; every other
/// transfer keeps natural order [.., row, .., col].
/// Whether the unit supports splat is left to the SplatLegalization pass.
void scalarizeTransfer(mlir::ktdf::DataTransferOp op,
                       llvm::ArrayRef<int64_t> perm, bool swap_strides,
                       std::pair<mlir::Value, mlir::Value> row_col,
                       const scheduler::arch_view::MemoryTree& memory_tree,
                       mlir::OpBuilder& builder) {
  // Only process transfers between local scratchpad memory and FIFOs.
  bool to_fifo = op.isSourceMemRef() && op.isDestFifo();
  bool from_fifo = op.isSourceFifo() && op.isDestMemRef();
  if (!to_fifo && !from_fifo) {
    return;
  }
  mlir::Value memref = to_fifo ? op.getSource() : op.getDestination();
  if (!isPerCoreScratchpad(memref, memory_tree)) {
    return;
  }

  // Splat loads into FIFO, extract stores from FIFO.
  mlir::ktdf::TransferMode mode = to_fifo ? mlir::ktdf::TransferMode::Splat
                                          : mlir::ktdf::TransferMode::Extract;

  // Shrink transfer sizes to 1 element.
  Access src = Access::source(op), dst = Access::dest(op);
  Access& mem = to_fifo ? src : dst;
  mlir::OpFoldResult one = builder.getIndexAttr(1);
  llvm::fill(mem.sizes, one);

  // Swap strides on strided-input splat loads [.., col, .., row]; keep natural
  // order otherwise [.., row, .., col].
  size_t rank = mlir::cast<mlir::MemRefType>(memref.getType()).getRank();
  builder.setInsertionPoint(op);
  if (mem.indices.size() < rank) {
    mem.indices.resize(
        rank, mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0)
                  .getResult());
  }
  auto [row, col] = row_col;
  mem.indices[widenedDim(perm, rank)] = swap_strides ? col : row;
  mem.indices[rank - 1] = swap_strides ? row : col;
  mem.map = mlir::AffineMap::getMultiDimIdentityMap(rank, op->getContext());

  // Replace original transfer op.
  rewriteTransfer(op, src, dst, mode, builder);
}

/// Wraps `pipeline` in two nested scf.for loops over [0, bound) so each
/// element of the widened block is addressed by the returned {row, col} IVs.
std::pair<mlir::Value, mlir::Value> wrapInElementLoops(
    mlir::ktdf::PipelineOp pipeline, int64_t bound, mlir::OpBuilder& builder) {
  // Create nested row and column loops over [0, bound) around the pipeline.
  mlir::Location loc = pipeline.getLoc();
  builder.setInsertionPoint(pipeline);
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value c_bound =
      mlir::arith::ConstantIndexOp::create(builder, loc, bound);

  auto row_loop = mlir::scf::ForOp::create(builder, loc, zero, c_bound, one);
  builder.setInsertionPoint(row_loop.getBody()->getTerminator());
  auto col_loop = mlir::scf::ForOp::create(builder, loc, zero, c_bound, one);
  pipeline->moveBefore(col_loop.getBody()->getTerminator());
  return {row_loop.getInductionVar(), col_loop.getInductionVar()};
}

/// {row, col} induction variables of the element loops already created,
/// keyed by the pipeline they wrap.
using ElementLoopMap =
    llvm::DenseMap<mlir::Operation*, std::pair<mlir::Value, mlir::Value>>;

/// Per-core staging buffers touched by the hops of `chain`, in discovery order.
llvm::SetVector<mlir::Value> collectStagingBuffers(
    const TransferChain& chain,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  llvm::SetVector<mlir::Value> staging_buffers;
  for (mlir::ktdf::DataTransferOp transfer : chain.hops) {
    for (mlir::Value operand :
         {transfer.getSource(), transfer.getDestination()}) {
      if (isPerCoreScratchpad(operand, memory_tree)) {
        staging_buffers.insert(operand);
      }
    }
  }
  return staging_buffers;
}

/// Aligns and legalizes a single transfer chain.
mlir::LogicalResult alignTransferChain(
    mlir::ktdf::PipelineOp pipeline, const TransferChain& chain,
    llvm::ArrayRef<int64_t> perm, int64_t factor, int64_t dim,
    int64_t innermost_size,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    const scheduler::arch_view::MemoryTree& memory_tree,
    ElementLoopMap& element_loops, mlir::OpBuilder& builder) {
  // Shrink the loops walking `dim` (for input chains).
  if (chain.is_input) {
    mlir::Value tile = tileSizeOfDim(chain.hops.front(), dim);
    for (mlir::scf::ForOp loop :
         collectTargetLoops(pipeline, innermost_size, tile)) {
      if (mlir::failed(shrinkLoop(loop, factor, builder))) {
        return mlir::failure();
      }
    }
  }

  // Widen the staging buffers of this chain.
  for (mlir::Value buffer : collectStagingBuffers(chain, memory_tree)) {
    if (mlir::failed(
            widenStagingAlloc(buffer, pipeline, perm, factor, builder))) {
      return mlir::failure();
    }
  }

  // Rewrite each transfer. Block units just move wider blocks. Element
  // units become scalar splat / extract inside element loops.
  for (mlir::ktdf::DataTransferOp transfer : chain.hops) {
    if (!isElementAddressable(transfer, resource_kinds)) {
      widenTransfer(transfer, perm, factor, builder);
      continue;
    }
    // Transfers of the outer pipeline do not touch a compute FIFO.
    auto nested = transfer->getParentOfType<mlir::ktdf::PipelineOp>();
    if (nested == pipeline) {
      continue;
    }
    // Wrap nested pipeline once in element loops.
    auto [it, first_use] = element_loops.try_emplace(nested.getOperation());
    if (first_use) {
      it->second = wrapInElementLoops(nested, innermost_size, builder);
    }
    scalarizeTransfer(transfer, perm, chain.needsStrideSwap(), it->second,
                      memory_tree, builder);
  }

  // Rewrite the strided global view to row-major order.
  if (!chain.permutation.empty()) {
    rewriteToMemoryOrder(chain.global);
  }
  return mlir::success();
}

/// Legalizes all global transfer chains of `pipeline` by aligning each chain
/// one at a time.
mlir::LogicalResult alignTransferChains(
    mlir::ktdf::PipelineOp pipeline, llvm::ArrayRef<TransferChain> chains,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  // The first strided input defines the alignment. If every global is
  // already in memory order there is nothing to do.
  const TransferChain* strided = llvm::find_if(
      chains,
      [](const TransferChain& chain) { return chain.needsStrideSwap(); });
  if (strided == chains.end()) {
    return mlir::success();
  }
  llvm::ArrayRef<int64_t> perm = strided->permutation;

  auto cdim =
      contiguousDim(mlir::cast<mlir::MemRefType>(strided->global.getType()));
  if (mlir::failed(cdim)) {
    return mlir::success();
  }
  const int64_t dim = cdim->dim;
  const int64_t innermost_size = cdim->size;

  // A single widening must serve every strided global, so they must all
  // have their unit stride on the same dim.
  for (const TransferChain& chain : chains) {
    if (!chain.permutation.empty() && chain.permutation.back() != perm.back()) {
      return chain.hops.front()->emitError() << PASS_NAME
             ": strided globals of one pipeline must share the "
             "same contiguous dimension";
    }
  }

  // Alignment factor: how many original transfers along `dim` fit in one
  // contiguous block of the global.
  mlir::ktdf::DataTransferOp first = strided->hops.front();
  auto sizes = first.getStaticSourceSizesArray();
  if (!sizes || static_cast<size_t>(dim) >= sizes->size()) {
    return first->emitError() << PASS_NAME ": static sizes are required";
  }
  const int64_t factor = innermost_size / std::max<int64_t>(1, (*sizes)[dim]);
  if (factor <= 1) {
    return mlir::success();
  }

  // Align each transfer chain one at a time.
  ElementLoopMap element_loops;
  for (const TransferChain& chain : chains) {
    if (mlir::failed(alignTransferChain(pipeline, chain, perm, factor, dim,
                                        innermost_size, resource_kinds,
                                        memory_tree, element_loops, builder))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";

    // Retrieve device architecture analysis views.
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

    // Collect all top-level pipelines.
    llvm::SmallVector<mlir::ktdf::PipelineOp> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(pipeline);
          return mlir::WalkResult::skip();
        });

    // Align and legalize transfers for each pipeline.
    mlir::OpBuilder builder(getOperation()->getContext());
    for (mlir::ktdf::PipelineOp pipeline : pipelines) {
      auto chains = collectTransferChains(pipeline, memory_tree);
      if (mlir::failed(alignTransferChains(pipeline, chains, resource_kinds,
                                           memory_tree, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
