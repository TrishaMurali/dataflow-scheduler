//===-- DataTransfers.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/ViewLikeInterfaceUtils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/ReshapeOpsUtils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Transforms/RegionUtils.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"

using namespace scheduler;

namespace {

/// Determines whether @p op is a `tensor.(collapse|expand)_shape`.
[[nodiscard]] auto isReshape(mlir::Operation* op) -> bool {
  return llvm::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
      op);
}

/// Obtains the slice taken by @p op .
[[nodiscard]] auto getSlice(mlir::OffsetSizeAndStrideOpInterface op)
    -> llvm::SmallVector<mlir::Range> {
  return llvm::map_to_vector(
      llvm::zip_equal(op.getMixedOffsets(), op.getMixedSizes(),
                      op.getMixedStrides()),
      [](auto range) -> mlir::Range {
        auto [offset, size, stride] = range;
        return {offset, size, stride};
      });
}

/// Obtains the static shape of a tensor of the sizes of @p slice .
[[nodiscard]] auto getSliceShape(llvm::ArrayRef<mlir::Range> slice)
    -> llvm::SmallVector<int64_t> {
  return llvm::map_to_vector(slice, [](const mlir::Range& range) {
    return mlir::getConstantIntValue(range.size)
        .value_or(mlir::ShapedType::kDynamic);
  });
}

/// Maps @p slice of a tensor of @p source_shape onto the same elements of a
/// reshaped tensor of @p target_shape .
///
/// In-bounds slices of unit dimensions always start at zero.
///
/// @retval failure   Shapes differ in more than static unit dimensions, or
///                   @p slice is not of size one in a dropped unit dimension.
/// @retval Slice     Mixed slice parameters.
[[nodiscard]] auto mapSlice(mlir::Builder& builder,
                            llvm::ArrayRef<mlir::Range> slice,
                            llvm::ArrayRef<int64_t> source_shape,
                            llvm::ArrayRef<int64_t> target_shape)
    -> llvm::FailureOr<llvm::SmallVector<mlir::Range>> {
  llvm::SmallVector<std::pair<int64_t, mlir::Range>> carried;
  for (auto [extent, range] : llvm::zip_equal(source_shape, slice)) {
    if (extent != 1) {
      carried.emplace_back(extent, range);
    } else if (!mlir::isConstantIntValue(range.size, 1)) {
      return llvm::failure();
    }
  }

  const mlir::Range unit{builder.getIndexAttr(0), builder.getIndexAttr(1),
                         builder.getIndexAttr(1)};
  llvm::SmallVector<mlir::Range> mapped;
  auto* next = carried.begin();
  for (const auto extent : target_shape) {
    if (extent == 1) {
      mapped.push_back(unit);
      continue;
    }
    if (next == carried.end() || mlir::ShapedType::isDynamic(extent) ||
        next->first != extent) {
      return llvm::failure();
    }
    mapped.push_back((next++)->second);
  }
  if (next != carried.end()) {
    return llvm::failure();
  }
  return mapped;
}

/// Reshapes @p value to @p shape , which may only differ in unit dimensions.
[[nodiscard]] auto reshapeTo(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value value, llvm::ArrayRef<int64_t> shape)
    -> mlir::Value {
  const auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
  if (type.getShape() == shape) {
    return value;
  }

  const auto reshape = [&](mlir::Value source,
                           mlir::RankedTensorType target) -> mlir::Value {
    const auto source_type =
        llvm::cast<mlir::RankedTensorType>(source.getType());
    const auto reassociation =
        mlir::getReassociationIndicesForReshape(source_type, target);
    assert(reassociation && "shapes must only differ in unit dimensions");
    if (source_type.getRank() > target.getRank()) {
      return mlir::tensor::CollapseShapeOp::create(builder, loc, source,
                                                   *reassociation);
    }
    return mlir::tensor::ExpandShapeOp::create(builder, loc, target, source,
                                               *reassociation);
  };

  // A single reshape suffices unless unit dimensions are dropped and added.
  const auto target = type.clone(shape);
  if (mlir::getReassociationIndicesForReshape(type, target)) {
    return reshape(value, target);
  }
  const auto squeezed = type.clone(llvm::to_vector(llvm::make_filter_range(
      shape, [](int64_t extent) { return extent != 1; })));
  return reshape(reshape(value, squeezed), target);
}

/// Memory region accessed by one side of a `ktdf.data_transfer`, given as the
/// affine map to its start, the indices of that map, and its sizes.
using TransferAccess =
    std::tuple<mlir::AffineMap, llvm::SmallVector<mlir::Value>,
               llvm::SmallVector<mlir::OpFoldResult>>;

/// Creates the access that a `ktdf.data_transfer` uses for @p slice .
///
/// Since `ktdf.data_transfer` always performs a contiguous access, non unit
/// strides are not allowed in dimensions of size not equal to one.
///
/// @retval failure         Invalid strided access.
/// @retval TransferAccess  Memory side of a `ktdf.data_transfer`.
[[nodiscard]] auto createTransferAccess(mlir::OpBuilder& builder,
                                        llvm::ArrayRef<mlir::Range> slice)
    -> llvm::FailureOr<TransferAccess> {
  llvm::SmallVector<mlir::AffineExpr> results;
  llvm::SmallVector<mlir::Value> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  for (const auto& range : slice) {
    if (!mlir::isConstantIntValue(range.stride, 1) &&
        !mlir::isConstantIntValue(range.size, 1)) {
      return llvm::failure();
    }

    if (const auto value = mlir::getConstantIntValue(range.offset); value) {
      results.push_back(builder.getAffineConstantExpr(*value));
    } else {
      results.push_back(builder.getAffineDimExpr(offsets.size()));
      offsets.push_back(llvm::cast<mlir::Value>(range.offset));
    }
    sizes.push_back(range.size);
  }

  return TransferAccess{
      mlir::AffineMap::get(offsets.size(), 0, results, builder.getContext()),
      offsets, sizes};
}

}  // namespace

//===----------------------------------------------------------------------===//
// scheduler::getMemorySpace
//===----------------------------------------------------------------------===//

auto scheduler::getMemorySpace(MemRef memref) -> mlir::Attribute {
  if (const auto space = memref.getType().getMemorySpace(); space) {
    return space;
  }

  if (auto source = memref.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>()) {
    return source.getMemorySpace();
  }

  return nullptr;
}

auto scheduler::getMemorySpace(AccessTile access_tile) -> mlir::Attribute {
  auto source = access_tile.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  if (!source) {
    return nullptr;
  }

  auto base = llvm::dyn_cast<MemRef>(source.getBase());
  if (!base) {
    return nullptr;
  }

  return getMemorySpace(base);
}

auto scheduler::getMemorySpace(mlir::Value value) -> mlir::Attribute {
  if (auto access_tile = llvm::dyn_cast<AccessTile>(value); access_tile) {
    return getMemorySpace(access_tile);
  }
  if (auto memref = llvm::dyn_cast<MemRef>(value); memref) {
    return getMemorySpace(memref);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// scheduler::(get|set)Throttle
//===----------------------------------------------------------------------===//

auto scheduler::getThrottle(mlir::Operation* op) -> int64_t {
  const auto attr =
      op->getAttrOfType<mlir::ktdf_arch::I64Attr>(kThrottleAttrName);
  return attr ? attr.getValue() : std::numeric_limits<int64_t>::max();
};

/// Whether @p op only moves data and passes on the throttle around it.
[[nodiscard]] static auto isThrottleTransparent(mlir::Operation* op) -> bool {
  return !op->hasAttr(kThrottleAttrName) &&
         (isReshape(op) ||
          llvm::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::InsertSliceOp>(
              op));
}

auto scheduler::getThrottle(mlir::OpResult result) -> int64_t {
  auto min = std::numeric_limits<int64_t>::max();
  for (auto* user : result.getUsers()) {
    min = std::min(min, isThrottleTransparent(user)
                            ? getThrottle(user->getOpResult(0))
                            : getThrottle(user));
  }
  return min;
};

auto scheduler::getThrottle(mlir::OpOperand& operand) -> int64_t {
  auto result = llvm::dyn_cast<mlir::OpResult>(operand.get());
  if (!result) {
    return std::numeric_limits<int64_t>::max();
  }

  auto* const producer = result.getOwner();
  if (!isThrottleTransparent(producer)) {
    return getThrottle(producer);
  }

  // An `insert_slice` combines its source and destination.
  if (auto insert = llvm::dyn_cast<mlir::tensor::InsertSliceOp>(producer)) {
    return std::min(getThrottle(insert.getSourceMutable()),
                    getThrottle(insert.getDestMutable()));
  }

  return getThrottle(producer->getOpOperand(0));
}

void scheduler::setThrottle(mlir::Operation* op, int64_t value) {
  op->setAttr(kThrottleAttrName,
              mlir::ktdf_arch::I64Attr::get(op->getContext(), value));
}

//===----------------------------------------------------------------------===//
// scheduler::populateDataMovementCoalescingPatterns
//===----------------------------------------------------------------------===//

namespace {

/// Folds a reshape of the data read from a FIFO into the read.
struct FoldReshapeIntoFifoRead
    : mlir::OpRewritePattern<mlir::ktdf::ReadFromFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::ReadFromFifoOp read,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    if (!read->hasOneUse() || !isReshape(*read->user_begin())) {
      return rewriter.notifyMatchFailure(read, "not reshaped");
    }

    auto* const reshape = *read->user_begin();
    rewriter.modifyOpInPlace(read, [&]() {
      read.getResult().setType(
          llvm::cast<mlir::ShapedType>(reshape->getResult(0).getType()));
    });
    rewriter.replaceOp(reshape, read.getResult());
    return llvm::success();
  }
};

/// Folds a reshape of the data written to a FIFO into the write.
struct FoldReshapeIntoFifoWrite
    : mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::WriteToFifoOp write,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto* const reshape = write.getData().getDefiningOp();
    if (!reshape || !isReshape(reshape)) {
      return rewriter.notifyMatchFailure(write, "not reshaped");
    }

    rewriter.modifyOpInPlace(write, [&]() {
      write.getDataMutable().assign(reshape->getOperand(0));
    });
    return llvm::success();
  }
};

auto createLoad(mlir::OpBuilder& builder, mlir::ktdp_lowering::LoadOp load,
                llvm::ArrayRef<mlir::Range> slice,
                std::optional<mlir::Location> slice_loc = {})
    -> llvm::FailureOr<mlir::ktdp_lowering::LoadOp> {
  const auto loc = slice_loc ? builder.getFusedLoc({load.getLoc(), *slice_loc})
                             : load->getLoc();

  const auto [offsets, sizes, strides] = mlir::getOffsetsSizesAndStrides(slice);
  llvm::SmallVector<mlir::OpFoldResult> merged_offsets;
  llvm::SmallVector<mlir::OpFoldResult> merged_sizes;
  llvm::SmallVector<mlir::OpFoldResult> merged_strides;
  if (mlir::failed(mlir::affine::mergeOffsetsSizesAndStrides(
          builder, loc, load.getMixedOffsets(), load.getMixedSizes(),
          load.getMixedStrides(), llvm::SmallBitVector(offsets.size()), offsets,
          sizes, strides, merged_offsets, merged_sizes, merged_strides))) {
    return llvm::failure();
  }

  auto new_load = mlir::ktdp_lowering::LoadOp::create(
      builder, load.getLoc(), load.getSource(), merged_offsets, merged_sizes,
      merged_strides, load.getType().getElementType());
  new_load->setDiscardableAttrs(load->getDiscardableAttrDictionary());
  return new_load;
}

/// Folds `ktdp_lowering.load` into its only `tensor.extract_slice` user.
struct FoldSliceIntoLoad
    : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::tensor::ExtractSliceOp extract,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Other users of the load would load the data twice.
    auto load =
        extract.getSource().getDefiningOp<mlir::ktdp_lowering::LoadOp>();
    if (!load || !load->hasOneUse()) {
      return rewriter.notifyMatchFailure(extract, "load has multiple users");
    }

    auto new_load =
        createLoad(rewriter, load, getSlice(extract), extract->getLoc());
    if (failed(new_load)) {
      return rewriter.notifyMatchFailure(extract, "can not merge slices");
    }
    rewriter.replaceOp(extract, *new_load);
    rewriter.eraseOp(load);
    return llvm::success();
  }
};

}  // namespace

void scheduler::populateDataMovementCoalescingPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<FoldReshapeIntoFifoRead, FoldReshapeIntoFifoWrite,
               FoldSliceIntoLoad>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// scheduler::populateKTDPTilingPatterns
//===----------------------------------------------------------------------===//

namespace {

/// Moves a via hop after a reshape of its data, such that via hops come after
/// the reshapes and consecutive vias can be folded.
struct MoveViaAfterReshape : mlir::OpRewritePattern<mlir::ktdf::ViaOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::ViaOp via,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    if (!via->hasOneUse() || !isReshape(*via->user_begin())) {
      return rewriter.notifyMatchFailure(via, "not reshaped");
    }

    auto* const reshape = *via->user_begin();
    const auto reshaped = reshape->getResult(0);
    rewriter.modifyOpInPlace(
        reshape, [&]() { reshape->setOperand(0, via.getOperand()); });
    rewriter.replaceAllUsesWith(reshaped, via.getResult());
    rewriter.moveOpAfter(via, reshape);
    rewriter.modifyOpInPlace(via, [&]() {
      via->setOperand(0, reshaped);
      via.getResult().setType(llvm::cast<mlir::ShapedType>(reshaped.getType()));
    });
    return llvm::success();
  }
};

/// Canonicalizes the via @p hops , remapping their memory spaces using
/// @p mem_space_map and dropping adjacent duplicates.
void canonicalizeHops(llvm::SmallVectorImpl<mlir::Attribute>& hops,
                      const AttrMapping& mem_space_map) {
  for (auto& hop : hops) {
    hop = mem_space_map.map(hop);
  }
  hops.erase(llvm::unique(hops), hops.end());
}

struct TileLoad : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  explicit TileLoad(mlir::MLIRContext* context,
                    const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::tensor::ExtractSliceOp extract,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    // Vias come after reshapes, and consecutive vias are folded.
    mlir::Value source = extract.getSource();
    auto via = source.getDefiningOp<mlir::ktdf::ViaOp>();
    if (via) {
      source = via.getOperand();
    }

    // Map the slice through the reshapes onto the loaded tensor.
    auto slice = getSlice(extract);
    for (auto* op = source.getDefiningOp(); op && isReshape(op);
         op = source.getDefiningOp()) {
      auto mapped = mapSlice(
          rewriter, slice,
          llvm::cast<mlir::ShapedType>(op->getResult(0).getType()).getShape(),
          llvm::cast<mlir::ShapedType>(op->getOperand(0).getType()).getShape());
      if (failed(mapped)) {
        return rewriter.notifyMatchFailure(op, "unsupported reshape");
      }
      slice = std::move(*mapped);
      source = op->getOperand(0);
    }
    auto load = source.getDefiningOp<mlir::ktdp::LoadOp>();
    if (!load) {
      return rewriter.notifyMatchFailure(extract, "not a data tile source");
    }

    // Lower the load of the entire access tile, and fold the slice into it.
    const auto element_type =
        llvm::cast<mlir::ShapedType>(load->getResult(0).getType())
            .getElementType();
    auto new_load = mlir::ktdp_lowering::LoadOp::create(
        rewriter, load.getLoc(), load.getAccessTile(), slice, element_type);
    setThrottle(new_load, getThrottle(extract->getOpResult(0)));

    // Reshape the tile, and add the hops back in, counting the load as one.
    mlir::Value loaded =
        reshapeTo(rewriter, extract.getLoc(), new_load.getResult(),
                  extract.getType().getShape());
    llvm::SmallVector<mlir::Attribute> hops{
        getMemorySpace(load.getAccessTile())};
    if (via) {
      llvm::append_range(hops, via.getHops());
    }
    canonicalizeHops(hops, mem_space_map_);
    if (hops.size() > 1) {
      loaded = mlir::ktdf::ViaOp::create(
          rewriter, load.getLoc(), loaded,
          rewriter.getArrayAttr(llvm::ArrayRef(hops).drop_front(1)));
    }

    rewriter.replaceOp(extract, loaded);
    if (load->use_empty()) {
      rewriter.eraseOp(load);
    }
    return llvm::success();
  }

 private:
  const AttrMapping& mem_space_map_;
};

/// Determines whether any op strictly between @p begin and @p end , which must
/// be in this order in the same block, may have memory effects.
[[nodiscard]] auto hasMemoryEffectsBetween(mlir::Operation* begin,
                                           mlir::Operation* end) -> bool {
  assert(begin->getBlock() == end->getBlock() && begin->isBeforeInBlock(end));
  return llvm::any_of(
      llvm::make_range(std::next(begin->getIterator()), end->getIterator()),
      [](mlir::Operation& op) { return !mlir::isMemoryEffectFree(&op); });
}

struct TileStore : mlir::OpRewritePattern<mlir::ktdp::StoreOp> {
  explicit TileStore(mlir::MLIRContext* context,
                     const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    // Vias come after reshapes, and consecutive vias are folded.
    mlir::Value data = store.getDataTile();
    auto via = data.getDefiningOp<mlir::ktdf::ViaOp>();
    if (via) {
      data = via.getOperand();
    }
    auto has_reshape = false;
    for (auto* op = data.getDefiningOp(); op && isReshape(op);
         op = data.getDefiningOp()) {
      data = op->getOperand(0);
      has_reshape = true;
    }

    // Find the tiled `insert_slice` producer.
    // NOTE: We wouldn't have to do any of this if we were tiling
    //       ktdp_lowering.store using scf::tileUsingSCF, but then we'd need
    //       to compute tile sizes based on the stores instead of the compute.
    auto nest = findInsertSlice(data);
    if (failed(nest)) {
      return rewriter.notifyMatchFailure(store, "not a data tile source");
    }
    auto insert_slice = nest->insert_slice;
    auto outermost = nest->outermost;

    // Map the inserted slice onto the stored tensor.
    auto slice = getSlice(insert_slice);
    if (has_reshape) {
      auto mapped =
          mapSlice(rewriter, slice, insert_slice.getDestType().getShape(),
                   store.getDataTile().getType().getShape());
      if (failed(mapped)) {
        return rewriter.notifyMatchFailure(store, "unsupported reshape");
      }
      slice = std::move(*mapped);
    }
    if (has_reshape && !nest->is_update) {
      return rewriter.notifyMatchFailure(insert_slice,
                                         "can not reshape inserted tile");
    }

    // The store has to sink into the loop nest, and for that it might have to
    // move its access tile computation.
    if (outermost) {
      if (nest->has_inner_user) {
        return rewriter.notifyMatchFailure(store, "data tile has other users");
      }
      if (outermost->getBlock() != store->getBlock()) {
        return rewriter.notifyMatchFailure(store, "not after the loop nest");
      }
      if (hasMemoryEffectsBetween(outermost, store)) {
        return rewriter.notifyMatchFailure(store,
                                           "memory effects before store");
      }
      if (mlir::failed(mlir::moveValueDefinitions(
              rewriter, store.getAccessTile(), outermost))) {
        return rewriter.notifyMatchFailure(store, "can not move access tile");
      }
    }

    // Determine the value to store. Conservatively, this must be the result of
    // insert_slice, since we didn't check how that is computed.
    mlir::Value stored = insert_slice.getResult();
    if (nest->is_update) {
      stored = insert_slice.getSource();
      rewriter.replaceAllUsesWith(insert_slice.getResult(),
                                  insert_slice.getDest());
    }

    // Reshape the tile, and add the hops back in, counting the store as one.
    rewriter.setInsertionPointAfter(insert_slice);
    if (has_reshape) {
      stored =
          reshapeTo(rewriter, store.getLoc(), stored, getSliceShape(slice));
    }
    llvm::SmallVector<mlir::Attribute> hops;
    if (via) {
      llvm::append_range(hops, via.getHops());
    }
    hops.push_back(getMemorySpace(store.getAccessTile()));
    canonicalizeHops(hops, mem_space_map_);
    if (hops.size() > 1) {
      stored = mlir::ktdf::ViaOp::create(
          rewriter, store.getLoc(), stored,
          rewriter.getArrayAttr(llvm::ArrayRef(hops).drop_back(1)));
    }

    // Create the `ktdp_lowering.store` that captures the applied tiling.
    auto new_store = mlir::ktdp_lowering::StoreOp::create(
        rewriter, store.getLoc(), stored, store.getAccessTile(), slice);
    setThrottle(new_store, getThrottle(insert_slice.getSourceMutable()));

    // The loop nest no longer produces the data, and remaining users must now
    // load it again from memory.
    if (outermost && nest->is_update) {
      rewriter.setInsertionPointAfter(outermost);
      auto reloaded = mlir::ktdp::LoadOp::create(rewriter, store.getLoc(),
                                                 store.getDataTile().getType(),
                                                 store.getAccessTile());
      rewriter.replaceAllUsesWith(
          nest->result,
          reshapeTo(
              rewriter, store.getLoc(), reloaded,
              llvm::cast<mlir::ShapedType>(nest->result.getType()).getShape()));
    }
    rewriter.eraseOp(store);
    return llvm::success();
  }

 private:
  /// Loop nest that yields the result of a `tensor.insert_slice`.
  struct LoopNest {
    mlir::tensor::InsertSliceOp insert_slice;
    /// Outermost loop of the nest, if any.
    mlir::LoopLikeOpInterface outermost;
    /// Result of outermost that yields the data, if any.
    mlir::OpResult result;
    /// Whether insert_slice updates the iter_arg that it is yielded to.
    bool is_update = false;
    /// Whether the data is used otherwise inside of the loop nest.
    bool has_inner_user = false;
  };

  /// Walks into the loop nest that produces @p data to the
  /// `tensor.insert_slice` that it yields.
  ///
  /// @retval failure   @p data is not produced by such a loop nest.
  [[nodiscard]] static auto findInsertSlice(mlir::Value data)
      -> llvm::FailureOr<LoopNest> {
    LoopNest nest;
    mlir::BlockArgument iter_arg;
    auto source = llvm::dyn_cast<mlir::OpResult>(data);
    while (source) {
      auto loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(source.getOwner());
      if (!loop) {
        break;
      }
      iter_arg = loop.getTiedLoopRegionIterArg(source);
      auto* const yielded =
          iter_arg ? loop.getTiedLoopYieldedValue(iter_arg) : nullptr;
      if (!yielded) {
        return llvm::failure();
      }
      if (!nest.outermost) {
        nest.outermost = loop;
        nest.result = source;
      }
      source = llvm::dyn_cast<mlir::OpResult>(yielded->get());
      nest.has_inner_user |= source && !source.hasOneUse();
    }

    nest.insert_slice =
        source ? llvm::dyn_cast<mlir::tensor::InsertSliceOp>(source.getOwner())
               : nullptr;
    if (!nest.insert_slice) {
      return llvm::failure();
    }
    nest.is_update = iter_arg && nest.insert_slice.getDest() == iter_arg;
    return nest;
  }

  const AttrMapping& mem_space_map_;
};

}  // namespace

void scheduler::populateKTDPTilingPatterns(mlir::RewritePatternSet& patterns,
                                           const AttrMapping& mem_space_map) {
  patterns.add<MoveViaAfterReshape>(patterns.getContext());
  patterns.add<TileLoad, TileStore>(patterns.getContext(), mem_space_map);
}

//===----------------------------------------------------------------------===//
// scheduler::populateConvertToDataTransferPatterns
//===----------------------------------------------------------------------===//

namespace {

struct LowerLoadToFifo : mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::WriteToFifoOp write,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto load = write.getData().getDefiningOp<mlir::ktdp_lowering::LoadOp>();
    if (!load) {
      return rewriter.notifyMatchFailure(write, "does not read from memory");
    }
    auto memref = mlir::dyn_cast<MemRef>(load.getSource());
    if (!memref) {
      return rewriter.notifyMatchFailure(load, "not bufferized");
    }

    // Dimensions of stride zero broadcast a single loaded element.
    auto slice = getSlice(load);
    for (auto& range : slice) {
      if (mlir::isConstantIntValue(range.stride, 0)) {
        range.size = rewriter.getIndexAttr(1);
      }
    }
    const auto source = createTransferAccess(rewriter, slice);
    if (failed(source)) {
      return rewriter.notifyMatchFailure(load, "strided load");
    }
    const auto& [source_map, source_offsets, source_sizes] = *source;

    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({load.getLoc(), write.getLoc()}), memref,
        source_map, source_offsets, source_sizes, write.getFifoSlot(), {}, {},
        load.getMixedSizes());
    transfer->setDiscardableAttrs(load->getRawDictionaryAttrs());
    rewriter.eraseOp(write);
    return llvm::success();
  }
};

struct LowerStoreFromFifo
    : mlir::OpRewritePattern<mlir::ktdp_lowering::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto read = store.getSource().getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
    if (!read) {
      return rewriter.notifyMatchFailure(store, "does not read from FIFO");
    }
    if (!read->hasOneUse()) {
      return read.emitError("has multiple uses");
    }
    auto memref = mlir::dyn_cast<MemRef>(store.getDest());
    if (!memref) {
      return rewriter.notifyMatchFailure(store, "not bufferized");
    }

    const auto dest = createTransferAccess(rewriter, getSlice(store));
    if (failed(dest)) {
      return rewriter.notifyMatchFailure(store, "strided store");
    }
    const auto& [map, ivs, sizes] = *dest;
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({read.getLoc(), store.getLoc()}),
        read.getFifoSlot(), {}, {}, sizes, memref, map, ivs, sizes);
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);
    rewriter.eraseOp(read);
    return llvm::success();
  }
};

struct LowerStoreFromLoad
    : mlir::OpRewritePattern<mlir::ktdp_lowering::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Ignore any sequence of `tensor.(collapse|expand)_shape`, since these are
    // order-preserving and thus folded into the `ktdf.data_transfer`.
    mlir::Value source = store.getSource();
    for (auto* op = source.getDefiningOp(); op && isReshape(op);
         op = source.getDefiningOp()) {
      source = op->getOperand(0);
    }
    auto load = source.getDefiningOp<mlir::ktdp_lowering::LoadOp>();
    if (!load) {
      return rewriter.notifyMatchFailure(store, "does not load from memory");
    }

    const auto load_access = createTransferAccess(rewriter, getSlice(load));
    const auto store_access = createTransferAccess(rewriter, getSlice(store));
    if (failed(load_access) || failed(store_access)) {
      return rewriter.notifyMatchFailure(store, "strided transfer");
    }
    const auto& [load_map, load_ivs, load_sizes] = *load_access;
    const auto& [store_map, store_ivs, store_sizes] = *store_access;
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({load.getLoc(), store.getLoc()}),
        load.getSource(), load_map, load_ivs, load_sizes, store.getDest(),
        store_map, store_ivs, store_sizes);
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);
    return llvm::success();
  }
};

/// Converts `ktdf.write_to_fifo(ktdp_lowering.ind_load)` to a gather
/// `ktdf.ind_data_transfer` from the base memref into the FIFO slot.
struct LowerIndLoadToFifo : mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::WriteToFifoOp write,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto load = write.getData().getDefiningOp<mlir::ktdp_lowering::IndLoadOp>();
    if (!load) {
      return rewriter.notifyMatchFailure(
          write, "does not read indirectly from memory");
    }
    const auto type = load.getResult().getType();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(load, "expected static shape");
    }

    const auto src_access = createTransferAccess(rewriter, getSlice(load));
    if (failed(src_access)) {
      return rewriter.notifyMatchFailure(load, "strided transfer");
    }
    const auto [source_map, source_offsets, source_sizes] = *src_access;
    auto transfer = mlir::ktdf::IndDataTransferOp::create(
        rewriter, rewriter.getFusedLoc({load.getLoc(), write.getLoc()}),
        load.getIndAddrBuf(), load.getIndAddrBufIndex(), load.getBase(),
        source_map, source_offsets, source_sizes, mlir::Value{}, mlir::Value{},
        write.getFifoSlot(), mlir::AffineMap{}, mlir::ValueRange{},
        mlir::getAsIndexOpFoldResult(rewriter.getContext(), type.getShape()));
    transfer->setDiscardableAttrs(load->getRawDictionaryAttrs());
    rewriter.eraseOp(write);
    return llvm::success();
  }
};

/// Converts `ktdp_lowering.ind_store(ktdf.read_from_fifo)` to a scatter
/// `ktdf.ind_data_transfer` from the FIFO slot into the base memref.
struct LowerIndStoreFromFifo
    : mlir::OpRewritePattern<mlir::ktdp_lowering::IndStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::IndStoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    llvm::SmallVector<mlir::Operation*> reshapes;
    auto read = store.getSource().getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
    if (!read) {
      return rewriter.notifyMatchFailure(store, "does not read from FIFO");
    }
    if (!read->hasOneUse()) {
      return read.emitError("has multiple uses");
    }
    if (!llvm::all_of(reshapes, [](auto* op) { return op->hasOneUse(); })) {
      return rewriter.notifyMatchFailure(store, "reshape has multiple uses");
    }
    const auto type = store.getSource().getType();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(store, "expected static shape");
    }

    const auto dest_access = createTransferAccess(rewriter, getSlice(store));
    if (failed(dest_access)) {
      return rewriter.notifyMatchFailure(store, "strided transfer");
    }
    const auto [dest_map, dest_offsets, dest_sizes] = *dest_access;
    auto transfer = mlir::ktdf::IndDataTransferOp::create(
        rewriter, rewriter.getFusedLoc({read.getLoc(), store.getLoc()}),
        mlir::Value{}, mlir::Value{}, read.getFifoSlot(), mlir::AffineMap{},
        mlir::ValueRange{},
        mlir::getAsIndexOpFoldResult(rewriter.getContext(), type.getShape()),
        store.getIndAddrBuf(), store.getIndAddrBufIndex(), store.getBase(),
        dest_map, dest_offsets, dest_sizes);
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);
    rewriter.eraseOp(read);
    return llvm::success();
  }
};

}  // namespace

void scheduler::populateConvertToDataTransferPatterns(
    mlir::RewritePatternSet& patterns) {
  populateDataMovementCoalescingPatterns(patterns);
  patterns.add<LowerLoadToFifo, LowerStoreFromFifo, LowerStoreFromLoad,
               LowerIndLoadToFifo, LowerIndStoreFromFifo>(
      patterns.getContext());
}
