//===-- KTIRMapAndTile.cpp --------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <cstdint>

#include "Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Utils/LoopSimplification.h"

#define PASS_NAME "ktir-map-and-tile"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Map and Tile pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRMAPANDTILEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

using I64Vec = llvm::SmallVector<int64_t>;

struct KTIRMapAndTilePass
    : public impl::KTIRMapAndTilePassBase<KTIRMapAndTilePass> {
  using KTIRMapAndTilePassBase<KTIRMapAndTilePass>::KTIRMapAndTilePassBase;

  void runOnOperation() override;
};

/// Given a shape and a target vector_length, compute per-dimension tile sizes
/// working rightmost-first until the product of covered dims reaches
/// vector_length.  Uncovered dims (leftmost) get tile size 1 so a loop is
/// generated for them; covered dims get the largest value that divides both
/// the dim size and the remaining vector-length product.
[[nodiscard]] auto determineTileSizes(
    llvm::ArrayRef<int64_t> shape, int64_t vector_length,
    llvm::SmallVectorImpl<int64_t>& tile_sizes) -> llvm::LogicalResult {
  const int64_t rank = shape.size();

  // Find how many rightmost dims are needed to cover vector_length.
  int64_t product = 1;
  unsigned covered_dims = 0;
  while (covered_dims < shape.size()) {
    const auto size = shape[shape.size() - covered_dims - 1];
    if (mlir::ShapedType::isDynamic(size)) {
      // There weren't enough statically known dims to cover the vector size.
      return llvm::failure();
    }

    product *= size;
    covered_dims++;
    if (product >= vector_length) {
      break;
    }
  }

  // Uncovered dims (leftmost) get tile size 1 — a loop is generated that
  // iterates over the full extent one element at a time.
  tile_sizes.resize(rank, 1);

  // Covered dims get a size that divides both the dim and the remaining
  // product.
  int64_t remaining = vector_length;
  for (int64_t i = rank - 1; i >= rank - covered_dims; --i) {
    const auto size = std::min(std::gcd(shape[i], remaining), shape[i]);
    tile_sizes[i] = size;
    remaining = std::max<int64_t>(remaining / size, 1);
  }

  return llvm::success();
}

/// Maps @p op to the default compute resource, unless it already is, and
/// annotates the vector width for @p element_type on it as the throttle.
///
/// @return The vector width.
auto mapToCompute(mlir::linalg::LinalgOp op, mlir::ktdf_arch::Mapping& mapping,
                  mlir::Type element_type) -> int64_t {
  auto compute =
      mapping.getOrMap(op, mapping.byKind().getDefaultCompute().getKind());
  assert(compute && "no default compute resource");

  LDBG() << "  compute: " << mlir::OpWithFlags(compute, kSkipRegions);

  // Determine the desired vector width on the compute resource.
  const auto simd_feature =
      compute.getFeature<mlir::ktdf_arch::feature::SIMD>();
  const auto vector_length =
      std::max(simd_feature.getLanes(element_type), int64_t{1});

  // Annotate the vector width as the throttle for this operation.
  setThrottle(op, vector_length);
  return vector_length;
}

auto determineTileSizes(mlir::linalg::LinalgOp op,
                        mlir::ktdf_arch::Mapping& mapping,
                        llvm::SmallVectorImpl<int64_t>& tile_sizes)
    -> llvm::LogicalResult {
  if (!op.hasPureTensorSemantics() || op->getNumResults() == 0) {
    return op->emitError("tiling error: expected pure tensor semantics");
  }
  const auto result_type =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!llvm::all_equal(op->getResultTypes())) {
    op.emitWarning("tiling heterogeneous results is not supported");
  }

  LDBG() << "determining tile sizes for "
         << mlir::OpWithFlags(op, kSkipRegions);

  // Map the op to a compute resource, if it isn't already.
  const auto vector_length =
      mapToCompute(op, mapping, result_type.getElementType());

  // The tile-sizes vector must cover every loop dimension (getNumLoops()).
  // Reduction dimensions are skipped (tile size = 0) so the tiling infra does
  // not create a loop for them.  Only parallel dimensions are tiled, using the
  // same rightmost-first shape-based algorithm as before.
  // Collect the loop bounds for each dimension by querying the op directly.
  // getStaticLoopRanges() scans all operand indexing maps so it works
  // regardless of whether any single operand has a full-rank map (e.g. when
  // an input has a broadcast/projection indexing map that skips some dims).
  const auto loop_ranges = op.getStaticLoopRanges();
  I64Vec parallel_loop_indices;
  I64Vec parallel_shape;
  for (auto [index, type] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (type == mlir::utils::IteratorType::parallel) {
      parallel_loop_indices.push_back(index);
      parallel_shape.push_back(loop_ranges[index]);
    }
  }

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "  parallel_shape: {";
    llvm::interleaveComma(parallel_shape, os);
    os << "}";
  });

  // Compute tile sizes for the parallel dimensions.
  I64Vec parallel_tile_sizes;
  if (failed(::determineTileSizes(parallel_shape, vector_length,
                                  parallel_tile_sizes))) {
    return op->emitError("tiling error: inner dimensions not tileable");
  }

  // Assemble the final vector: 0 for reduction dims, computed size for
  // parallel dims.
  tile_sizes.resize(loop_ranges.size(), 0);
  for (int64_t k = 0; k < static_cast<int64_t>(parallel_loop_indices.size());
       ++k) {
    tile_sizes[parallel_loop_indices[k]] = parallel_tile_sizes[k];
  }

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "  tile_sizes: {";
    llvm::interleaveComma(tile_sizes, os);
    os << "}";
  });

  return llvm::success();
}

auto compressInput(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op,
                   mlir::OpOperand* input) -> llvm::LogicalResult {
  const auto type =
      llvm::dyn_cast<mlir::RankedTensorType>(input->get().getType());
  const auto bcast_dims = op.getMatchingIndexingMap(input).getBroadcastDims();
  if (!type || bcast_dims.empty()) {
    return llvm::failure();
  }

  const auto zero = rewriter.getI64IntegerAttr(0);
  const auto one = rewriter.getI64IntegerAttr(1);

  llvm::SmallVector<mlir::Range> ranges;
  ranges.reserve(type.getRank());
  for (auto dim : llvm::iota_range<int64_t>(0, type.getRank(), false)) {
    const auto size =
        mlir::tensor::getMixedSize(rewriter, op.getLoc(), input->get(), dim);
    ranges.push_back({zero, size, one});
  }

  for (auto dim : bcast_dims) {
    ranges[dim] = {zero, one, zero};
  }

  input->set(mlir::tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                  input->get(), ranges));
  return llvm::success();
}

auto compressInputs(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op)
    -> llvm::LogicalResult {
  rewriter.startOpModification(op);
  auto changed = false;

  for (auto* const input : op.getDpsInputOperands()) {
    changed |= succeeded(compressInput(rewriter, op, input));
  }

  if (!changed) {
    rewriter.cancelOpModification(op);
    return llvm::failure();
  }

  LDBG() << "compressed " << mlir::OpWithFlags(op, kSkipRegions);
  rewriter.finalizeOpModification(op);
  return llvm::success();
}

auto tile(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp& op,
          mlir::ktdf_arch::Mapping& mapping)
    -> llvm::FailureOr<mlir::scf::LoopVector> {
  rewriter.setInsertionPoint(op);

  // Remove needless dependencies on broadcasted dimensions.
  std::ignore = compressInputs(rewriter, op);

  // Determine tile sizes from output operand shape (needed for loop creation).
  I64Vec tile_sizes;
  if (failed(determineTileSizes(op, mapping, tile_sizes))) {
    return llvm::failure();
  }
  if (tile_sizes.empty()) {
    op.emitError("tiling error: no tileable dimensions");
    return llvm::failure();
  }

  // Tile the linalg.generic operation with the computed tile sizes
  mlir::linalg::LinalgTilingOptions tiling_options;
  tiling_options.setTileSizes(tile_sizes);
  tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
  const auto maybe_tiling = tileLinalgOp(rewriter, op, tiling_options);
  if (failed(maybe_tiling)) {
    op->emitError("tiling error: unable to create loops");
    return llvm::failure();
  }

  // Annotate the created loops with their iterator type.
  {
    unsigned loop_index = 0;
    const auto iterator_types = op.getIteratorTypesArray();
    for (auto [dim, tile_size] : llvm::enumerate(tile_sizes)) {
      if (tile_size != 0) {
        // FIXME: Put the attribute name somewhere else.
        maybe_tiling->loops[loop_index++]->setAttr(
            "loop_type",
            getLoopTypeAttr(rewriter.getContext(), iterator_types[dim]));
      }
    }
  }

  // Replace the old with the tiled op.
  maybe_tiling->op->setDiscardableAttrs(op->getRawDictionaryAttrs());
  rewriter.replaceOp(op, maybe_tiling->tensorResults);
  op = maybe_tiling->op;

  return llvm::map_to_vector(maybe_tiling->loops, [](mlir::Operation* op) {
    return llvm::cast<mlir::scf::ForOp>(op);
  });
}

void bubbleUpExtractSlice(mlir::RewriterBase& rewriter,
                          mlir::tensor::ExtractSliceOp op,
                          const AttrMapping& mem_space_map) {
  while (auto via = op.getSource().getDefiningOp<mlir::ktdf::ViaOp>()) {
    rewriter.modifyOpInPlace(op, [&]() { op.setOperand(0, via.getOperand()); });
    rewriter.modifyOpInPlace(via, [&]() {
      via.getResult().setType(op.getType());
      via.setOperand(op);
    });
    rewriter.replaceAllUsesExcept(op, via, via);
    rewriter.moveOpBefore(op, via);
  }

  if (auto load = op.getSource().getDefiningOp<mlir::ktdp_lowering::LoadOp>();
      load) {
    rewriter.modifyOpInPlace(op, [&]() {
      std::ignore = mlir::ktdf_arch::Mappable::setMapsTo(
          op, mlir::ktdf_arch::MapsToAttr::get(
                  rewriter.getContext(),
                  {llvm::cast<mlir::ktdf_arch::ResourceSpecAttr>(
                      mem_space_map.map(getMemorySpace(load.getSource())))}));
    });
  }
}

[[nodiscard]] auto isBroadcast(mlir::Value value, int64_t dim) -> bool {
  while (auto via = value.getDefiningOp<mlir::ktdf::ViaOp>()) {
    value = via.getOperand();
  }
  auto slice = value.getDefiningOp<mlir::OffsetSizeAndStrideOpInterface>();
  if (!slice) {
    return false;
  }
  const auto strides = slice.getMixedStrides();
  if (static_cast<int64_t>(strides.size()) !=
      llvm::cast<mlir::ShapedType>(value.getType()).getRank()) {
    return false;
  }
  return mlir::isZeroInteger(strides[dim]) &&
         !mlir::isConstantIntValue(slice.getMixedSizes()[dim], 1);
}

[[nodiscard]] auto getReassociationIndices(
    mlir::Operation* op,
    llvm::SmallVectorImpl<mlir::ReassociationIndices>& groups,
    bool& is_collapse) -> mlir::LogicalResult {
  if (auto collapse = llvm::dyn_cast<mlir::tensor::CollapseShapeOp>(op);
      collapse) {
    groups = collapse.getReassociationIndices();
    is_collapse = true;
    return llvm::success();
  }

  if (auto expand = llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(op); expand) {
    groups = expand.getReassociationIndices();
    is_collapse = false;
    return llvm::success();
  }

  return llvm::failure();
}

/// Bubbles `tensor.extract_slice` up through reshaping operations.
struct BubbleUpSliceThroughReshape
    : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::tensor::ExtractSliceOp slice,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto* reshape = slice.getSource().getDefiningOp();
    bool is_collapse;
    llvm::SmallVector<mlir::ReassociationIndices> groups;
    // A reshape of static result has a static source, and one without groups
    // is from or to rank 0, where a sliced dimension has nowhere to go.
    const auto extents = slice.getSourceType().getShape();
    if (!reshape ||
        failed(getReassociationIndices(reshape, groups, is_collapse)) ||
        groups.empty() || !slice.getSizes().empty() ||
        !slice.getStrides().empty() ||
        slice.getResultType().getRank() != slice.getSourceType().getRank() ||
        !slice.getSourceType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(slice, "not a slice of a reshape");
    }

    auto sizes = llvm::to_vector(
        llvm::cast<mlir::ShapedType>(reshape->getOperand(0).getType())
            .getShape());
    llvm::SmallVector<mlir::OpFoldResult> offsets(sizes.size(),
                                                  rewriter.getIndexAttr(0));
    llvm::SmallVector<int64_t> strides(sizes.size(), 1);
    llvm::SmallVector<bool> sliced(sizes.size(), false);
    for (auto [index, group] : llvm::enumerate(groups)) {
      const mlir::ReassociationIndices single{static_cast<int64_t>(index)};
      const auto& target = is_collapse ? group : single;
      for (auto from : is_collapse ? single : group) {
        const auto size = slice.getStaticSizes()[from];
        const auto stride = slice.getStaticStrides()[from];
        if (size == extents[from] && (stride == 1 || extents[from] == 1)) {
          continue;
        }
        const auto to = llvm::find_if(llvm::reverse(target), [&](int64_t dim) {
          return !sliced[dim] && sizes[dim] == extents[from];
        });
        if (to == target.rend()) {
          return rewriter.notifyMatchFailure(slice, "unsupported reshape");
        }
        std::tie(offsets[*to], sizes[*to], strides[*to]) =
            std::tuple(slice.getMixedOffsets()[from], size, stride);
        sliced[*to] = true;
      }
    }

    // Slice the source, then reshape with the same op unless it is shared.
    const auto type = slice.getResultType();
    if (!reshape->hasOneUse()) {
      reshape = rewriter.clone(*reshape);
    }
    rewriter.modifyOpInPlace(slice, [&]() {
      const auto [static_offsets, dynamic_offsets] =
          mlir::decomposeMixedValues(offsets);
      slice.getSourceMutable().assign(reshape->getOperand(0));
      slice.getOffsetsMutable().assign(dynamic_offsets);
      slice.setStaticOffsets(static_offsets);
      slice.setStaticSizes(sizes);
      slice.setStaticStrides(strides);
      slice.getResult().setType(type.clone(sizes));
    });
    rewriter.replaceAllUsesWith(slice.getResult(), reshape->getResult(0));
    rewriter.moveOpAfter(reshape, slice);
    rewriter.modifyOpInPlace(reshape, [&]() {
      reshape->setOperand(0, slice.getResult());
      reshape->getResult(0).setType(type);
      if (auto expanded =
              llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(reshape)) {
        expanded.setStaticOutputShape(type.getShape());
      }
    });
    return llvm::success();
  }
};

struct BroadcastInputs : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  explicit BroadcastInputs(mlir::MLIRContext* context,
                           const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::linalg::GenericOp op,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    const auto throttle_attr =
        op->getAttrOfType<mlir::ktdf_arch::I64Attr>(kThrottleAttrName);
    if (!throttle_attr) {
      return rewriter.notifyMatchFailure(op, "missing throttle attribute");
    }
    const auto vector_lanes = throttle_attr.getValue();
    const auto zero = rewriter.getIndexAttr(0);
    const auto one = rewriter.getIndexAttr(1);

    rewriter.startOpModification(op);
    auto changed = false;
    for (auto* const input : op.getDpsInputOperands()) {
      auto type =
          llvm::dyn_cast<mlir::RankedTensorType>(input->get().getType());
      const auto map = op.getMatchingIndexingMap(input);
      if (!type || map.getNumDims() == 0) {
        continue;
      }

      // An operand whose map does not involve the innermost, vectorized loop is
      // broadcast along a unit dimension that is added, unless its innermost
      // dimension already is one. This is a rank-0 operand, or one whose
      // trailing unit dimension was dropped.
      auto bcast_dims = map.getBroadcastDims();
      const auto add_unit_dim =
          !map.isFunctionOfDim(map.getNumDims() - 1) &&
          (bcast_dims.empty() || bcast_dims.back() != type.getRank() - 1);
      if (add_unit_dim) {
        bcast_dims.push_back(type.getRank());
      }
      const auto rank = type.getRank() + (add_unit_dim ? 1 : 0);
      if (bcast_dims.empty() || bcast_dims.back() != rank - 1 ||
          (!add_unit_dim && isBroadcast(input->get(), bcast_dims.back()))) {
        continue;
      }

      int64_t elements = 1;
      for (auto [dim, size] : llvm::enumerate(type.getShape())) {
        if (llvm::is_contained(bcast_dims, dim)) {
          continue;
        }
        if (mlir::ShapedType::isDynamic(size) ||
            llvm::MulOverflow(elements, size, elements) != 0) {
          elements = 0;
          break;
        }
      }

      // The operand fills one vector, so only the lanes the other dimensions
      // leave are spread along the innermost broadcast dimension. A broadcast
      // outside the lanes keeps size 1.
      if (elements <= 0 || vector_lanes % elements != 0) {
        continue;
      }
      const auto fill = vector_lanes / elements;
      if (fill == 1) {
        continue;
      }
      // Already broadcast, possibly by a slice that was bubbled up past a
      // reshape and is therefore not seen by `isBroadcast`.
      if (!add_unit_dim && type.getDimSize(bcast_dims.back()) == fill) {
        continue;
      }

      if (add_unit_dim) {
        llvm::SmallVector<int64_t> shape(type.getShape());
        shape.push_back(1);
        llvm::SmallVector<mlir::ReassociationIndices> reassociation;
        for (auto dim : llvm::seq<int64_t>(type.getRank())) {
          reassociation.push_back({dim});
        }
        if (!reassociation.empty()) {
          reassociation.back().push_back(type.getRank());
        }
        auto expand = mlir::tensor::ExpandShapeOp::create(
            rewriter, op.getLoc(),
            mlir::RankedTensorType::get(shape, type.getElementType()),
            input->get(), reassociation);
        input->set(expand);
        type = expand.getResultType();

        auto maps = op.getIndexingMapsArray();
        auto& expanded = maps[input->getOperandNumber()];
        expanded = expanded.insertResult(rewriter.getAffineConstantExpr(0),
                                         expanded.getNumResults());
        op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      }

      auto sizes =
          mlir::tensor::getMixedSizes(rewriter, op.getLoc(), input->get());
      llvm::SmallVector<mlir::OpFoldResult> offsets(type.getRank(), zero);
      llvm::SmallVector<mlir::OpFoldResult> strides(type.getRank(), one);
      for (auto dim : bcast_dims) {
        sizes[dim] = one;
        strides[dim] = zero;
      }
      sizes[bcast_dims.back()] = rewriter.getIndexAttr(fill);

      auto broadcast = mlir::tensor::ExtractSliceOp::create(
          rewriter, op.getLoc(), input->get(), offsets, sizes, strides);
      input->set(broadcast);
      bubbleUpExtractSlice(rewriter, broadcast, mem_space_map_);
      changed = true;
    }

    if (!changed) {
      rewriter.cancelOpModification(op);
      return rewriter.notifyMatchFailure(op, "nothing to broadcast");
    }

    LDBG() << "broadcasted " << mlir::OpWithFlags(op, kSkipRegions);
    rewriter.finalizeOpModification(op);
    return llvm::success();
  }

 private:
  const AttrMapping& mem_space_map_;
};

}  // namespace

void KTIRMapAndTilePass::runOnOperation() {
  if (disable_this_pass) {
    return;
  }

  mlir::func::FuncOp func = getOperation();
  mlir::IRRewriter rewriter(func);

  // Compute operations already nested in loops (e.g. by
  // IndirectAccessLoopMaterialization) are mapped but not tiled.
  auto computes = llvm::to_vector(func.getOps<mlir::linalg::LinalgOp>());
  llvm::SmallVector<mlir::linalg::LinalgOp> nested_computes;
  func.walk([&](mlir::linalg::LinalgOp op) {
    if (op->getParentOp() != func) {
      nested_computes.push_back(op);
    }
  });
  if (computes.empty() && nested_computes.empty()) {
    return;
  }

  // Obtain the default device and start a mapping.
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());

  // This pass supports remapping of the memory space attributes.
  AttrMapping mem_space_map;
  if (auto map = default_device->getAttrOfType<mlir::ktdf_arch::MapAttr>(
          "mem_space_mapping");
      map) {
    mem_space_map.insert_range(map);
  }

  for (auto op : nested_computes) {
    const auto result_type =
        op->getNumResults() == 1
            ? llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType())
            : nullptr;
    if (!op.hasPureTensorSemantics() || !result_type) {
      op->emitError("mapping error: expected pure tensor semantics");
      signalPassFailure();
      return;
    }

    LDBG() << "mapping nested " << mlir::OpWithFlags(op, kSkipRegions);
    mapToCompute(op, mapping, result_type.getElementType());
  }

  // Tile all the compute operations and put them into loop nests that feature
  // loop-carried dependencies.
  for (auto& op : computes) {
    auto maybe_loops = tile(rewriter, op, mapping);
    if (failed(maybe_loops)) {
      signalPassFailure();
      return;
    }
    if (maybe_loops->empty()) {
      LDBG() << "no loops generated for "
             << mlir::OpWithFlags(op, kSkipRegions);
      continue;
    }
  }

  // Merge `tensor.(insert|extract)_slice` ops, tile `ktdp.(load|store)`,
  // broadcast compute operands back to target vector size and remove all the
  // unneeded `iter_args`.
  {
    mlir::RewritePatternSet patterns(&getContext());
    populateKTDPTilingPatterns(patterns, mem_space_map);
    populateDataMovementCoalescingPatterns(patterns);
    patterns.add<BroadcastInputs>(patterns.getContext(), mem_space_map);
    patterns.add<BubbleUpSliceThroughReshape>(patterns.getContext());
    mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    mlir::tensor::populateFoldTensorEmptyPatterns(patterns);
    populateSimplifyLoopsPatterns(patterns);

    if (failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // Clean up all the unused function-level constants that remain.
  for (auto& empty : llvm::make_early_inc_range(
           llvm::reverse(func.getFunctionBody().front()))) {
    if (mlir::isOpTriviallyDead(&empty)) {
      rewriter.eraseOp(&empty);
    }
  }
}
