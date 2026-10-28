//===-- SimplifyLinalg.cpp --------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Transforms/Utils/SimplifyLinalg.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Analysis/FlatLinearValueConstraints.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Analysis/Presburger/PresburgerSpace.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep

#define PASS_NAME "simplify-linalg"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Simplify Linalg pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_SIMPLIFYLINALGPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

//===----------------------------------------------------------------------===//
// scheduler::matchFill
//===----------------------------------------------------------------------===//

auto scheduler::matchFill(mlir::linalg::GenericOp op)
    -> llvm::FailureOr<mlir::OpFoldResult> {
  if (!op.hasPureTensorSemantics() || op.getNumResults() != 1 ||
      !op.getBody()->without_terminator().empty() ||
      !op.getMatchingIndexingMap(&op.getOutputsMutable()[0]).isPermutation()) {
    return llvm::failure();
  }

  auto value = op.getMatchingYieldValue(&op.getOutputsMutable()[0])->get();
  llvm::SmallVector<int64_t> point;
  if (const auto arg = llvm::dyn_cast<mlir::BlockArgument>(value);
      arg && arg.getOwner() == op.getBody()) {
    auto* const input = op.getMatchingOpOperand(arg);
    const auto map = op.getMatchingIndexingMap(input);
    if (!op.isDpsInput(input) || !map.isConstant()) {
      return llvm::failure();
    }

    value = input->get();
    point = map.getConstantResults();
  }
  if (!value.getParentRegion()->isAncestor(op->getParentRegion())) {
    return llvm::failure();
  }

  mlir::Attribute attr;
  if (!mlir::matchPattern(value, mlir::m_Constant(&attr))) {
    if (llvm::isa<mlir::ShapedType>(value.getType())) {
      return llvm::failure();
    }

    return mlir::OpFoldResult(value);
  }
  if (!llvm::isa<mlir::ShapedType>(value.getType())) {
    return mlir::OpFoldResult(attr);
  }

  const llvm::ArrayRef<uint64_t> index(
      reinterpret_cast<const uint64_t*>(point.data()), point.size());
  const auto elements = llvm::dyn_cast<mlir::ElementsAttr>(attr);
  if (!elements || !elements.isValidIndex(index)) {
    return llvm::failure();
  }
  return mlir::OpFoldResult(elements.getValues<mlir::Attribute>()[index]);
}

//===----------------------------------------------------------------------===//
// scheduler::populateSimplifyLinalgPatterns
//===----------------------------------------------------------------------===//

namespace {

/// Folds an invariant `linalg.generic` fill to `arith.constant` if possible.
struct FoldFill : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::linalg::GenericOp op,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    const auto fill = matchFill(op);
    if (failed(fill)) {
      return rewriter.notifyMatchFailure(op, "not a fill operation");
    }
    const auto result_type =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!result_type.hasStaticShape() ||
        !result_type.getElementType().isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(op, "unsupported result type");
    }
    const auto fill_attr = llvm::dyn_cast_if_present<mlir::Attribute>(*fill);
    if (!fill_attr) {
      return rewriter.notifyMatchFailure(op, "fill value is not constant");
    }

    rewriter.replaceOp(
        op, mlir::arith::ConstantOp::create(
                rewriter, op.getLoc(),
                mlir::DenseElementsAttr::get(result_type, fill_attr)));
    return llvm::success();
  }
};

/// Static strided slice of a single operand dimension.
struct DimSlice {
  int64_t offset;
  int64_t size;
  int64_t stride;
};

/// Hyperrectangular slice of an operand.
struct OperandSlice {
  /// Slice of each dimension, or `std::nullopt` for the whole dimension.
  llvm::SmallVector<std::optional<DimSlice>> dims;
  /// Indexing map into the slice.
  mlir::AffineMap map;
};

using SliceDimFn = llvm::function_ref<std::optional<DimSlice>(
    unsigned idx, mlir::AffineExpr expr)>;

/// Slices the dimensions of @p type indexed through @p map using
/// @p slice_dim .
///
/// Sliced dimensions are rebased to be accessed with unit stride. Slices that
/// would change the inferred @p loop_ranges are discarded.
///
/// @retval failure       No dimension can be sliced.
/// @retval OperandSlice  The slice, which is a proper subset.
[[nodiscard]] auto sliceDims(mlir::AffineMap map,
                             llvm::ArrayRef<int64_t> loop_ranges,
                             mlir::RankedTensorType type, SliceDimFn slice_dim)
    -> llvm::FailureOr<OperandSlice> {
  assert(map.getNumSymbols() == 0 && map.getNumDims() == loop_ranges.size() &&
         map.getNumResults() == type.getRank());

  const auto num_dims = map.getNumDims();
  OperandSlice slice;
  auto exprs = llvm::to_vector(map.getResults());
  auto sliced = false;
  for (auto [idx, expr] : llvm::enumerate(exprs)) {
    auto& dim = slice.dims.emplace_back(slice_dim(idx, expr));
    if (!dim) {
      continue;
    }
    if (dim->offset == 0 && dim->stride == 1 &&
        type.getDimSize(idx) == dim->size) {
      dim.reset();
      continue;
    }

    const auto rebased = mlir::simplifyAffineExpr(
        (expr - dim->offset).floorDiv(dim->stride), num_dims, 0);
    // Loop ranges are inferred from plain dimension accesses.
    if (const auto loop = llvm::dyn_cast<mlir::AffineDimExpr>(rebased);
        loop && loop_ranges[loop.getPosition()] != dim->size) {
      dim.reset();
      continue;
    }

    expr = rebased;
    sliced |= true;
  }
  if (!sliced) {
    return llvm::failure();
  }

  slice.map = mlir::AffineMap::get(num_dims, 0, exprs, map.getContext());
  return slice;
}

/// Computes the smallest hyperrectangle of @p type that is accessed through
/// @p map when iterating over @p loop_ranges .
///
/// Accesses outside of @p type are assumed to be impossible.
///
/// @retval failure       The operand can not be shrunk.
/// @retval OperandSlice  The accessed slice, which is a proper subset.
[[nodiscard]] auto computeAccessedSlice(mlir::AffineMap map,
                                        llvm::ArrayRef<int64_t> loop_ranges,
                                        mlir::RankedTensorType type)
    -> llvm::FailureOr<OperandSlice> {
  using mlir::presburger::BoundType;

  if (map.isConstant()) {
    return llvm::failure();
  }

  const auto num_dims = map.getNumDims();
  mlir::presburger::IntegerRelation access(
      mlir::presburger::PresburgerSpace::getRelationSpace());
  if (mlir::failed(mlir::affine::getRelationFromMap(map, access))) {
    return llvm::failure();
  }

  // Restrict to the iteration domain and the in-bounds indices.
  for (auto [pos, range] : llvm::enumerate(loop_ranges)) {
    access.addBound(BoundType::LB, pos, 0);
    if (!mlir::ShapedType::isDynamic(range)) {
      access.addBound(BoundType::UB, pos, range - 1);
    }
  }
  for (auto [idx, extent] : llvm::enumerate(type.getShape())) {
    access.addBound(BoundType::LB, num_dims + idx, 0);
    if (!mlir::ShapedType::isDynamic(extent)) {
      access.addBound(BoundType::UB, num_dims + idx, extent - 1);
    }
  }
  if (access.isEmpty()) {
    return llvm::failure();
  }

  return sliceDims(
      map, loop_ranges, type,
      [&](unsigned idx, mlir::AffineExpr) -> std::optional<DimSlice> {
        const auto lb =
            access.getConstantBound64(BoundType::LB, num_dims + idx);
        const auto ub =
            access.getConstantBound64(BoundType::UB, num_dims + idx);
        if (!lb || !ub) {
          return std::nullopt;
        }
        return DimSlice{*lb, *ub - *lb + 1, 1};
      });
}

/// Computes the slice of @p type that contains all indices which can be
/// accessed through @p map , such that it is accessed with unit stride.
///
/// Indices accessed through an expression are congruent to its constant term
/// modulo the gcd of all its coefficients, which is the stride.
///
/// @retval failure       No dimension can be normalized.
/// @retval OperandSlice  The strided slice, which is a proper subset.
[[nodiscard]] auto computeStridedSlice(mlir::AffineMap map,
                                       llvm::ArrayRef<int64_t> loop_ranges,
                                       mlir::RankedTensorType type)
    -> llvm::FailureOr<OperandSlice> {
  const auto num_dims = map.getNumDims();
  return sliceDims(
      map, loop_ranges, type,
      [&](unsigned idx, mlir::AffineExpr expr) -> std::optional<DimSlice> {
        const auto extent = type.getDimSize(idx);
        if (mlir::ShapedType::isDynamic(extent)) {
          return std::nullopt;
        }

        llvm::SmallVector<int64_t> flat;
        mlir::FlatLinearConstraints locals;
        if (mlir::failed(mlir::getFlattenedAffineExpr(expr, num_dims, 0, &flat,
                                                      &locals))) {
          return std::nullopt;
        }
        int64_t stride = 0;
        for (const auto coeff : llvm::ArrayRef(flat).drop_back()) {
          stride = std::gcd(stride, coeff);
        }
        if (stride <= 1) {
          return std::nullopt;
        }

        const auto offset = llvm::mod(flat.back(), stride);
        if (offset >= extent) {
          return std::nullopt;
        }
        return DimSlice{offset, llvm::divideCeilSigned(extent - offset, stride),
                        stride};
      });
}

using ComputeSliceFn = llvm::function_ref<llvm::FailureOr<OperandSlice>(
    mlir::AffineMap, llvm::ArrayRef<int64_t>, mlir::RankedTensorType)>;

/// Replaces the operands of @p generic in-place by slices computed with
/// @p compute .
///
/// Inputs are replaced by an extract slice. Inits are replaced by an extract
/// slice, and the corresponding result is inserted back into the old init.
///
/// @return Whether the IR was modified.
[[nodiscard]] auto sliceOperands(mlir::linalg::GenericOp generic,
                                 mlir::PatternRewriter& rewriter,
                                 ComputeSliceFn compute)
    -> llvm::LogicalResult {
  if (!generic.hasPureTensorSemantics()) {
    return rewriter.notifyMatchFailure(generic, "not pure tensor semantics");
  }

  const auto loc = generic.getLoc();
  const auto materialize =
      [&](mlir::Value source,
          const OperandSlice& slice) -> llvm::SmallVector<mlir::Range> {
    llvm::SmallVector<mlir::Range> ranges;
    for (auto [idx, dim] : llvm::enumerate(slice.dims)) {
      if (!dim) {
        ranges.push_back(
            {rewriter.getIndexAttr(0),
             mlir::tensor::getMixedSize(rewriter, loc, source, idx),
             rewriter.getIndexAttr(1)});
        continue;
      }

      ranges.push_back({rewriter.getIndexAttr(dim->offset),
                        rewriter.getIndexAttr(dim->size),
                        rewriter.getIndexAttr(dim->stride)});
    }
    return ranges;
  };

  const auto loop_ranges = generic.getStaticLoopRanges();
  auto maps = generic.getIndexingMapsArray();
  auto changed = false;
  rewriter.startOpModification(generic);
  for (auto& operand : generic->getOpOperands()) {
    const auto source = operand.get();
    const auto type = llvm::dyn_cast<mlir::RankedTensorType>(source.getType());
    if (!type) {
      continue;
    }
    const auto slice =
        compute(maps[operand.getOperandNumber()], loop_ranges, type);
    if (mlir::failed(slice)) {
      continue;
    }

    auto extract = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, source, materialize(source, *slice));
    operand.set(extract);
    maps[operand.getOperandNumber()] = slice->map;
    changed |= true;
    if (!generic.isDpsInit(&operand)) {
      continue;
    }

    auto result = generic.getTiedOpResult(&operand);
    result.setType(extract.getType());
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(generic);
    auto insert = mlir::tensor::InsertSliceOp::create(
        rewriter, loc, result, source, extract.getMixedOffsets(),
        extract.getMixedSizes(), extract.getMixedStrides());
    rewriter.replaceAllUsesExcept(result, insert, insert);
  }
  if (!changed) {
    rewriter.cancelOpModification(generic);
    return rewriter.notifyMatchFailure(generic, "no operand can be sliced");
  }

  generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
  rewriter.finalizeOpModification(generic);
  return llvm::success();
}

/// Shrinks the operands of a `linalg.generic` to the slices it accesses.
struct ShrinkOperands : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::linalg::GenericOp generic,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    return sliceOperands(generic, rewriter, computeAccessedSlice);
  }
};

/// Normalizes strided operand accesses of a `linalg.generic` to unit stride.
struct NormalizeStrides : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::linalg::GenericOp generic,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    return sliceOperands(generic, rewriter, computeStridedSlice);
  }
};

/// Moves the inits of a `linalg.generic` that are yielded unchanged to the
/// inputs.
///
/// The moved inits are replaced by empty tensors, and the results that were
/// tied to them are replaced by the old inits.
struct DemoteInitToInput : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::linalg::GenericOp generic,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    if (!generic.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(generic, "not pure tensor semantics");
    }

    const auto loc = generic.getLoc();
    auto maps = generic.getIndexingMapsArray();
    auto changed = false;
    rewriter.startOpModification(generic);
    for (unsigned idx = 0, end = generic.getNumDpsInits(); idx < end; ++idx) {
      auto& init = generic.getDpsInitsMutable()[idx];
      const auto init_arg = generic.getMatchingBlockArgument(&init);
      if (generic.getMatchingYieldValue(&init)->get() != init_arg) {
        continue;
      }

      const auto value = init.get();
      const auto type =
          llvm::cast<mlir::RankedTensorType>(init.get().getType());
      const auto map = maps[init.getOperandNumber()];
      const auto result = generic.getTiedOpResult(&init);
      init.set(mlir::tensor::EmptyOp::create(
          rewriter, loc, mlir::tensor::getMixedSizes(rewriter, loc, value),
          type.getElementType(), type.getEncoding()));

      const auto pos = generic.getNumDpsInputs();
      generic.getInputsMutable().append(value);
      maps.insert(maps.begin() + pos, map);
      const auto input_arg = generic.getBody()->insertArgument(
          pos, init_arg.getType(), init_arg.getLoc());

      rewriter.replaceAllUsesWith(init_arg, input_arg);
      rewriter.replaceAllUsesWith(result, value);
      changed |= true;
    }
    if (!changed) {
      rewriter.cancelOpModification(generic);
      return rewriter.notifyMatchFailure(generic, "no unchanged init");
    }

    generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    rewriter.finalizeOpModification(generic);
    return llvm::success();
  }
};

/// Drops unit dimensions of a `linalg.generic`, preserving its discardable
/// attributes.
struct DropUnitDims : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::linalg::GenericOp generic,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    if (!generic.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(generic, "not pure tensor semantics");
    }
    // Encoded tensors can not be reshaped, which aborts after creating ops.
    if (llvm::any_of(generic->getOperandTypes(), [](mlir::Type type) {
          const auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
          return tensor && tensor.getEncoding();
        })) {
      return rewriter.notifyMatchFailure(generic, "encoded tensor operand");
    }

    const auto dropped = mlir::linalg::dropUnitDims(rewriter, generic, {});
    if (mlir::failed(dropped)) {
      return rewriter.notifyMatchFailure(generic, "no unit dims to drop");
    }

    dropped->resultOp->setDiscardableAttrs(
        generic->getDiscardableAttrDictionary());
    rewriter.replaceOp(generic, dropped->replacements);
    return llvm::success();
  }
};

struct SimplifyLinalgPass
    : public impl::SimplifyLinalgPassBase<SimplifyLinalgPass> {
  using SimplifyLinalgPassBase<SimplifyLinalgPass>::SimplifyLinalgPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    mlir::func::FuncOp func = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    populateSimplifyLinalgPatterns(patterns);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

void scheduler::populateSimplifyLinalgPatterns(
    mlir::RewritePatternSet& patterns) {
  auto* const ctx = patterns.getContext();
  patterns.add<FoldFill, ShrinkOperands, NormalizeStrides, DemoteInitToInput,
               DropUnitDims>(ctx);
  mlir::tensor::populateFoldTensorEmptyPatterns(patterns);
  mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  mlir::linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  mlir::linalg::populateEraseUnnecessaryInputsPatterns(patterns);
  ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>()
      ->getCanonicalizationPatterns(patterns);
  mlir::linalg::GenericOp::getCanonicalizationPatterns(patterns, ctx);
}
