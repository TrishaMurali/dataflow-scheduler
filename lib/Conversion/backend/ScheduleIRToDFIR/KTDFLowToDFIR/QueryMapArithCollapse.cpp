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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/QueryMapArithCollapse.h"

#include <cstdint>
#include <limits>

#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace scheduler;

namespace {

// Extract all map values as int64 constants; returns false if any non-constant.
bool allConstantValues(mlir::uniform::DefImmutableMappingOp map,
                       llvm::SmallVectorImpl<int64_t>& out) {
  for (mlir::Value v : map.getValues()) {
    auto c = mlir::getConstantIntValue(v);
    if (!c) return false;
    out.push_back(*c);
  }
  return true;
}

// Evaluate an integer arith binop at compile time. `op` identifies the kind.
int64_t evalBinop(mlir::Operation* op, int64_t lhs, int64_t rhs) {
  if (mlir::isa<mlir::arith::AddIOp>(op)) return lhs + rhs;
  if (mlir::isa<mlir::arith::SubIOp>(op)) return lhs - rhs;
  if (mlir::isa<mlir::arith::MulIOp>(op)) return lhs * rhs;
  if (mlir::isa<mlir::arith::DivUIOp>(op))
    return rhs == 0 ? 0
                    : static_cast<int64_t>(static_cast<uint64_t>(lhs) /
                                           static_cast<uint64_t>(rhs));
  if (mlir::isa<mlir::arith::DivSIOp>(op)) {
    // INT64_MIN / -1 overflows (UB in C++ and in arith.divsi); fold it to 0
    // like the divide-by-zero case.
    if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1))
      return 0;
    return lhs / rhs;
  }
  if (mlir::isa<mlir::arith::RemUIOp>(op))
    return rhs == 0 ? 0
                    : static_cast<int64_t>(static_cast<uint64_t>(lhs) %
                                           static_cast<uint64_t>(rhs));
  llvm_unreachable("unsupported arith binop in query-map collapse");
}

// Build query_map(def_immutable_mapping(keys, folded_consts), key) at
// `rewriter`'s insertion point and return its result. The folded constant
// values are materialized HERE via the rewriter (stateless — no builder or
// Value cache survives across greedy rewrites; identical constants across
// program_units are left for a later CSE pass to dedup).
mlir::Value buildFoldedQuery(mlir::PatternRewriter& rewriter,
                             mlir::Location loc, mlir::ValueRange keys,
                             mlir::ArrayRef<int64_t> folded_values,
                             mlir::Value key) {
  llvm::SmallVector<mlir::Value> key_vec(keys.begin(), keys.end());
  llvm::SmallVector<mlir::Value> val_vec;
  val_vec.reserve(folded_values.size());
  for (int64_t v : folded_values) {
    val_vec.push_back(mlir::arith::ConstantIndexOp::create(rewriter, loc, v));
  }
  auto map = mlir::uniform::DefImmutableMappingOp::create(
      rewriter, loc, rewriter.getIndexType(), key_vec, val_vec);
  auto q = mlir::uniform::QueryMapOp::create(
      rewriter, loc, rewriter.getIndexType(), map.getResult(), key);
  return q.getResult();
}

// Rule A: arith.binop(query, const) -> query(map').
template <typename BinOp>
struct FoldConstIntoQuery : public mlir::OpRewritePattern<BinOp> {
  using mlir::OpRewritePattern<BinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      BinOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Value lhs = op.getLhs(), rhs = op.getRhs();
    if (!op.getType().isIntOrIndex()) return mlir::failure();

    // Identify the query operand and the constant operand (either side).
    mlir::uniform::QueryMapOp q;
    bool query_is_lhs = false;
    std::optional<int64_t> cst;
    if (auto ql = lhs.getDefiningOp<mlir::uniform::QueryMapOp>()) {
      q = ql;
      query_is_lhs = true;
      cst = mlir::getConstantIntValue(rhs);
    } else if (auto qr = rhs.getDefiningOp<mlir::uniform::QueryMapOp>()) {
      q = qr;
      query_is_lhs = false;
      cst = mlir::getConstantIntValue(lhs);
    }
    if (!q || !cst) return mlir::failure();

    auto map = q.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
    if (!map) return mlir::failure();
    llvm::SmallVector<int64_t> vals;
    if (!allConstantValues(map, vals)) return mlir::failure();

    // Fold each value through the binop with the constant on the correct side.
    llvm::SmallVector<int64_t> folded_vals;
    folded_vals.reserve(vals.size());
    for (int64_t v : vals) {
      folded_vals.push_back(query_is_lhs ? evalBinop(op, v, *cst)
                                         : evalBinop(op, *cst, v));
    }
    mlir::Value q2 = buildFoldedQuery(rewriter, op.getLoc(), map.getKeys(),
                                      folded_vals, q.getKey());
    rewriter.replaceOp(op, q2);
    return mlir::success();
  }
};

// Rule B: arith.binop(query(mapA, k), query(mapB, k)) over the SAME key ->
// query(mapAB, k) with per-key folded values.
template <typename BinOp>
struct MergeQueriesSameKey : public mlir::OpRewritePattern<BinOp> {
  using mlir::OpRewritePattern<BinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      BinOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op.getType().isIntOrIndex()) return mlir::failure();
    auto qa = op.getLhs().template getDefiningOp<mlir::uniform::QueryMapOp>();
    auto qb = op.getRhs().template getDefiningOp<mlir::uniform::QueryMapOp>();
    if (!qa || !qb) return mlir::failure();
    if (qa.getKey() != qb.getKey()) return mlir::failure();  // same key only

    auto mapA =
        qa.getMap()
            .template getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
    auto mapB =
        qb.getMap()
            .template getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
    if (!mapA || !mapB) return mlir::failure();

    // Both fully constant.
    llvm::SmallVector<int64_t> valsA;
    if (!allConstantValues(mapA, valsA)) return mlir::failure();
    // Pair B's values BY KEY against A's keys.
    llvm::SmallVector<mlir::Value> keysA(mapA.getKeys().begin(),
                                         mapA.getKeys().end());
    llvm::SmallVector<int64_t> folded_vals;
    folded_vals.reserve(keysA.size());
    for (auto [i, key] : llvm::enumerate(keysA)) {
      std::optional<mlir::Value> bval = mapB.getValue(key);
      if (!bval) return mlir::failure();  // key not in B -> bail
      auto bc = mlir::getConstantIntValue(*bval);
      if (!bc) return mlir::failure();
      folded_vals.push_back(evalBinop(op, valsA[i], *bc));
    }
    mlir::Value q2 = buildFoldedQuery(rewriter, op.getLoc(), mapA.getKeys(),
                                      folded_vals, qa.getKey());
    rewriter.replaceOp(op, q2);
    return mlir::success();
  }
};

}  // namespace

mlir::LogicalResult scheduler::collapseArithOverQueryMaps(
    mlir::func::FuncOp func) {
  mlir::MLIRContext* ctx = func.getContext();
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<FoldConstIntoQuery<mlir::arith::AddIOp>,
               FoldConstIntoQuery<mlir::arith::SubIOp>,
               FoldConstIntoQuery<mlir::arith::MulIOp>,
               FoldConstIntoQuery<mlir::arith::DivUIOp>,
               FoldConstIntoQuery<mlir::arith::DivSIOp>,
               FoldConstIntoQuery<mlir::arith::RemUIOp>,
               MergeQueriesSameKey<mlir::arith::AddIOp>,
               MergeQueriesSameKey<mlir::arith::SubIOp>,
               MergeQueriesSameKey<mlir::arith::MulIOp>,
               MergeQueriesSameKey<mlir::arith::DivUIOp>,
               MergeQueriesSameKey<mlir::arith::DivSIOp>,
               MergeQueriesSameKey<mlir::arith::RemUIOp>>(ctx);
  return mlir::applyPatternsGreedily(func, std::move(patterns));
}
