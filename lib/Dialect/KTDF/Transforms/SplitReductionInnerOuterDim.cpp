//===-- SplitReductionInnerOuterDim.cpp -------------------------*- c++ -*-===//
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
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "split-reduction-inner-outer-dim"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_SPLITREDUCTIONINNEROUTERDIMPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Split Reduction Inner/Outer Dim pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// A reduction dim expressed in loop-space together with its size in the input
// tensor type.
// ---------------------------------------------------------------------------
struct ReductionDimInfo {
  unsigned loop_dim;  // index into the iterator-types list
  int64_t size;       // static size of this dim in the input tensor
};

// ---------------------------------------------------------------------------
// Per-candidate analysis result.
// ---------------------------------------------------------------------------
struct CandidateInfo {
  linalg::GenericOp generic_op;
  // Inner dim: exactly one — the rightmost reduction dim in input order.
  SmallVector<ReductionDimInfo> inner_dim;
  // Outer dims: all remaining (leftward) reduction dims, same ordering.
  SmallVector<ReductionDimInfo> outer_dims;
};

// ---------------------------------------------------------------------------
// Collect all reduction dims of `generic_op` ordered by their position in the
// first input tensor (leftmost first), together with their static sizes.
// Returns failure (and emits a debug note) if any reduction dim has a dynamic
// size or a non-trivial (non-AffineDimExpr) map result.
// ---------------------------------------------------------------------------
static LogicalResult collectReductionDims(
    linalg::GenericOp generic_op, SmallVectorImpl<ReductionDimInfo>& dims) {
  auto iter_types = generic_op.getIteratorTypesArray();
  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  ArrayRef<int64_t> shape = input_type.getShape();

  // Iterate input dims left-to-right so the result list is in input order.
  for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(d));
    if (!dim_expr) continue;
    unsigned loop_dim = dim_expr.getPosition();
    if (iter_types[loop_dim] != utils::IteratorType::reduction) continue;

    int64_t sz = shape[d];
    if (sz == ShapedType::kDynamic) {
      LDBG(1) << PASS_NAME ": dynamic reduction dim size — skipping";
      return failure();
    }
    dims.push_back({loop_dim, sz});
  }
  return success();
}

// ---------------------------------------------------------------------------
// Return true when `generic_op` qualifies for the split:
//   1. It has at least two reduction iterator types (one inner + at least
//      one outer).
//   2. The rightmost input dimension maps to a reduction loop dim.
// ---------------------------------------------------------------------------
static bool isEligible(linalg::GenericOp generic_op) {
  auto iter_types = generic_op.getIteratorTypesArray();

  // Condition 1: count reduction dims.
  int num_reductions = 0;
  for (auto it : iter_types)
    if (it == utils::IteratorType::reduction) ++num_reductions;
  if (num_reductions < 2) return false;

  // Condition 2: check that the rightmost input dimension maps to a reduction
  // loop dim (via the input's indexing map).
  auto input_type =
      dyn_cast<RankedTensorType>(generic_op.getInputs().front().getType());
  if (!input_type) return false;

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  unsigned last = input_map.getNumResults() - 1;
  auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(last));
  if (!dim_expr) return false;

  return iter_types[dim_expr.getPosition()] == utils::IteratorType::reduction;
}

// ---------------------------------------------------------------------------
// Partition the reduction dims of `generic_op` into:
//   inner dim  — exactly the rightmost reduction dim (in input order).
//   outer dims — all remaining (leftward) reduction dims.
// ---------------------------------------------------------------------------
static FailureOr<CandidateInfo> partitionReductionDims(
    linalg::GenericOp generic_op, int64_t vector_length) {
  CandidateInfo info;
  info.generic_op = generic_op;

  SmallVector<ReductionDimInfo> all_dims;
  if (failed(collectReductionDims(generic_op, all_dims))) return failure();

  // The rightmost reduction dim (last in all_dims) is the inner dim.
  assert(!all_dims.empty());
  ReductionDimInfo& inner_dim = all_dims.back();
  assert(inner_dim.size == vector_length &&
         "inner (rightmost) reduction dim size must equal vector length");

  // all_dims[0 .. N-2] are outer dims; all_dims[N-1] is the inner dim.
  for (int i = 0; i < static_cast<int>(all_dims.size()) - 1; ++i)
    info.outer_dims.push_back(all_dims[i]);
  info.inner_dim.push_back(inner_dim);

  LDBG(1) << PASS_NAME ": generic at " << generic_op.getLoc()
          << " vector_length=" << vector_length
          << " inner dim (loop_dim:size):";
  for (auto& d : info.inner_dim)
    LDBG(1) << "  [" << d.loop_dim << "]=" << d.size;
  LDBG(1) << " outer dims (loop_dim:size):";
  for (auto& d : info.outer_dims)
    LDBG(1) << "  [" << d.loop_dim << "]=" << d.size;

  return info;
}

// ---------------------------------------------------------------------------
// Clone the body region of `src` into a freshly-created (but not yet emitted)
// GenericOp `dst`.  The cloneInto helper prepends an empty placeholder block
// that must be removed after the copy.
// ---------------------------------------------------------------------------
static void cloneGenericBody(linalg::GenericOp src, linalg::GenericOp dst) {
  IRMapping mapping;
  src.getRegion().cloneInto(&dst.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it.
  Block& placeholder = dst.getRegion().front();
  if (&placeholder != &dst.getRegion().back()) placeholder.erase();
}

// ---------------------------------------------------------------------------
// Split a reduction linalg.generic into two:
//
//   Generic 1 (outer reduction): reduces the outer dims.
//     Input:  original input tensor
//     Output: intermediate tensor — input dims with outer reduction dims
//             removed; the inner dim is kept as parallel.
//     Iterator types: outer dims → reduction, inner dim → parallel,
//                     other input-map dims unchanged.
//
//   Generic 2 (inner reduction): reduces the inner dim.
//     Input:  intermediate tensor (result of Generic 1)
//     Output: same type as the original output
//     Iterator types: inner dim → reduction, all other dims → parallel.
//
// The original generic is replaced by the result of Generic 2 and erased.
// ---------------------------------------------------------------------------
static void splitDim(CandidateInfo& info) {
  linalg::GenericOp generic_op = info.generic_op;
  LDBG(1) << PASS_NAME ": splitDim on generic at " << generic_op.getLoc();

  MLIRContext* ctx = generic_op.getContext();
  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // ── Collect structural info ──────────────────────────────────────────────
  SmallVector<utils::IteratorType> orig_iter_types =
      generic_op.getIteratorTypesArray();
  unsigned N = static_cast<unsigned>(orig_iter_types.size());

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  AffineMap output_map = generic_op.getIndexingMapsArray().back();

  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  auto output_type =
      cast<RankedTensorType>(generic_op.getOutputs().front().getType());
  ArrayRef<int64_t> input_shape = input_type.getShape();
  Type elem_type = output_type.getElementType();

  // Build sets of outer and inner loop dim indices for quick lookup.
  llvm::SmallDenseSet<unsigned> inner_loop_dims;
  llvm::SmallDenseSet<unsigned> outer_loop_dims;
  for (auto& d : info.inner_dim) inner_loop_dims.insert(d.loop_dim);
  for (auto& d : info.outer_dims) outer_loop_dims.insert(d.loop_dim);

  // ── Build G1 ─────────────────────────────────────────────────────────────
  // G1 loops over exactly the dims that appear in the input map, in order
  // (g1_num_dims = input map rank).  Any dims that appear only in the output
  // map are not part of G1's loop space — they belong entirely to G2.
  // orig_to_g1_dim: original loop dim index → G1 loop dim index.
  unsigned g1_num_dims = input_map.getNumResults();
  llvm::SmallDenseMap<unsigned, unsigned> orig_to_g1_dim;
  for (unsigned r = 0; r < g1_num_dims; ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(input_map.getResult(r)).getPosition();
    orig_to_g1_dim[orig_ld] = r;
  }

  // G1 input map: identity over G1's loop dims.
  SmallVector<AffineExpr> g1_in_exprs;
  for (unsigned r = 0; r < g1_num_dims; ++r)
    g1_in_exprs.push_back(getAffineDimExpr(r, ctx));
  AffineMap g1_in_map = AffineMap::get(g1_num_dims, 0, g1_in_exprs, ctx);

  // G1 output map and intermediate shape: walk the input map left-to-right,
  // skipping outer reduction dims (which G1 reduces away).  Each surviving
  // dim contributes one result to G1's output map and one dim to the
  // intermediate tensor.
  SmallVector<int64_t> inter_shape;
  SmallVector<AffineExpr> g1_out_exprs;
  llvm::SmallDenseMap<unsigned, unsigned> orig_ld_to_inter_slot;
  for (unsigned r = 0; r < g1_num_dims; ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(input_map.getResult(r)).getPosition();
    if (outer_loop_dims.count(orig_ld)) continue;
    orig_ld_to_inter_slot[orig_ld] = static_cast<unsigned>(inter_shape.size());
    inter_shape.push_back(input_shape[r]);
    g1_out_exprs.push_back(getAffineDimExpr(orig_to_g1_dim[orig_ld], ctx));
  }
  AffineMap g1_out_map = AffineMap::get(g1_num_dims, 0, g1_out_exprs, ctx);

  // G1 iterator types: outer dims → reduction, inner dim → parallel,
  // other input-map dims → same as original.
  SmallVector<utils::IteratorType> g1_iter_types(g1_num_dims);
  for (unsigned r = 0; r < g1_num_dims; ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(input_map.getResult(r)).getPosition();
    if (outer_loop_dims.count(orig_ld))
      g1_iter_types[r] = utils::IteratorType::reduction;
    else if (inner_loop_dims.count(orig_ld))
      g1_iter_types[r] = utils::IteratorType::parallel;
    else
      g1_iter_types[r] = orig_iter_types[orig_ld];
  }

  // ── Build G2 ─────────────────────────────────────────────────────────────
  // G2 loops over all original dims except the outer reduction dims, in
  // original order.  This includes the inner dim (now a reduction) and any
  // output-only dims.
  // orig_to_g2_dim: original loop dim index → G2 loop dim index (0..M-1).
  llvm::SmallDenseMap<unsigned, unsigned> orig_to_g2_dim;
  unsigned g2_idx = 0;
  for (unsigned i = 0; i < N; ++i) {
    if (!outer_loop_dims.count(i)) orig_to_g2_dim[i] = g2_idx++;
  }
  unsigned M = g2_idx;

  // G2 input map: maps G2's loop dims to the intermediate tensor slots.
  unsigned inter_rank = static_cast<unsigned>(inter_shape.size());
  SmallVector<AffineExpr> g2_in_exprs(inter_rank);
  for (auto& [orig_ld, slot] : orig_ld_to_inter_slot)
    g2_in_exprs[slot] = getAffineDimExpr(orig_to_g2_dim[orig_ld], ctx);
  AffineMap g2_in_map = AffineMap::get(M, 0, g2_in_exprs, ctx);

  // G2 output map: remap the original output map into G2's loop dim space.
  SmallVector<AffineExpr> g2_out_exprs;
  for (unsigned r = 0; r < output_map.getNumResults(); ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(output_map.getResult(r)).getPosition();
    assert(orig_to_g2_dim.count(orig_ld) &&
           "output dim maps to an outer loop dim — unexpected");
    g2_out_exprs.push_back(getAffineDimExpr(orig_to_g2_dim[orig_ld], ctx));
  }
  AffineMap g2_out_map = AffineMap::get(M, 0, g2_out_exprs, ctx);

  // G2 iterator types: inner dim → reduction, all other non-outer dims →
  // parallel.
  SmallVector<utils::IteratorType> g2_iter_types(M);
  for (unsigned i = 0; i < N; ++i) {
    if (outer_loop_dims.count(i)) continue;
    g2_iter_types[orig_to_g2_dim[i]] = inner_loop_dims.count(i)
                                           ? utils::IteratorType::reduction
                                           : utils::IteratorType::parallel;
  }

  // ── Emit intermediate tensor initialiser ─────────────────────────────────
  auto inter_tensor_type = RankedTensorType::get(inter_shape, elem_type);
  auto inter_empty =
      tensor::EmptyOp::create(builder, loc, inter_shape, elem_type);

  // ── Emit Generic 1 ───────────────────────────────────────────────────────
  auto g1 = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{inter_tensor_type},
      /*inputs=*/generic_op.getInputs(),
      /*outputs=*/ValueRange{inter_empty.getResult()},
      ArrayRef<AffineMap>{g1_in_map, g1_out_map}, g1_iter_types);
  cloneGenericBody(generic_op, g1);

  // ── Emit output tensor initialiser for Generic 2 ─────────────────────────
  auto out_empty =
      tensor::EmptyOp::create(builder, loc, output_type.getShape(), elem_type);

  // ── Emit Generic 2 ───────────────────────────────────────────────────────
  auto g2 = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{output_type},
      /*inputs=*/ValueRange{g1.getResult(0)},
      /*outputs=*/ValueRange{out_empty.getResult()},
      ArrayRef<AffineMap>{g2_in_map, g2_out_map}, g2_iter_types);
  cloneGenericBody(generic_op, g2);

  // ── Replace and erase original generic and its now-dead output init ──────
  generic_op.getResult(0).replaceAllUsesWith(g2.getResult(0));
  Value orig_out_init = generic_op.getOutputs().front();
  generic_op.erase();
  if (orig_out_init.use_empty())
    if (auto* def = orig_out_init.getDefiningOp()) def->erase();
}

struct SplitReductionInnerOuterDimPass
    : public mlir::ktdf::impl::SplitReductionInnerOuterDimPassBase<
          SplitReductionInnerOuterDimPass> {
  using SplitReductionInnerOuterDimPassBase<
      SplitReductionInnerOuterDimPass>::SplitReductionInnerOuterDimPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();

    // Obtain the device manager and derive vector_length from the SIMD feature.
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError(PASS_NAME
                        ": unable to import the device specification");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<scheduler::arch_view::ResourceKinds>(
            *device);

    // Collect eligible linalg.generic ops: at least two reduction iterator
    // types, with one reduction dim mapping to the rightmost non-1 input dim.
    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](linalg::GenericOp generic_op) {
      if (isEligible(generic_op)) candidates.push_back(generic_op);
      return WalkResult::advance();
    });

    if (candidates.empty()) {
      LDBG(1) << PASS_NAME ": no eligible linalg.generic found — skipping";
      return;
    }

    auto simd_feature =
        resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(
            resource_kinds.getComputeKind());

    for (linalg::GenericOp generic_op : candidates) {
      // Derive the element type from the first output (the accumulator).
      auto output_type =
          dyn_cast<ShapedType>(generic_op.getOutputs().front().getType());
      assert(output_type);
      Type elem_type = output_type.getElementType();

      const int64_t vector_length =
          std::max(simd_feature.getLanes(elem_type), int64_t(1));

      FailureOr<CandidateInfo> info =
          partitionReductionDims(generic_op, vector_length);
      if (failed(info)) continue;
      splitDim(*info);
    }
  }
};

}  // namespace

auto mlir::ktdf::createSplitReductionInnerOuterDimPass()
    -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<SplitReductionInnerOuterDimPass>();
}
