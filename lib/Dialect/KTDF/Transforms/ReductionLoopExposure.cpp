//===-- ReductionLoopExposure.cpp -------------------------------*- c++ -*-===//
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
// ReductionLoopExposure: expose the reduction dimension(s) of a linalg.generic
// inside a ktdf.pipeline as explicit scf.for loops with a loop-carry
// accumulator tensor.
//
// Algorithm:
//   1. Walk the module for any ktdf.stage containing a linalg.generic with
//      a reduction iterator — this is the compute stage.
//   2. Collect reduction_dims[] and their per-dimension sizes dim_sizes[].
//      The total shrink factor R = product(dim_sizes[]).
//   3. Get the parent ktdf.pipeline of the compute stage.
//   4. Find the load stage via StageFactory::findLoadStage (upstream of
//      compute via depends_in/depends_out token chain).  Identify the
//      FIFO-dest data_transfer that feeds linalg.generic ins() — its
//      destination is fifo_in.
//   5. Shrink only fifo_in in ktdf.private (divide its element count by R).
//      Patch the corresponding fifo.allocate inside the private body to match.
//   6. Patch every data_transfer inside the pipeline whose source or
//      destination is a shrunken FIFO: divide its static sizes by R.
//   7. For every stage in the pipeline, wrap its entire body in one nested
//      scf.for loop per reduction dimension:
//        scf.for %r0 = 0 to dim_sizes[0] step 1
//          scf.for %r1 = 0 to dim_sizes[1] step 1
//            ...
//   8. For the compute stage specifically:
//      - tensor.empty is emitted before the outermost loop (accumulator init).
//      - Each loop level carries the accumulator as iter_args; the value
//        threads down to the innermost loop where linalg.generic runs.
//      - write_to_fifo is wrapped in scf.if (all ivs == last) in the
//        innermost loop body.
//      - The outermost loop is tagged {loop_type = reduction_loop}.
//   9. Find the conditional-store stage via StageFactory::findStoreStage
//      (downstream of compute via depends_out/depends_in token chain).
//      Wrap its data_transfer in scf.if (all ivs == last) inside the
//      innermost already-created scf.for.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "reduction-loop-exposure"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONLOOPEXPOSUREPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Reduction Loop Exposure pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Carry the result of per-dimension reduction analysis.
// ---------------------------------------------------------------------------
struct ReductionInfo {
  SmallVector<int64_t> reduction_dims;  // indices into iterator-type list
  SmallVector<int64_t> dim_sizes;       // size of each reduction dimension
  int64_t reduction_size = 1;           // product of dim_sizes
};

// ---------------------------------------------------------------------------
// Fill `info` with the reduction dimensions and sizes derived from
// `generic_op`, excluding the inner reduction dimension so
// that it is left for later handling by the ktdf.opaque wrapping pass.
// Returns true on success.  Returns false (and emits a diagnostic or debug
// message) when:
//   - there are no outer-dim reduction dimensions to expose, or
//   - any qualifying reduction dimension has a dynamic size.
// ---------------------------------------------------------------------------
static bool collectReductionInfo(linalg::GenericOp generic_op,
                                 ReductionInfo& info) {
  std::optional<unsigned> inner_dim = ktdf::findInnerDimLoopDim(generic_op);

  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  int64_t input_rank = input_type.getRank();

  // Collect reduction dims to expose as loops, excluding only the inner
  // dim (loop dim at input axis input_rank-1).  Output-only reduction dims
  // (i >= input_rank) are included.  Size-1 dims are skipped.
  auto output_type =
      cast<RankedTensorType>(generic_op.getOutputs().front().getType());
  AffineMap output_map = generic_op.getIndexingMapsArray().back();

  auto iter_types = generic_op.getIteratorTypesArray();
  for (int64_t i = 0; i < static_cast<int64_t>(iter_types.size()); ++i) {
    if (iter_types[i] != utils::IteratorType::reduction) continue;
    if (inner_dim && static_cast<unsigned>(i) == *inner_dim) {
      LDBG(1) << PASS_NAME ": skipping inner-dim reduction dim " << i;
      continue;
    }

    // Determine the size of this loop dim from the input tensor (if it
    // appears there) or from the output tensor (output-only dim).
    int64_t dim_sz = ShapedType::kDynamic;
    if (i < input_rank) {
      dim_sz = input_type.getDimSize(i);
    } else {
      for (unsigned r = 0; r < output_map.getNumResults(); ++r) {
        auto expr = dyn_cast<AffineDimExpr>(output_map.getResult(r));
        if (expr && expr.getPosition() == static_cast<unsigned>(i)) {
          dim_sz = output_type.getDimSize(r);
          break;
        }
      }
    }

    if (dim_sz == ShapedType::kDynamic) {
      LDBG(1) << PASS_NAME ": dynamic reduction size not supported — skipping";
      return false;
    }
    if (dim_sz == 1) continue;

    info.reduction_dims.push_back(i);
    info.dim_sizes.push_back(dim_sz);
    info.reduction_size *= dim_sz;
  }

  if (info.reduction_dims.empty()) {
    LDBG(1) << PASS_NAME ": no outer-dim reduction dims to expose — skipping";
    return false;
  }

  LDBG(1) << PASS_NAME ": reduction_dims=[";
  for (size_t i = 0; i < info.reduction_dims.size(); ++i) {
    if (i > 0) LDBG(1) << ", ";
    LDBG(1) << info.reduction_dims[i] << " (size=" << info.dim_sizes[i] << ")";
  }
  LDBG(1) << "] total_reduction_size=" << info.reduction_size;

  return true;
}

// ---------------------------------------------------------------------------
// Collect all linalg.generic ops directly inside `stage` that have outer-dim
// reduction dims to expose (i.e. collectReductionInfo succeeds for them).
//
// Generics whose first input is produced by another linalg.generic are
// chained inner-dim ops (e.g. G2 after SplitReductionInnerOuterDim) and are
// skipped (their reduction loops should not be exposed).
// ---------------------------------------------------------------------------
static void collectReductionGenericOps(
    ktdf::StageOp stage, SmallVectorImpl<linalg::GenericOp>& result) {
  // PreOrder so we see a nested ktdf.stage before its children, allowing
  // WalkResult::skip() to prevent descent into a different pipeline.
  stage.getBody()->walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<ktdf::StageOp>(op)) return WalkResult::skip();
    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      // Skip chained generics whose input comes from another generic — those
      // are inner-dim ops left for a later pass.
      if (generic.getInputs().front().getDefiningOp<linalg::GenericOp>())
        return WalkResult::advance();
      ReductionInfo info;
      if (collectReductionInfo(generic, info)) result.push_back(generic);
    }
    return WalkResult::advance();
  });
}

// ---------------------------------------------------------------------------
// Walk all data_transfer ops inside `pipeline` and divide their static source
// or destination sizes by `reduction_size` wherever the FIFO endpoint was
// shrunken (i.e. its result index appears in `shrunken_fifo_map`).
// ---------------------------------------------------------------------------
static void patchDataTransferSizes(
    ktdf::PipelineOp pipeline, ktdf::PrivateOp new_priv, int64_t reduction_size,
    const llvm::DenseMap<unsigned, ktdf::FifoSlotType>& shrunken_fifo_map,
    MLIRContext* ctx) {
  pipeline->walk([&](ktdf::DataTransferOp xfer) {
    if (xfer.isDestFifo()) {
      auto fifo_res = dyn_cast<OpResult>(xfer.getDestination());
      if (fifo_res && fifo_res.getOwner() == new_priv &&
          shrunken_fifo_map.count(fifo_res.getResultNumber())) {
        if (auto sizes = xfer.getStaticDestSizes()) {
          SmallVector<int64_t> new_sizes;
          for (int64_t sz : *sizes) new_sizes.push_back(sz / reduction_size);
          xfer.setStaticDestSizesAttr(DenseI64ArrayAttr::get(ctx, new_sizes));
        }
      }
    }
    if (xfer.isSourceFifo()) {
      auto fifo_res = dyn_cast<OpResult>(xfer.getSource());
      if (fifo_res && fifo_res.getOwner() == new_priv &&
          shrunken_fifo_map.count(fifo_res.getResultNumber())) {
        if (auto sizes = xfer.getStaticSourceSizes()) {
          SmallVector<int64_t> new_sizes;
          for (int64_t sz : *sizes) new_sizes.push_back(sz / reduction_size);
          xfer.setStaticSourceSizesAttr(DenseI64ArrayAttr::get(ctx, new_sizes));
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Build a nested scf.for structure with one loop per entry in `dim_sizes`.
// Returns the innermost loop body's insertion point via `innermost_loop` and
// a vector of all induction variables (outermost first).
//
// All loops are created inside `parent_block` at the current insertion point
// of `rewriter`.  Every loop in the nest is tagged {loop_type =
// reduction_loop}.
// ---------------------------------------------------------------------------
struct NestedForResult {
  scf::ForOp outermost_loop;
  scf::ForOp innermost_loop;
  SmallVector<Value> ivs;  // outermost first
};

NestedForResult buildNestedForLoops(OpBuilder& builder, Location loc,
                                    MLIRContext* context,
                                    ArrayRef<int64_t> dim_sizes, Value start,
                                    Value step, ValueRange iter_args = {}) {
  assert(!dim_sizes.empty() && "need at least one dimension");

  NestedForResult result;
  result.ivs.reserve(dim_sizes.size());

  // Build upper-bound constants for each dimension.
  SmallVector<Value> upper_bounds;
  for (int64_t dim_sz : dim_sizes)
    upper_bounds.push_back(
        arith::ConstantIndexOp::create(builder, loc, dim_sz));

  // We build the loops from outermost to innermost, using the body-builder
  // callback so that each inner loop is created inside the outer loop.
  // Because scf.ForOp::create with a builder callback invokes the callback
  // immediately, we can recurse through the dimensions in one pass.
  //
  // To keep the logic flat (avoid actual C++ recursion), we use a lambda that
  // captures a mutable state and is called iteratively via the builder chain.
  // The outermost loop carries `iter_args`; each nesting level passes the
  // iter_arg through as-is until the innermost level where the user code runs.

  scf::ForOp outermost;
  scf::ForOp innermost;
  SmallVector<Value> collected_ivs;

  // Recursive lambda (std::function for self-reference).
  std::function<Value(OpBuilder&, Location, size_t, Value)> build_level =
      [&](OpBuilder& loop_builder, Location loop_loc, size_t depth,
          Value carry) -> Value {
    bool is_outermost = (depth == 0);
    bool is_innermost = (depth == dim_sizes.size() - 1);

    // Use a SmallVector to avoid a dangling ValueRange pointing at a temporary.
    SmallVector<Value> level_args_vec =
        is_outermost
            ? SmallVector<Value>(iter_args.begin(), iter_args.end())
            : (carry ? SmallVector<Value>{carry} : SmallVector<Value>{});
    ValueRange level_args(level_args_vec);

    auto loop = scf::ForOp::create(
        loop_builder, loop_loc, start, upper_bounds[depth], step, level_args,
        [&](OpBuilder& body_builder, Location body_loc, Value iv,
            ValueRange loop_iter_args) {
          collected_ivs.push_back(iv);
          (void)is_outermost;  // captured for level_args logic above

          Value inner_carry =
              loop_iter_args.empty() ? Value{} : loop_iter_args.front();

          if (is_innermost) {
            // Yield the carry unchanged; the caller will replace the yield.
            if (inner_carry)
              scf::YieldOp::create(body_builder, body_loc,
                                   ValueRange{inner_carry});
            else
              scf::YieldOp::create(body_builder, body_loc, ValueRange{});
          } else {
            Value returned =
                build_level(body_builder, body_loc, depth + 1, inner_carry);
            if (returned)
              scf::YieldOp::create(body_builder, body_loc,
                                   ValueRange{returned});
            else
              scf::YieldOp::create(body_builder, body_loc, ValueRange{});
          }
        });

    loop->setAttr("loop_type", ktdf::LoopTypeAttr::get(
                                   context, ktdf::LoopType::ReductionLoop));
    if (depth == 0) outermost = loop;
    if (is_innermost) innermost = loop;
    return loop.getResults().empty() ? Value{} : loop.getResult(0);
  };

  // Build a temporary placeholder carry if iter_args has an element so the
  // lambda signature works.  The returned Value from the outermost level is
  // the final result of the loop chain (unused outside — the caller sets up
  // yield ops separately via the innermost loop body).
  Value placeholder_carry = iter_args.empty() ? Value{} : iter_args.front();
  OpBuilder root_builder = builder;
  build_level(root_builder, loc, 0, placeholder_carry);

  result.outermost_loop = outermost;
  result.innermost_loop = innermost;
  result.ivs = collected_ivs;

  return result;
}

// ---------------------------------------------------------------------------
// Find the data_transfer in `load_stage` whose FIFO-slot destination feeds the
// linalg.generic inputs in `compute_stage`.  When there is only one FIFO-dest
// transfer the choice is trivial.  When there are several (e.g. a main-input
// transfer and a conditional accumulator transfer) we identify the right one by
// tracing generic_op.getInputs().front() back to its ReadFromFifoOp and
// matching on the FIFO slot value.
//
// Returns the destination Value of the chosen transfer, or a null Value and
// emits an error on `inner_pipeline` if selection fails.
// ---------------------------------------------------------------------------
Value findFifoInTransfer(ktdf::PipelineOp inner_pipeline,
                         ktdf::StageOp load_stage,
                         linalg::GenericOp generic_op) {
  SmallVector<ktdf::DataTransferOp> fifo_dest_transfers;
  load_stage.getBody()->walk([&](ktdf::DataTransferOp xfer) {
    if (xfer.isDestFifo()) fifo_dest_transfers.push_back(xfer);
  });

  if (fifo_dest_transfers.empty()) {
    inner_pipeline.emitError(PASS_NAME
                             ": no FIFO-dest data_transfer in load stage");
    return {};
  }

  if (fifo_dest_transfers.size() == 1) {
    return fifo_dest_transfers.front().getDestination();
  }

  // Multiple FIFO-dest transfers: identify the one whose destination matches
  // the FIFO slot consumed by linalg.generic ins().
  Value generic_input = generic_op.getInputs().front();

  // The input to linalg.generic must be directly produced by a read_from_fifo.
  auto read_op = generic_input.getDefiningOp<ktdf::ReadFromFifoOp>();
  if (!read_op) {
    inner_pipeline.emitError(
        PASS_NAME
        ": multiple FIFO-dest transfers in load stage and cannot "
        "identify which feeds linalg.generic ins()");
    return {};
  }
  Value input_fifo = read_op.getFifoSlot();

  for (auto xfer : fifo_dest_transfers) {
    if (xfer.getDestination() == input_fifo) {
      return xfer.getDestination();
    }
  }

  inner_pipeline.emitError(
      PASS_NAME ": none of the FIFO-dest transfers feeds linalg.generic ins()");
  return {};
}

// ---------------------------------------------------------------------------
// Build an i1 value that is true only when every iv[i] == last[i].
// ---------------------------------------------------------------------------
Value buildAllLast(OpBuilder& builder, Location loc, ArrayRef<Value> ivs,
                   ArrayRef<Value> last_vals) {
  assert(ivs.size() == last_vals.size());
  Value cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                     ivs.front(), last_vals.front());
  for (size_t i = 1; i < ivs.size(); ++i) {
    Value cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                      ivs[i], last_vals[i]);
    cond = arith::AndIOp::create(builder, loc, cond, cmp);
  }
  return cond;
}

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------
struct ReductionLoopExposurePass
    : public ktdf::impl::ReductionLoopExposurePassBase<
          ReductionLoopExposurePass> {
  using ReductionLoopExposurePassBase<
      ReductionLoopExposurePass>::ReductionLoopExposurePassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();
    if (failed(transformModule(module))) signalPassFailure();
  }

 private:
  // -------------------------------------------------------------------------
  // Collect every (stage, generic) pair where the generic has outer-dim
  // reduction dims to expose, then rewrite each.
  //
  // Post-order walk ensures inner stages are processed before outer ones.
  // This matters because rewriting an outer stage splices its body into new
  // scf.for loops (cloning contents), so any inner generic must already be
  // in its final form before the outer rewrite runs.
  // -------------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    // Collect (stage, generic) pairs in post-order so inner pipelines come
    // before outer ones.  Rewriting happens after the walk completes to avoid
    // iterator invalidation from mutations made during traversal.
    SmallVector<std::pair<ktdf::StageOp, linalg::GenericOp>> work_items;
    module.walk<WalkOrder::PostOrder>([&](ktdf::StageOp stage) {
      SmallVector<linalg::GenericOp> generics;
      collectReductionGenericOps(stage, generics);
      for (auto generic : generics) work_items.emplace_back(stage, generic);
      return WalkResult::advance();
    });

    // Process inner pipelines first (guaranteed by post-order collection).
    // Each generic is valid at the time its rewrite runs because inner rewrites
    // only mutate their own pipeline's stages, not the enclosing ones.
    for (auto& [stage, generic] : work_items)
      if (failed(rewritePipeline(stage, generic))) return failure();

    return success();
  }

  // -------------------------------------------------------------------------
  // Rewrite the ktdf.pipeline that contains `compute_stage` for the given
  // `generic_op` (which has outer-dim reduction dims to expose).
  // -------------------------------------------------------------------------
  LogicalResult rewritePipeline(ktdf::StageOp compute_stage,
                                linalg::GenericOp generic_op) {
    // --- Collect per-dimension reduction sizes ---
    // Each reduction dimension gets its own scf.for loop.  The total shrink
    // factor R = product(dim_sizes[]) is used only for FIFO resizing.
    ReductionInfo reduction_info;
    if (!collectReductionInfo(generic_op, reduction_info)) return success();
    SmallVector<int64_t>& reduction_dims = reduction_info.reduction_dims;
    SmallVector<int64_t>& dim_sizes = reduction_info.dim_sizes;
    int64_t reduction_size = reduction_info.reduction_size;

    // --- Get the parent pipeline ---
    auto inner_pipeline = compute_stage->getParentOfType<ktdf::PipelineOp>();
    if (!inner_pipeline) {
      compute_stage.emitError(PASS_NAME
                              ": compute stage has no parent pipeline");
      return failure();
    }

    MLIRContext* ctx = inner_pipeline.getContext();
    IRRewriter rewriter(ctx);
    Location loc = inner_pipeline.getLoc();

    // -----------------------------------------------------------------------
    // 1. Find the load stage: the sibling whose depends_out token appears in
    //    the compute stage's depends_in.  Its single FIFO-dest data_transfer
    //    identifies fifo_in (a ktdf.private result) by result index.
    // -----------------------------------------------------------------------
    ktdf::PrivateOp priv_op = inner_pipeline.getPrivateOp();
    if (!priv_op) {
      inner_pipeline.emitError(PASS_NAME ": no ktdf.private in pipeline");
      return failure();
    }

    ktdf::StageOp load_stage =
        ktdf::StageFactory::findLoadStage(inner_pipeline, compute_stage);
    if (!load_stage) {
      inner_pipeline.emitError(
          PASS_NAME ": cannot find load stage upstream of compute stage");
      return failure();
    }

    // From the load stage, find the data_transfer whose FIFO destination feeds
    // the linalg.generic inputs in the compute stage.
    Value orig_fifo_in =
        findFifoInTransfer(inner_pipeline, load_stage, generic_op);
    if (!orig_fifo_in) return failure();

    // Validate that fifo_in is a ktdf.private result.
    auto orig_fifo_in_result = dyn_cast<OpResult>(orig_fifo_in);
    if (!orig_fifo_in_result || orig_fifo_in_result.getOwner() != priv_op) {
      inner_pipeline.emitError(PASS_NAME
                               ": fifo_in is not a ktdf.private result");
      return failure();
    }
    unsigned fifo_in_idx = orig_fifo_in_result.getResultNumber();

    // Build a map: only shrink the input FIFO (fifo_in_idx) — the one whose
    // destination feeds linalg.generic ins().  Accumulator and other FIFOs
    // must keep their original sizes; shrinking them by reduction_size would
    // corrupt the accumulator slot that carries the running result.
    llvm::DenseMap<unsigned, ktdf::FifoSlotType> shrunken_fifo_map;
    SmallVector<Type> new_result_types(priv_op.getResultTypes());
    {
      auto fifo_type = dyn_cast<ktdf::FifoSlotType>(
          priv_op.getResult(fifo_in_idx).getType());
      if (!fifo_type) {
        inner_pipeline.emitError(PASS_NAME ": fifo_in is not a FifoSlotType");
        return failure();
      }
      int64_t num_elems = fifo_type.getNumElements();
      if (num_elems % reduction_size != 0) {
        inner_pipeline.emitError(PASS_NAME
                                 ": input FIFO element count not divisible by "
                                 "reduction_size");
        return failure();
      }
      auto shrunken = ktdf::FifoSlotType::get(
          ctx, fifo_type.getSrc(), fifo_type.getDest(),
          num_elems / reduction_size, fifo_type.getElementType());
      shrunken_fifo_map[fifo_in_idx] = shrunken;
      new_result_types[fifo_in_idx] = shrunken;
    }

    // Replace priv_op with a new one carrying all shrunken FIFO types.
    rewriter.setInsertionPoint(priv_op);
    auto new_priv = ktdf::PrivateOp::create(rewriter, loc, new_result_types);
    rewriter.mergeBlocks(&priv_op.getRegion().front(),
                         &new_priv.getRegion().front(), {});

    // Patch every fifo.allocate inside the new private body.
    new_priv.getRegion().front().walk([&](ktdf::FifoAllocateOp alloc) {
      for (auto& [idx, shrunken] : shrunken_fifo_map) {
        auto orig_type =
            dyn_cast<ktdf::FifoSlotType>(priv_op.getResult(idx).getType());
        if (orig_type && alloc.getResult(0).getType() == orig_type)
          alloc.getResult(0).setType(shrunken);
      }
    });

    for (auto [old_res, new_res] :
         llvm::zip(priv_op.getResults(), new_priv.getResults()))
      old_res.replaceAllUsesWith(new_res);
    rewriter.eraseOp(priv_op);

    Value fifo_in = new_priv.getResult(fifo_in_idx);

    // fifo_in_partial: the other FIFO-dest transfer in the load stage — the
    // one that conditionally sends the previous chunk's partial result to the
    // compute stage for accumulator seeding.  It is any FIFO-dest transfer
    // whose destination is not fifo_in.
    Value fifo_in_partial;
    load_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
      if (dt.isDestFifo() && dt.getDestination() != fifo_in) {
        fifo_in_partial = dt.getDestination();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // is_first_chunk: the i1 condition that guards the scf.if in the load
    // stage (true on the first chunk, which skips the partial transfer).
    // It is the condition of the first scf.if in the load stage body.
    Value is_first_chunk;
    load_stage.getBody()->walk([&](scf::IfOp if_op) {
      is_first_chunk = if_op.getCondition();
      return WalkResult::interrupt();
    });

    if (fifo_in_partial && !is_first_chunk) {
      inner_pipeline.emitError(
          PASS_NAME
          ": found partial FIFO but no scf.if condition in load stage");
      return failure();
    }

    // fifo_out: the fifo_slot operand of the write_to_fifo whose tensor input
    // is directly produced by generic_op (i.e. generic_op.getResult(0) is the
    // first operand of the write_to_fifo).  When the generic feeds another op
    // first (e.g. a second-stage inner-dim reduction after SplitReduction),
    // this will be null and we take the non-FIFO path in rewriteComputeStage.
    Value fifo_out;
    for (Operation* user : generic_op.getResult(0).getUsers()) {
      if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user)) {
        fifo_out = write_op.getFifoSlot();
        break;
      }
    }

    // Output accumulator tensor type (from the generic's output).
    auto output_tensor_type =
        cast<RankedTensorType>(generic_op.getOutputs().front().getType());

    // Per-iteration input slice tensor: same shape as the generic input but
    // size-1 in every reduction dimension.
    // Example: input [1, 256, 64] with reduction dims [1] → slice [1, 1, 64].
    // For non-contiguous reduction dims (e.g. [0, 2] of [d0, d1, d2]) both
    // axes independently become 1, e.g. [1, d1, 1].
    auto slice_tensor_type =
        cast<RankedTensorType>(generic_op.getInputs().front().getType());
    SmallVector<int64_t> slice_shape(slice_tensor_type.getShape().begin(),
                                     slice_tensor_type.getShape().end());
    for (int64_t reduction_dim : reduction_dims)
      slice_shape[static_cast<size_t>(reduction_dim)] = 1;
    auto one_row_tensor_type =
        RankedTensorType::get(slice_shape, slice_tensor_type.getElementType());

    // -----------------------------------------------------------------------
    // 1b. Patch data_transfer static sizes for any transfer that uses a
    //     shrunken FIFO.
    // -----------------------------------------------------------------------
    patchDataTransferSizes(inner_pipeline, new_priv, reduction_size,
                           shrunken_fifo_map, ctx);

    // -----------------------------------------------------------------------
    // 2. Build shared constants before the inner pipeline.
    // -----------------------------------------------------------------------
    rewriter.setInsertionPoint(inner_pipeline);
    Value start = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Per-dim "last index" constants (dim_size - 1).
    SmallVector<Value> last_vals;
    for (int64_t sz : dim_sizes)
      last_vals.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, sz - 1));

    // -----------------------------------------------------------------------
    // 3. Find the "conditional store" stage.
    // -----------------------------------------------------------------------
    ktdf::StageOp conditional_store_stage =
        ktdf::StageFactory::findStoreStage(inner_pipeline, compute_stage);

    // -----------------------------------------------------------------------
    // 4. Rewrite every stage.
    // -----------------------------------------------------------------------
    for (auto stage : inner_pipeline.getStages()) {
      if (stage == compute_stage) {
        rewriteComputeStage(rewriter, loc, ctx, stage, generic_op,
                            one_row_tensor_type, output_tensor_type, fifo_in,
                            fifo_out, fifo_in_partial, is_first_chunk,
                            dim_sizes, start, step, last_vals);
      } else if (stage == conditional_store_stage) {
        rewriteConditionalStoreStage(rewriter, loc, ctx, stage, dim_sizes,
                                     start, step, last_vals);
      } else if (stage == load_stage) {
        rewriteLoadStage(rewriter, loc, ctx, stage, dim_sizes, start, step,
                         reduction_dims, slice_tensor_type.getRank(), fifo_in);
      } else {
        rewritePlainStage(rewriter, loc, ctx, stage, dim_sizes, start, step);
      }
    }

    return success();
  }

  // -------------------------------------------------------------------------
  // Wrap the entire body of a plain (non-compute, non-conditional) stage in
  // N nested scf.for loops, one per reduction dimension.  All existing ops
  // move into the innermost loop body.
  // Returns the NestedForResult so callers have access to their IVs if needed.
  // -------------------------------------------------------------------------
  NestedForResult rewritePlainStage(IRRewriter& rewriter, Location loc,
                                    MLIRContext* ctx, ktdf::StageOp stage,
                                    ArrayRef<int64_t> dim_sizes, Value start,
                                    Value step) {
    Block* body = stage.getBody();

    // Capture an iterator to the first original op *before* the loop shell is
    // prepended.  iplist iterators survive insertions elsewhere in the list, so
    // this remains valid after buildNestedForLoops runs.
    auto first_original = body->begin();

    rewriter.setInsertionPointToStart(body);
    auto nested =
        buildNestedForLoops(rewriter, loc, ctx, dim_sizes, start, step);

    // Splice [first_original, body->end()) into the innermost loop body before
    // its yield.  body and inner_body are different lists so body->end() is a
    // safe past-the-end sentinel for the source range.
    Block* inner_body = nested.innermost_loop.getBody();
    Operation* inner_term = inner_body->getTerminator();
    inner_body->getOperations().splice(inner_term->getIterator(),
                                       body->getOperations(), first_original,
                                       body->end());

    return nested;
  }

  // -------------------------------------------------------------------------
  // Patch a single main-input data_transfer that is already inside the
  // innermost reduction loop: shrink its source sizes at every reduction
  // dimension to 1 and add the corresponding loop IV to its source map.
  //
  // Two cases for the map result at reduction dimension d:
  //   a) Zero constant (pre-chunking) — replace with the loop IV.
  //   b) Non-zero expression (post-chunking chunk-base) — add the loop IV.
  // -------------------------------------------------------------------------
  void patchInputTransfer(ktdf::DataTransferOp dt,
                          const NestedForResult& nested,
                          ArrayRef<int64_t> reduction_dims, int64_t tensor_rank,
                          MLIRContext* ctx) {
    auto sizes_attr = dt.getStaticSourceSizes();
    if (!sizes_attr) return;
    int64_t memref_rank = static_cast<int64_t>(sizes_attr->size());
    // Stage coarsening may have introduced extra leading dimensions into the
    // source memref (to express additional granularity), so the reduction
    // dimensions sit at an offset from the right relative to the linalg
    // tensor rank.
    int64_t rank_offset = memref_rank - tensor_rank;

    // --- 1. Set size to 1 at each memref reduction dimension ---
    SmallVector<int64_t> new_sizes(sizes_attr->begin(), sizes_attr->end());
    for (size_t i = 0; i < reduction_dims.size(); ++i) {
      unsigned dim = static_cast<unsigned>(reduction_dims[i] + rank_offset);
      if (dim < new_sizes.size()) {
        new_sizes[dim] = 1;
      }
    }
    dt.setStaticSourceSizesAttr(DenseI64ArrayAttr::get(ctx, new_sizes));

    // --- 2. Rewrite source_map + source_indices ---
    // The source indices are AffineMap operands, not a flat per-dimension
    // list — constant indices live inside the map as affine constants.
    //
    // Two cases for the map result at the reduction dimension d:
    //
    //   a) It is a zero constant — the pre-chunking shape (index = 0).
    //      Append a new dim for the reduction IV and replace the result.
    //      Combined index: nested.ivs[i]
    //
    //   b) It is any non-zero expression — ReductionDimChunking has already
    //      encoded a chunk-base offset here (e.g. d_j * chunk_size where
    //      d_j is the chunk loop IV).  We must keep that base and add the
    //      new reduction IV on top:
    //        new_result[d] = existing + new_dim_expr
    //      so the combined index is: chunk_base + nested.ivs[i].
    //
    // Note: after ReductionDimChunking the expression is AffineBinaryOpExpr
    // (d_j * chunk_size), not a bare AffineDimExpr, so we cannot use
    // dyn_cast<AffineDimExpr> as the discriminant — use isZero() instead.
    std::optional<AffineMap> maybe_map = dt.getSourceMap();
    if (!maybe_map) return;
    AffineMap map = *maybe_map;

    unsigned base_iv_dim = map.getNumDims();
    SmallVector<AffineExpr> new_results(map.getResults().begin(),
                                        map.getResults().end());
    SmallVector<Value> new_indices(dt.getSourceIndices().begin(),
                                   dt.getSourceIndices().end());
    unsigned extra_dims = 0;
    for (size_t i = 0; i < reduction_dims.size(); ++i) {
      unsigned dim = static_cast<unsigned>(reduction_dims[i] + rank_offset);
      if (dim >= new_results.size()) {
        continue;
      }

      AffineExpr existing = new_results[dim];
      unsigned new_dim = base_iv_dim + extra_dims++;
      AffineExpr new_iv_expr = getAffineDimExpr(new_dim, ctx);
      auto const_expr = dyn_cast<AffineConstantExpr>(existing);
      if (const_expr && const_expr.getValue() == 0) {
        // Case (a): no prior chunking — replace outright with the IV.
        new_results[dim] = new_iv_expr;
      } else {
        // Case (b): chunk-base already encoded — add the reduction IV.
        new_results[dim] = existing + new_iv_expr;
      }
      new_indices.push_back(nested.ivs[i]);
    }

    AffineMap new_map = AffineMap::get(base_iv_dim + extra_dims,
                                       map.getNumSymbols(), new_results, ctx);
    dt.setSourceMapAttr(AffineMapAttr::get(new_map));
    dt.getSourceIndicesMutable().assign(new_indices);
  }

  // -------------------------------------------------------------------------
  // Rewrite the load stage:
  //   - The partial accumulator transfer (scf.if that conditionally sends the
  //     previous chunk's result) stays outside the reduction loop — it is a
  //     once-per-chunk operation.
  //   - The main input transfer (dest == fifo_in) is moved inside the new
  //     reduction scf.for loop, with its source size and address patched.
  //
  // Example (reduction_dim=1 in a 1x256x64 tensor, source memref 2x1x256x64):
  //   before: data_transfer from %src[%b, 0, 0, 0] size [1, 1, 256, 64]
  //                          to fifo_in size [16384]
  //           scf.if %is_first {} else {
  //             data_transfer from %partial[...] to fifo_in_partial size [64]
  //           }
  //   after:  scf.if %is_first {} else {
  //             data_transfer from %partial[...] to fifo_in_partial size [64]
  //           }
  //           scf.for %r = 0 to 256 {
  //             data_transfer from %src[%b, 0, %r, 0] size [1, 1, 1, 64]
  //                            to fifo_in size [64]
  //           }
  // -------------------------------------------------------------------------
  void rewriteLoadStage(IRRewriter& rewriter, Location loc, MLIRContext* ctx,
                        ktdf::StageOp stage, ArrayRef<int64_t> dim_sizes,
                        Value start, Value step,
                        ArrayRef<int64_t> reduction_dims, int64_t tensor_rank,
                        Value fifo_in) {
    Block* body = stage.getBody();

    // Find the main input transfer (dest == fifo_in).
    ktdf::DataTransferOp main_transfer;
    body->walk([&](ktdf::DataTransferOp dt) {
      if (dt.isDestFifo() && dt.getDestination() == fifo_in) {
        main_transfer = dt;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // If we cannot identify the main transfer, fall back to wrapping everything
    // (original behaviour) — keeps the pass safe for pipelines without a
    // partial accumulator path.
    if (!main_transfer) {
      auto nested =
          rewritePlainStage(rewriter, loc, ctx, stage, dim_sizes, start, step);
      nested.innermost_loop.getBody()->walk([&](ktdf::DataTransferOp dt) {
        if (dt.isDestFifo()) {
          patchInputTransfer(dt, nested, reduction_dims, tensor_rank, ctx);
        }
      });
      return;
    }

    // Build the reduction loop shell at the end of the stage body (after all
    // existing ops), so the scf.if for the partial stays before the loop.
    rewriter.setInsertionPointToEnd(body);
    auto nested =
        buildNestedForLoops(rewriter, loc, ctx, dim_sizes, start, step);

    // Move the main input transfer into the innermost loop body (before yield).
    Block* inner_body = nested.innermost_loop.getBody();
    Operation* inner_term = inner_body->getTerminator();
    main_transfer->moveBefore(inner_term);

    // Patch sizes and address map of the main transfer now that it is inside
    // the reduction loop.
    patchInputTransfer(main_transfer, nested, reduction_dims, tensor_rank, ctx);
  }

  // -------------------------------------------------------------------------
  // Rewrite the compute stage with N nested scf.for loops (one per
  // reduction dim), each carrying the accumulator tensor as iter_arg.
  //
  // The accumulator seed is determined before the outermost loop:
  //   - On the first chunk (is_first_chunk == true): tensor.empty (zero init).
  //   - On subsequent chunks: read the previous partial result from
  //     fifo_in_partial.
  //
  //   %seed = scf.if %is_first -> tensor<...> {
  //     %e = tensor.empty(); scf.yield %e
  //   } else {
  //     %p = ktdf.read_from_fifo fifo_in_partial; scf.yield %p
  //   }
  //   scf.for %r0 = 0 to D0 iter_args(%a0 = %seed) {loop_type = reduction}
  //     ...
  //       %slice = ktdf.read_from_fifo fifo_in
  //       %updated = linalg.generic(%slice, %aInner)
  //       if (all ivs == last): ktdf.write_to_fifo %updated, fifo_out
  //       scf.yield %updated
  //     ...
  //   scf.yield %r0_result
  // -------------------------------------------------------------------------
  void rewriteComputeStage(IRRewriter& rewriter, Location loc, MLIRContext* ctx,
                           ktdf::StageOp stage, linalg::GenericOp generic_op,
                           RankedTensorType one_row_tensor_type,
                           RankedTensorType output_tensor_type, Value fifo_in,
                           Value fifo_out, Value fifo_in_partial,
                           Value is_first_chunk, ArrayRef<int64_t> dim_sizes,
                           Value start, Value step, ArrayRef<Value> last_vals) {
    Block* body = stage.getBody();

    // Collect existing ops to erase after the rewrite.
    SmallVector<Operation*> to_erase;
    for (auto& op : *body) to_erase.push_back(&op);

    rewriter.setInsertionPointToStart(body);

    // Accumulator seed: when a partial FIFO path exists, on the first chunk
    // zero-init via tensor.empty; on subsequent chunks read the previous
    // partial result from fifo_in_partial.  When there is no partial path
    // (pipeline has no accumulator feedback), always use tensor.empty.
    Value seed;
    if (fifo_in_partial && is_first_chunk) {
      // Build the seed scf.if with an else region.  The regions start empty,
      // so we use OpBuilder::atBlockBegin (not getTerminator()) to populate
      // them before inserting the scf.yield terminator.
      auto seed_if =
          scf::IfOp::create(rewriter, loc, TypeRange{output_tensor_type},
                            is_first_chunk, /*withElseRegion=*/true);
      {
        Block& then_block = seed_if.getThenRegion().front();
        OpBuilder then_b = OpBuilder::atBlockBegin(&then_block);
        auto empty =
            tensor::EmptyOp::create(then_b, loc, output_tensor_type.getShape(),
                                    output_tensor_type.getElementType());
        scf::YieldOp::create(then_b, loc, ValueRange{empty.getResult()});
      }
      {
        Block& else_block = seed_if.getElseRegion().front();
        OpBuilder else_b = OpBuilder::atBlockBegin(&else_block);
        auto partial_read = ktdf::ReadFromFifoOp::create(
            else_b, loc, output_tensor_type, fifo_in_partial);
        scf::YieldOp::create(else_b, loc, ValueRange{partial_read.getResult()});
      }
      seed = seed_if.getResult(0);
    } else {
      auto empty =
          tensor::EmptyOp::create(rewriter, loc, output_tensor_type.getShape(),
                                  output_tensor_type.getElementType());
      seed = empty.getResult();
    }

    // Build the nested loops, threading the accumulator through each level.
    auto nested = buildNestedForLoops(rewriter, loc, ctx, dim_sizes, start,
                                      step, ValueRange{seed});

    // Now populate the innermost loop body (before its yield).
    scf::ForOp innermost = nested.innermost_loop;
    Operation* inner_yield = innermost.getBody()->getTerminator();

    OpBuilder body_builder(inner_yield);

    // The iter_arg of the innermost loop is the accumulator carried in.
    Value carry = innermost.getRegionIterArgs().front();

    // read_from_fifo: one slice per innermost iteration.
    auto slice = ktdf::ReadFromFifoOp::create(body_builder, loc,
                                              one_row_tensor_type, fifo_in);

    // Clone the original linalg.generic, remapping its operands.
    IRMapping mapping;
    mapping.map(generic_op.getInputs().front(), slice.getResult());
    mapping.map(generic_op.getOutputs().front(), carry);
    auto new_generic = cast<linalg::GenericOp>(
        body_builder.clone(*generic_op.getOperation(), mapping));
    Value updated_carry = new_generic.getResult(0);

    if (fifo_out) {
      // The generic's result feeds a write_to_fifo directly.  Emit the
      // guarded write on the last iteration and drop the original write op.
      Value is_last = buildAllLast(body_builder, loc, nested.ivs, last_vals);
      auto if_op = scf::IfOp::create(body_builder, loc, TypeRange{}, is_last,
                                     /*withElseRegion=*/false);
      OpBuilder then_builder(if_op.getThenRegion().front().getTerminator());
      ktdf::WriteToFifoOp::create(then_builder, loc, updated_carry, fifo_out);
    } else {
      // The generic's result feeds other ops (e.g. a downstream inner-dim
      // reduction).  Replace all uses of the original generic with the
      // outermost loop result so those ops pick up the fully-accumulated
      // tensor after the loop completes.
      generic_op.getResult(0).replaceAllUsesWith(
          nested.outermost_loop.getResult(0));
      // write_to_fifo has no results so use_empty() is always true; exclude
      // it from the erase list so it is preserved together with the ops that
      // feed it (G2 etc).
      llvm::erase_if(
          to_erase, [](Operation* op) { return isa<ktdf::WriteToFifoOp>(op); });
    }

    // Replace the placeholder yield in the innermost loop with the real one.
    inner_yield->setOperand(0, updated_carry);

    // Erase original body ops (reverse order, only if unused).
    for (auto* op : llvm::reverse(to_erase))
      if (op->use_empty()) rewriter.eraseOp(op);
  }

  // -------------------------------------------------------------------------
  // Rewrite the conditional-store stage with N nested scf.for loops:
  //   scf.for %r0 = 0 to D0
  //     scf.for %r1 = 0 to D1
  //       ...
  //         <all original body ops except the data_transfer>
  //         if (all ivs == last):
  //           <the data_transfer>
  // -------------------------------------------------------------------------
  void rewriteConditionalStoreStage(IRRewriter& rewriter, Location loc,
                                    MLIRContext* ctx, ktdf::StageOp stage,
                                    ArrayRef<int64_t> dim_sizes, Value start,
                                    Value step, ArrayRef<Value> last_vals) {
    Block* body = stage.getBody();

    // Find the data_transfer that must be conditioned.
    ktdf::DataTransferOp transfer;
    body->walk([&](ktdf::DataTransferOp xfer) {
      transfer = xfer;
      return WalkResult::interrupt();
    });

    // Capture an iterator to the first original op *before* the loop shell is
    // prepended.  iplist iterators survive insertions elsewhere in the list, so
    // this remains valid after buildNestedForLoops runs.
    auto first_original = body->begin();

    rewriter.setInsertionPointToStart(body);
    auto nested =
        buildNestedForLoops(rewriter, loc, ctx, dim_sizes, start, step);
    Block* inner_body = nested.innermost_loop.getBody();
    Operation* inner_term = inner_body->getTerminator();

    // Splice [first_original, body->end()) into the innermost loop body before
    // its yield.
    inner_body->getOperations().splice(inner_term->getIterator(),
                                       body->getOperations(), first_original,
                                       body->end());

    if (!transfer) return;  // no transfer found — nothing to guard

    // Wrap just the data_transfer in scf.if (all ivs == last).
    rewriter.setInsertionPoint(transfer);
    Value is_last = buildAllLast(rewriter, loc, nested.ivs, last_vals);
    auto if_op = scf::IfOp::create(rewriter, loc, TypeRange{}, is_last,
                                   /*withElseRegion=*/false);
    // Move the transfer inside the then-block.
    transfer->moveBefore(if_op.getThenRegion().front().getTerminator());
  }
};

}  // namespace

auto mlir::ktdf::createReductionLoopExposurePass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionLoopExposurePass>();
}
