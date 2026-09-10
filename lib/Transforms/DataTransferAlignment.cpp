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
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/ADT/SmallSetVector.h"
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

//===----------------------------------------------------------------------===//
// widenAlloc
//
// If ts.transfer writes into a ct_local memref, traces back to the defining
// AllocOp and widens pa.targetDim to E (pa.requiredSize). Non-ct_local
// destinations are skipped. Missing AllocOp on a ct_local destination is an
// error.
//===----------------------------------------------------------------------===//
static void widenAlloc(const TransferStep& ts, const PipelineAnalysis& pa,
                       mlir::OpBuilder& builder) {
  const int64_t E = pa.requiredSize;

  // Only act when the transfer destination is a ct_local memref.
  auto dest_type =
      mlir::dyn_cast<mlir::MemRefType>(ts.transfer.getDestination().getType());
  if (!dest_type) return;

  auto mem_space = mlir::dyn_cast_or_null<mlir::ktdp::MemorySpaceAttr>(
      dest_type.getMemorySpace());
  if (!mem_space || mem_space.getValue() != mlir::ktdp::MemorySpace::ct_local)
    return;

  // Destination is ct_local — trace back to the defining AllocOp.
  // A ct_local buffer must be backed by a memref.alloc.
  mlir::Value base = ts.transfer.getDestination();
  while (auto view_like =
             mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
                 base.getDefiningOp()))
    base = view_like.getViewSource();

  auto alloc = mlir::dyn_cast_or_null<mlir::memref::AllocOp>(
      base.getDefiningOp());
  if (!alloc) {
    ts.transfer->emitError(
        "widenAlloc: ct_local destination has no backing AllocOp");
    return;
  }

  mlir::MemRefType orig_type = alloc.getType();
  llvm::SmallVector<int64_t> new_shape(orig_type.getShape());

  // Already widened — nothing to do.
  if (new_shape[pa.targetDim] == E) {
    LDBG(1) << "  widenAlloc: already widened, skipping";
    return;
  }

  // Widen targetDim to E, keeping the original flat ct_local layout.
  new_shape[pa.targetDim] = E;

  mlir::MemRefType new_type =
      mlir::MemRefType::get(new_shape, orig_type.getElementType(),
                            orig_type.getLayout(),
                            orig_type.getMemorySpace());

  builder.setInsertionPoint(alloc);
  auto new_alloc = mlir::memref::AllocOp::create(
      builder, alloc.getLoc(), new_type, alloc.getDynamicSizes());

  LDBG(1) << "  widenAlloc: " << orig_type << " → " << new_type;
  alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
  alloc.erase();
}

//===----------------------------------------------------------------------===//
// adjustLoopBounds
//
// Divides the upper bound of sa.targetDimFor by E (pa.requiredSize). Only
// the loop iterating over the dimension for this stage is adjusted —
// outer tile loops and other stages' loops are left untouched.
// Called once per StageAnalysis before visiting its transfers.
//===----------------------------------------------------------------------===//
static void adjustLoopBounds(const StageAnalysis& sa,
                              const PipelineAnalysis& pa,
                              mlir::OpBuilder& builder) {
  if (!sa.targetDimFor) return;

  mlir::scf::ForOp loop = sa.targetDimFor;
  builder.setInsertionPoint(loop);

  mlir::Value e_val = mlir::arith::ConstantIndexOp::create(
      builder, loop.getLoc(), pa.requiredSize).getResult();

  // new_ub = original_ub/E
  mlir::Value new_ub = mlir::arith::DivUIOp::create(
      builder, loop.getLoc(),
      loop.getUpperBound(), e_val).getResult();
  loop.setUpperBound(new_ub);

  LDBG(1) << "  adjustLoopBounds: divided ub by " << pa.requiredSize
          << " on targetDimFor at " << loop.getLoc();
}

//===----------------------------------------------------------------------===//
// rewriteTransferShape
//
// Replace ts.transfer with a new DataTransferOp where pa.targetDim is widened
// to E. All other dimensions are preserved. No hardcoding — pa.targetDim and
// pa.requiredSize drive the change entirely.
//===----------------------------------------------------------------------===//
static void rewriteTransferShape(const TransferStep& ts,
                                 const PipelineAnalysis& pa,
                                 mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult e_ofr =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), pa.requiredSize);

  // Copy existing sizes and update only pa.targetDim.
  auto new_src_sizes = op.getMixedSourceSizes();
  new_src_sizes[pa.targetDim] = e_ofr;

  auto new_dst_sizes = op.getMixedDestSizes();
  new_dst_sizes[pa.targetDim] = e_ofr;

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

// Shrink pa.targetDim to 1 and stamp transfer_mode as mode ("splat" or
// "extract"). All other dimensions are preserved, mirroring rewriteTransferShape.
static void rewriteTransferShrink(const TransferStep& ts,
                                  const PipelineAnalysis& pa,
                                  llvm::StringRef mode,
                                  mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult one =
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1);

  // Only shrink targetDim, leave all other dimensions as-is.
  auto new_src_sizes = op.getMixedSourceSizes();
  if (pa.targetDim < (int64_t)new_src_sizes.size())
    new_src_sizes[pa.targetDim] = one;

  auto new_dst_sizes = op.getMixedDestSizes();
  if (pa.targetDim < (int64_t)new_dst_sizes.size())
    new_dst_sizes[pa.targetDim] = one;

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

  new_op->setDiscardableAttr(
      mlir::StringAttr::get(ctx, "transfer_mode"),
      mlir::StringAttr::get(ctx, mode));

  LDBG(1) << "  rewriteTransferShrink(" << mode << "): shrunk dim "
          << pa.targetDim << " to 1: " << new_op;
  op.erase();
}

//===----------------------------------------------------------------------===//
// insertLoopAroundPipeline
//
// Wrap the nested ktdf.pipeline in a new scf.for %col = 0 to E step 1.
// This is the column-element loop that is required so that
// transfers can index each column of the widened alloc
// independently.
//===----------------------------------------------------------------------===//
static void insertLoopAroundPipeline(PipelineAnalysis* nestedPA, int64_t E,
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

//===----------------------------------------------------------------------===//
// fixPipeline  (depth-first recursive rewrite)
//
// Consumes the PipelineAnalysis tree and applies the corrective actions:
//   StageAnalysis level: insert col loop around nested pipeline, or visit
//                         TransferSteps directly
//   TransferStep level: widen ct_local alloc at targetDim to E,
//                         divide targetDimFor loop bound by E,
//                         rewrite transfer shape at targetDim to E,
//                         or shrink targetDim to 1 + stamp transfer_mode
//                         for FIFO transfers (splat/extract)
//===----------------------------------------------------------------------===//
static mlir::LogicalResult fixPipeline(PipelineAnalysis& pa,
                                       mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " E=" << pa.requiredSize;

  for (StageAnalysis& sa : pa.stages) {

    if (sa.nestedPipeline != nullptr) {
      // This stage contains a nested pipeline — insert the column loop and
      // recurse.
      insertLoopAroundPipeline(sa.nestedPipeline.get(), pa.requiredSize, builder);
      if (mlir::failed(fixPipeline(*sa.nestedPipeline, builder)))
        return mlir::failure();
      continue;
    }

    // Adjust the dimension loop bound for this stage before visiting
    // its transfers.
    adjustLoopBounds(sa, pa, builder);

    for (TransferStep& ts : sa.transfers) {
      if (ts.isIllegal || ts.isDisplaced) {
        // 1. Widen the ct_local destination alloc at targetDim.
        widenAlloc(ts, pa, builder);
        // 2. Rewrite the transfer sizes at targetDim to E.
        rewriteTransferShape(ts, pa, builder);
        continue;
      }

      if (ts.needsSplat) {
        rewriteTransferShrink(ts, pa, "splat", builder);
        continue;
      }

      if (ts.needsExtract) {
        rewriteTransferShrink(ts, pa, "extract", builder);
      }
    }
  }
  return mlir::success();
}

}  // namespace