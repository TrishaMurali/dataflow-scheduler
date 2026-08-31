//===-- ReductionOpacification.cpp ------------------------------*- c++ -*-===//
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
// ReductionOpacification: replace inner-dimension reduction linalg.generic
// ops with ktdf.opaque.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "reduction-opacification"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONOPACIFICATIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Reduction Opacification pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Identify an inner-dimension reduction linalg.generic in buffered form:
//   ins(%alloc : memref<...>)
//   outs(%sv   : memref<..., strided>)   where %sv = memref.subview %alloc
//
// Returns the subview op on success, nullptr otherwise.
// ---------------------------------------------------------------------------
static memref::SubViewOp matchInnerDimGeneric(linalg::GenericOp generic_op) {
  if (generic_op.getInputs().empty() || generic_op.getOutputs().empty())
    return {};
  // ins[0] must be a MemRefType (buffered form produced by
  // MapReductionPartials)
  if (!isa<MemRefType>(generic_op.getInputs().front().getType())) return {};
  // outs[0] must be defined by a memref.subview
  return generic_op.getOutputs().front().getDefiningOp<memref::SubViewOp>();
}

// ---------------------------------------------------------------------------
// Replace a matched inner-dim reduction generic with ktdf.opaque and remove
// its subview:
//
//   linalg.generic ins(%alloc) outs(%sv)      ← %sv = memref.subview %alloc
//   →
//   ktdf.opaque "simd_reduction" ins(%alloc) outs(%alloc)
//
// The subview is erased once the generic (its only user) is gone.
// ---------------------------------------------------------------------------
static void opaqifyReduction(linalg::GenericOp generic_op,
                             memref::SubViewOp subview) {
  LDBG(1) << PASS_NAME
      ": replacing inner-dim reduction generic with "
      "ktdf.opaque \"simd_reduction\"";

  // %alloc is the source of the subview — use it for both ins and outs.
  Value alloc = subview.getSource();
  assert(alloc == generic_op.getInputs().front() &&
         "expected ins[0] to be the subview source (%alloc)");

  OpBuilder builder(generic_op);
  // No result tensors: the opaque op writes in-place through outs(%alloc).
  auto* ctx = builder.getContext();
  auto opaque = ktdf::OpaqueOp::create(builder, generic_op.getLoc(),
                                       /*resultTypes=*/TypeRange{},
                                       /*template_name=*/
                                       builder.getStringAttr("simd_reduction"),
                                       /*inputs=*/ValueRange{alloc},
                                       /*outputs=*/ValueRange{alloc});

  // Discardable attributes required by LowerOpaquePattern in KTDFLowToDFIR.
  opaque->setDiscardableAttr(builder.getStringAttr("func_name"),
                             builder.getStringAttr("simdreduction"));
  opaque->setDiscardableAttr(
      builder.getStringAttr("dataflow_scheduler.register_names"),
      builder.getArrayAttr({
          builder.getStringAttr("t0_0"),  // ins[0]
          builder.getStringAttr("t0_0"),  // outs[0]
      }));
  opaque->setDiscardableAttr(
      builder.getStringAttr("parameter_dictionary"),
      DictionaryAttr::get(
          ctx, {builder.getNamedAttr("unroll", builder.getStringAttr("1"))}));

  generic_op.erase();
  // Erase the subview only after the generic that used it is gone.
  if (subview.use_empty()) subview.erase();
}

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------
struct ReductionOpacificationPass
    : public ktdf::impl::ReductionOpacificationPassBase<
          ReductionOpacificationPass> {
  using ReductionOpacificationPassBase<
      ReductionOpacificationPass>::ReductionOpacificationPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    // Collect work items before mutating to avoid iterator invalidation.
    SmallVector<std::pair<linalg::GenericOp, memref::SubViewOp>> work_items;
    module.walk([&](linalg::GenericOp generic_op) {
      if (!generic_op->getParentOfType<ktdf::StageOp>()) return;
      auto subview = matchInnerDimGeneric(generic_op);
      if (!subview) return;
      work_items.emplace_back(generic_op, subview);
    });

    for (auto& [generic_op, subview] : work_items)
      opaqifyReduction(generic_op, subview);
  }
};

}  // namespace

auto mlir::ktdf::createReductionOpacificationPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionOpacificationPass>();
}
