//===-- HoistConstantStorage.cpp --------------------------------*- c++ -*-===//
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

#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/CodeMotion.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"

#define PASS_NAME "hoist-constant-storage"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Hoist Constant Storage pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_HOISTCONSTANTSTORAGEPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

struct HoistConstantStoragePass
    : impl::HoistConstantStoragePassBase<HoistConstantStoragePass> {
  using HoistConstantStoragePassBase::HoistConstantStoragePassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    getOperation()->walk([&](mlir::ktdf::PipelineOp pipeline) {
      hoistConstantStorage(pipeline);
    });
  }

 private:
  void hoistConstantStorage(mlir::ktdf::PipelineOp pipeline) {
    mlir::OpBuilder builder(pipeline);

    llvm::DenseSet<mlir::Operation*> invariant_writes;
    const auto hoist = [&](mlir::Operation* op) {
      // Find the target scope to hoist to.
      auto* const target = findHoistingTarget(
          op, [](mlir::Region* /*region*/) -> bool { return true; });
      assert(target->isProperAncestor(op));

      if (!invariant_writes.contains(op)) {
        op->moveBefore(target);
        return;
      }

      // Create a pipeline with a single stage and hoist into it.
      builder.setInsertionPoint(target);
      mlir::ktdf::PipelineOp::create(
          builder, pipeline.getLoc(),
          [&](mlir::OpBuilder& builder, mlir::Location /*loc*/) {
            auto stage =
                mlir::ktdf::StageOp::create(builder, op->getLoc(), {}, {});
            stage.setApplicableUnitsAttr(
                op->getParentOfType<mlir::ktdf::StageOp>()
                    .getApplicableUnitsAttr());
            op->moveBefore(stage.getBody(), stage.getBody()->end());
          });
    };

    // Hoist allocations and their invariant writes, as well as pure ops so
    // that we munch as many as possible.
    mlir::ktdf::hoistPipelineContents(
        pipeline,
        [&](mlir::Operation* op) -> mlir::ktdf::PipelineAnchor {
          if (mlir::isPure(op) || invariant_writes.contains(op)) {
            return mlir::ktdf::PipelineAnchor::Parent;
          }

          if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op); alloc) {
            auto* const write = findInvariantWriteIn(alloc.getResult(),
                                                     &pipeline.getBodyRegion());
            if (write) {
              LDBG() << "found invariant write "
                     << mlir::OpWithFlags(write, kSkipRegions);
              LDBG() << "hoisting allocation "
                     << mlir::OpWithFlags(alloc, kSkipRegions);
              invariant_writes.insert(write);
              ++num_hoisted;
              return mlir::ktdf::PipelineAnchor::Parent;
            }
          }

          return mlir::ktdf::PipelineAnchor::Stage;
        },
        hoist);
  }
};

}  // namespace
