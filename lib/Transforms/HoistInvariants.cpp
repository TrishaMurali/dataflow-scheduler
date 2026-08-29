//===-- HoistInvariants.cpp -------------------------------------*- c++ -*-===//
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
//===----------------------------------------------------------------------===//"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/LoopInvariantCodeMotionUtils.h>

#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"

#define PASS_NAME "hoist-invariants"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Hoist Invariants pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_HOISTINVARIANTSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

struct HoistInvariantsPass
    : impl::HoistInvariantsPassBase<HoistInvariantsPass> {
  using HoistInvariantsPassBase::HoistInvariantsPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    getOperation()->walk(
        [&](mlir::Operation* op) { num_hoisted += hoistInvariants(op); });
  }
};

}  // namespace
