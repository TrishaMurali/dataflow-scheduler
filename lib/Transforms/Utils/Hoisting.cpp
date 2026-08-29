//===-------------------------------------------------------------*- c++ -*-==//
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

#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/LoopInvariantCodeMotionUtils.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/CodeMotion.h"

#define DEBUG_TYPE "dataflow-scheduler-hoisting"

using namespace scheduler;

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

/// Finds the topmost Region which dominates all uses held by @p op and its
/// children.
[[nodiscard]] auto findSSAEarliestDominator(mlir::Operation* op)
    -> mlir::Region* {
  // Walk upwards once to collect all ancestor regions.
  llvm::SmallVector<mlir::Region*> parent_regions;
  for (auto* parent = op->getParentRegion(); parent;
       parent = parent->getParentRegion()) {
    parent_regions.push_back(parent);
  }

  const auto visitor = [&](mlir::Operation* child) -> mlir::WalkResult {
    for (auto operand : child->getOperands()) {
      // Erase all regions above the one where the operand is defined.
      if (const auto it = llvm::find(parent_regions, operand.getParentRegion());
          it != parent_regions.end()) {
        parent_regions.erase(std::next(it), parent_regions.end());
        if (parent_regions.size() == 1) {
          // The last remaining region (op->getParentRegion()) is the trivial
          // dominator.
          return mlir::WalkResult::interrupt();
        }
      }
    }

    if (child->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  };
  op->walk<mlir::WalkOrder::PreOrder>(visitor);

  return parent_regions.back();
}

}  // namespace

auto scheduler::findHoistingTarget(
    mlir::Operation* op,
    llvm::function_ref<bool(mlir::Region*)> should_hoist_out_of)
    -> mlir::Operation* {
  if (op->mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
    // Terminators can never be hoisted.
    return op;
  }

  auto* result = op;
  const auto has_uses = !op->use_empty();
  auto* const ssa_dominator = findSSAEarliestDominator(op);

  while (auto* const target = result->getParentOp()) {
    if (has_uses &&
        target->mightHaveTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      // The definitions would no longer be able to reach the uses if we hoist
      // to the parent region.
      LDBG() << "can't hoist " << mlir::OpWithFlags(op, kSkipRegions);
      LDBG() << "  above " << mlir::OpWithFlags(target, kSkipRegions);
      LDBG() << "  target is isolated from above";
      break;
    }

    auto* const source = result->getParentRegion();
    if (source == ssa_dominator) {
      // The uses would no longer be reached by the definitions if we hoist to
      // the parent region.
      LDBG() << "can't hoist " << mlir::OpWithFlags(op, kSkipRegions);
      LDBG() << "  above " << mlir::OpWithFlags(target, kSkipRegions);
      LDBG() << "  uses are not dominated";
      break;
    }

    if (!should_hoist_out_of(source)) {
      break;
    }

    result = target;
  }

  return result;
}

auto scheduler::hoistInvariants(mlir::Operation* source) -> size_t {
  if (auto iface = mlir::dyn_cast<mlir::LoopLikeOpInterface>(source); iface) {
    // Hoist all pure operations without inter-iteration dependencies directly
    // in front of the loop.
    return mlir::moveLoopInvariantCode(iface);
  }

  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(source); generic) {
    // Hoist all pure operations without inter-iteration dependencies directly
    // in front of the 'linalg.generic' operation.
    return mlir::moveLoopInvariantCode(
        {&generic.getBodyRegion()},
        [&](mlir::Value value, mlir::Region* /*region*/) -> bool {
          return value.getParentRegion()->isProperAncestor(
              &generic.getBodyRegion());
        },
        [&](mlir::Operation* op, mlir::Region* /*region*/) -> bool {
          return mlir::isPure(op);
        },
        [&](mlir::Operation* op, mlir::Region* /*region*/) {
          op->moveBefore(generic);
        });
  }

  if (auto pipeline = mlir::dyn_cast<mlir::ktdf::PipelineOp>(source);
      pipeline) {
    // Hoist all pure operations without dependencies directly in front of the
    // 'ktdf.pipeline' operation.
    return mlir::ktdf::hoistPipelineContents(
        pipeline, [&](mlir::Operation* op) -> mlir::ktdf::PipelineAnchor {
          return mlir::isPure(op) ? mlir::ktdf::PipelineAnchor::Parent
                                  : mlir::ktdf::PipelineAnchor::Stage;
        });
  }

  return 0;
}

namespace {

template <class... Effects>
auto visitRestrictedUsersIn(
    mlir::Value restricted, mlir::Region* region,
    llvm::function_ref<llvm::LogicalResult(
        mlir::Operation*, llvm::ArrayRef<mlir::MemoryEffects::EffectInstance>)>
        visitor) -> llvm::LogicalResult {
  // Process just the users, since the location is restricted.
  for (auto* const user : restricted.getUsers()) {
    if (llvm::isa<mlir::ViewLikeOpInterface>(user)) {
      // There are potentially aliasing values.
      LDBG() << "giving up on " << restricted;
      LDBG() << "  potential alias via "
             << mlir::OpWithFlags(user, kSkipRegions);
      return llvm::failure();
    }

    if (user->getParentRegion()->isProperAncestor(region)) {
      // User is outside the region.
      continue;
    }

    auto effects = mlir::getEffectsRecursively(user);
    if (!effects) {
      // We can't judge the effects of this operation.
      LDBG() << "giving up on " << restricted;
      LDBG() << "  unknown user " << mlir::OpWithFlags(user, kSkipRegions);
      return llvm::failure();
    }

    llvm::erase_if(
        *effects,
        [&](const mlir::MemoryEffects::EffectInstance& effect) -> bool {
          return !llvm::isa<Effects...>(effect.getEffect()) ||
                 effect.getValue() != restricted;
        });
    if (effects->empty()) {
      continue;
    }
    if (failed(visitor(user, *effects))) {
      return llvm::failure();
    }
  }

  return llvm::success();
}

}  // namespace

auto scheduler::findSingleWriteIn(mlir::Value restricted, mlir::Region* region)
    -> mlir::Operation* {
  mlir::Operation* result = nullptr;
  const auto visit_effects =
      [&](mlir::Operation* user,
          mlir::ArrayRef<mlir::MemoryEffects::EffectInstance> /*effects*/)
      -> llvm::LogicalResult {
    if (result) {
      LDBG() << "giving up on " << restricted;
      LDBG() << "  written by " << mlir::OpWithFlags(result, kSkipRegions);
      LDBG() << "  written by " << mlir::OpWithFlags(user, kSkipRegions);
      LDBG() << "  and potentially more";
      return llvm::failure();
    }
    result = user;
    return llvm::success();
  };
  if (llvm::failed(visitRestrictedUsersIn<mlir::MemoryEffects::Write>(
          restricted, region, visit_effects))) {
    return nullptr;
  }

  return result;
}

auto scheduler::findInvariantWriteIn(mlir::Value restricted,
                                     mlir::Region* region) -> mlir::Operation* {
  auto* const result = findSingleWriteIn(restricted, region);
  if (!result) {
    return nullptr;
  }

  // Ensure that the write is invariant. We use a simple algorithm that
  // checks whether the operands to the write are defined outside of the
  // region of interest, but ignoring the restricted location and the children
  // of the write we found.
  const auto is_invariant_value = [&](mlir::Value value) -> bool {
    if (value == restricted) {
      return true;
    }
    if (result->isAncestor(value.getParentRegion()->getParentOp())) {
      return true;
    }

    if (auto* const definition = value.getDefiningOp();
        definition && !mlir::isMemoryEffectFree(definition)) {
      // Give up.
      return false;
    }

    return value.getParentRegion()->isProperAncestor(region);
  };
  const auto is_invariant_op = [&](mlir::Operation* op) -> bool {
    for (auto operand : op->getOperands()) {
      if (!is_invariant_value(operand)) {
        LDBG() << "giving up on " << mlir::OpWithFlags(result, kSkipRegions);
        LDBG() << "  " << operand << " is not invariant";
        return false;
      }
    }

    return true;
  };
  const auto visitor = [&](mlir::Operation* child) -> mlir::WalkResult {
    if (!is_invariant_op(child)) {
      return mlir::WalkResult::interrupt();
    }

    if (child->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  };
  if (result->walk<mlir::WalkOrder::PreOrder>(visitor).wasInterrupted()) {
    return nullptr;
  }

  return result;
}
