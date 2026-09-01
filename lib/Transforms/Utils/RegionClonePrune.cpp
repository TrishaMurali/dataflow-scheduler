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

#include "dataflow-scheduler/Transforms/Utils/RegionClonePrune.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define PRUNE_ANCHOR_MARKER "prune_anchor"

using namespace scheduler;

namespace {

/// True iff some op in `kept` reads `iv` itself.
///
/// Itself, rather than "has an ancestor in `kept`": a kept enclosing loop is an
/// ancestor of everything within it, so that question answers yes for the
/// variable of every loop inside one and no loop is ever dropped. Reaching the
/// variable through a chain of ops is the caller's part -- it looks at a loop
/// again each time an op inside it joins `kept`.
bool ivUsedByKept(mlir::Value iv,
                  const llvm::DenseSet<mlir::Operation*>& kept) {
  return llvm::any_of(iv.getUsers(), [&](mlir::Operation* user) {
    return kept.contains(user);
  });
}

/// Compute the set of cloned ops to keep, starting from `anchor` and
/// walking transitively (a) operand-producing defs and (b) parent ops,
/// stopping at `prune_root` (never add prune_root itself or anything
/// above it).
///
/// When walking up to a parent scf.for, the loop is only added to the kept
/// set if its induction variable is actually used by something already kept.
/// If the IV is unused, the loop is skipped and the walk continues to the
/// loop's parent — so irrelevant loop skeletons are not cloned. Whether a
/// skipped loop may then go is bodyCanRunOnce()'s question, asked once the set
/// is complete.
void computeKeptSet(mlir::Operation* anchor, mlir::Operation* prune_root,
                    llvm::DenseSet<mlir::Operation*>& kept) {
  llvm::SmallVector<mlir::Operation*> worklist;
  worklist.push_back(anchor);
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    if (op == prune_root || op == nullptr) {
      continue;
    }
    if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      // Only keep this loop if its IV is used by something already kept.
      // If not, skip it and walk up to its parent instead.
      if (!ivUsedByKept(for_op.getInductionVar(), kept)) {
        worklist.push_back(for_op->getParentOp());
        continue;
      }
    }
    if (!kept.insert(op).second) {
      continue;
    }
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation* def = operand.getDefiningOp()) {
        worklist.push_back(def);
      }
    }
    if (mlir::Operation* parent = op->getParentOp()) {
      worklist.push_back(parent);
    }
  }
}

/// Whether the body of \p loop can run once in place of every iteration.
///
/// The pruning leaves \p anchor and the ops it reads, and this is only asked of
/// a loop whose variable none of those reads -- so every iteration computes the
/// same values and performs the same transfer. Once stands for all of them
/// where nothing else depends on there having been several: the loop carries no
/// value out, and nothing surviving in it besides the anchor touches memory.
///
/// The anchor is the exception on purpose. Collapsing repeats of one transfer
/// is what hoisting an invariant transfer does in the first place.
bool bodyCanRunOnce(mlir::scf::ForOp loop, mlir::Operation* anchor,
                    const llvm::DenseSet<mlir::Operation*>& kept) {
  if (loop->getNumResults() != 0) {
    return false;
  }

  bool once = true;
  loop.getBody()->walk([&](mlir::Operation* op) {
    if (op == anchor || !kept.contains(op)) return;
    if (!mlir::isMemoryEffectFree(op)) once = false;
  });
  return once;
}

}  // namespace

mlir::Operation* scheduler::cloneRegionAndPruneToAnchor(
    mlir::Region& source_region, mlir::Block* target_block,
    mlir::Block::iterator ip, mlir::Operation* keep_anchor,
    mlir::IRMapping& value_map) {
  mlir::OpBuilder builder(target_block, ip);

  mlir::Operation* prune_root = target_block->getParentOp();
  assert(prune_root && "target_block must have a parent op");

  // Tag keep_anchor so its clone can be found by attribute after cloning,
  // avoiding a full origin_map traversal of the cloned subtree.
  keep_anchor->setAttr(PRUNE_ANCHOR_MARKER, builder.getI32IntegerAttr(1));

  // Clone top-level ops from source_region's entry block.
  mlir::Block& source_block = source_region.front();
  llvm::SmallVector<mlir::Operation*> cloned_top_level;
  for (mlir::Operation& op : source_block) {
    cloned_top_level.push_back(builder.clone(op, value_map));
  }

  // Remove the marker from the original now that cloning is done.
  keep_anchor->removeAttr(PRUNE_ANCHOR_MARKER);

  // Find the cloned anchor by locating the marker in the cloned tree.
  mlir::Operation* cloned_anchor = nullptr;
  for (mlir::Operation* top : cloned_top_level) {
    top->walk([&](mlir::Operation* op) {
      if (!op->hasAttr(PRUNE_ANCHOR_MARKER)) return mlir::WalkResult::advance();
      cloned_anchor = op;
      op->removeAttr(PRUNE_ANCHOR_MARKER);
      return mlir::WalkResult::interrupt();
    });
    if (cloned_anchor) break;
  }
  assert(cloned_anchor && "anchor not found in cloned region");

  // Compute kept set rooted in the cloned anchor.
  llvm::DenseSet<mlir::Operation*> kept;
  computeKeptSet(cloned_anchor, prune_root, kept);

  // Collect erasure candidates in post-order (children before parents).
  // For a non-kept scf.for whose body contains kept ops, inline its body
  // into the loop's parent block before erasing — this handles skipped
  // skeleton loops without a separate move pass.
  llvm::SmallVector<mlir::Operation*> erasable;
  for (mlir::Operation* top : cloned_top_level) {
    top->walk([&](mlir::Operation* op) {
      if (kept.contains(op)) return;
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) return;

      // For a skipped scf.for: splice its body ops (except the terminator)
      // immediately before the loop op in its parent block, then erase. A loop
      // whose iterations are not all the same one stays where it is, holding
      // whatever the pruning left in it.
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        if (!bodyCanRunOnce(loop, cloned_anchor, kept)) return;

        mlir::Block* loop_body = &op->getRegion(0).front();
        // Splice everything up to (but not including) the terminator.
        op->getBlock()->getOperations().splice(
            mlir::Block::iterator(op), loop_body->getOperations(),
            loop_body->begin(), loop_body->getTerminator()->getIterator());
      }

      erasable.push_back(op);
    });
  }

  // Erase. Drop remaining uses first to avoid dangling-use asserts.
  for (mlir::Operation* op : erasable) {
    op->dropAllUses();
  }
  for (mlir::Operation* op : erasable) {
    op->erase();
  }

  return cloned_anchor;
}
