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
//
// Pass: -parallelize-loops-across-instances
//
// Rewrites parallel scf.for loops enclosing per-corelet ktdf.pipelines into
// ktdf.parallel ops that distribute iterations across hardware instances.
// The number of instances is computed from the device architecture.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SetVector.h>

#include <optional>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/ApplicableUnits.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/PipelineScope.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/NodeEndpoints.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "parallelize-loops-across-instances"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Parallelize Loops Across Instances pass"),
    llvm::cl::init(false));

// Temporary workaround: treat every group as a single instance so no
// ktdf.parallel is created. Flip the default to true to make this permanent
// until the splitter checks tile-size feasibility (dynamic trip counts can end
// up not divisible by num_instances, which fails in ParallelLowering).
static llvm::cl::opt<bool> SingleInstance(
    PASS_NAME "-single-instance",
    llvm::cl::desc("Do not split loops across instances"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_PARALLELIZELOOPSACROSSINSTANCESPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

//===----------------------------------------------------------------------===//
// Balance Status Enum
//===----------------------------------------------------------------------===//

enum class BalanceStatus {
  kBalanced,       // Trip count is evenly divisible by num_instances
  kUnbalanced,     // Trip count is not evenly divisible by num_instances
  kBalanceUnknown  // Trip count is dynamic, balance cannot be determined
};

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

struct Candidate {
  mlir::ktdf::PipelineOp pipeline;
  mlir::scf::ForOp loop;
  // Dynamic upper bound Value
  mlir::Value upper_bound;
  // Static trip count if known
  std::optional<int64_t> trip_count;
  // Balance status indicating whether trip count is evenly divisible by
  // num_instances
  BalanceStatus balance_status;
  // Collection of groups this candidate can be mapped to.
  llvm::SmallSetVector<mlir::ktdf_arch::GroupOp, 4> applicable_groups;
};

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Returns true if @p group acts as a named alias for @p resource rather than
/// a true enclosing scope.
///
/// A group is a named alias when at least one of its yielded results traces
/// back (via endpoint analysis) to the same underlying Node as @p resource.
/// This is transitive through any depth of nested groups, unlike a direct
/// yield-operand check.
[[nodiscard]] auto isAliasGroup(mlir::ktdf_arch::Resource resource,
                                mlir::ktdf_arch::GroupOp group) -> bool {
  auto target = resource->getResult(0);
  for (auto result : group->getResults()) {
    if (mlir::ktdf_arch::getEndpoint(result) == target) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] auto findEnclosingGroup(
    llvm::ArrayRef<ResourceType> resources,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds)
    -> mlir::ktdf_arch::GroupOp {
  if (resources.empty()) {
    return nullptr;
  }

  mlir::ktdf_arch::GroupOp result;
  for (auto kind : resources) {
    auto resource = resource_kinds.getInstance(kind);
    if (!resource) {
      return nullptr;
    }
    auto group = resource->getParentOfType<mlir::ktdf_arch::GroupOp>();
    if (!group) {
      return nullptr;
    }
    // If the group is a transparent alias for this resource (its result traces
    // back to the same Node endpoint), it is not a true enclosing scope.
    // Step past it so that the LCA search operates at the correct granularity.
    if (isAliasGroup(resource, group)) {
      group = group->getParentOfType<mlir::ktdf_arch::GroupOp>();
      if (!group) {
        return nullptr;
      }
    }
    if (!result || group->isAncestor(result)) {
      result = group;
      continue;
    }

    while (!result->isAncestor(group)) {
      result = result->getParentOfType<mlir::ktdf_arch::GroupOp>();
      if (!result) {
        return nullptr;
      }
    }
  }

  return result;
}

[[nodiscard]] auto findEnclosingGroup(
    mlir::ktdf::PipelineOp pipeline,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds)
    -> mlir::ktdf_arch::GroupOp {
  return findEnclosingGroup(
      mlir::ktdf::collectPipelineApplicableUnits(pipeline).getArrayRef(),
      resource_kinds);
}

[[nodiscard]] auto getEquivalenceClass(mlir::ktdf_arch::GroupOp group)
    -> llvm::SmallSetVector<mlir::ktdf_arch::GroupOp, 4> {
  llvm::SmallSetVector<mlir::ktdf_arch::GroupOp, 4> result;
  result.insert(group);
  for (auto sibling : group->getBlock()->getOps<mlir::ktdf_arch::GroupOp>()) {
    if (sibling.getKind() == group.getKind()) {
      result.insert(sibling);
    }
  }
  return result;
}

/// Walk scope_loops outermost-to-innermost; return the first loop matching
/// the predicate `pred(loop, trip_count_opt)`.
/// For dynamic loops, trip_count_opt will be std::nullopt.
template <typename Pred>
std::optional<mlir::scf::ForOp> firstMatching(
    llvm::ArrayRef<mlir::scf::ForOp> scope_loops_innermost_first, Pred pred) {
  for (mlir::scf::ForOp loop : llvm::reverse(scope_loops_innermost_first)) {
    if (!mlir::ktdf::isParallelLoop(loop)) {
      continue;
    }
    auto trip = getStaticTripCount(loop);
    if (pred(loop, trip)) {
      return loop;
    }
  }
  return std::nullopt;
}

/// Three-pass loop selection: balanced first, imbalanced fallback, dynamic
/// last.
std::optional<mlir::scf::ForOp> selectLoop(
    llvm::ArrayRef<mlir::scf::ForOp> scope_loops_innermost_first,
    unsigned num_instances) {
  // Pass 1: trip >= N and trip % N == 0 (balanced static).
  auto balanced = firstMatching(
      scope_loops_innermost_first,
      [num_instances](mlir::scf::ForOp, std::optional<int64_t> trip) {
        if (!trip.has_value()) return false;
        return trip.value() >= static_cast<int64_t>(num_instances) &&
               trip.value() % static_cast<int64_t>(num_instances) == 0;
      });
  if (balanced.has_value()) {
    return balanced;
  }
  // Pass 2: trip >= N (imbalanced static).
  auto imbalanced = firstMatching(
      scope_loops_innermost_first,
      [num_instances](mlir::scf::ForOp, std::optional<int64_t> trip) {
        if (!trip.has_value()) return false;
        return *trip >= static_cast<int64_t>(num_instances);
      });
  if (imbalanced.has_value()) {
    return imbalanced;
  }
  // Pass 3: dynamic loops (balance unknown).
  return firstMatching(scope_loops_innermost_first,
                       [](mlir::scf::ForOp, std::optional<int64_t> trip) {
                         return !trip.has_value();
                       });
}

std::optional<Candidate> findCandidate(
    mlir::ktdf::PipelineOp pipeline,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  // Step 1+2: applicable_units.
  auto enclosing_group = findEnclosingGroup(pipeline, resource_kinds);
  if (!enclosing_group) {
    return std::nullopt;
  }
  LDBG(1) << " pipeline at " << pipeline->getLoc() << " is enclosed by "
          << enclosing_group->getLoc();
  auto applicable_groups = getEquivalenceClass(enclosing_group);
  const auto num_instances = applicable_groups.size();
  if (SingleInstance || num_instances < 2) {
    return std::nullopt;
  }
  LDBG(1) << " pipeline at " << pipeline->getLoc() << " has " << num_instances
          << " applicable groups ";

  // Step 3: enclosing scope.
  auto scope = mlir::ktdf::getPipelineEnclosingScope(pipeline);
  if (scope.loops.empty()) {
    LDBG(1) << " skip pipeline at " << pipeline.getLoc()
            << ": no enclosing scf.for scope";
    return std::nullopt;
  }

  // Step 4: loop selection.
  auto picked = selectLoop(scope.loops, num_instances);
  if (!picked.has_value()) {
    LDBG(1) << "skip pipeline at " << pipeline.getLoc()
            << ": no parallel loop in scope satisfies requirements";
    return std::nullopt;
  }
  mlir::scf::ForOp loop = *picked;
  mlir::Value upper_bound = loop.getUpperBound();
  auto trip = getStaticTripCount(loop);

  BalanceStatus balance_status;
  if (trip.has_value()) {
    // Static trip count - determine balance
    bool balanced = (*trip % static_cast<int64_t>(num_instances)) == 0;
    balance_status =
        balanced ? BalanceStatus::kBalanced : BalanceStatus::kUnbalanced;

    if (!balanced) {
      int64_t chunk = (*trip + num_instances - 1) / num_instances;
      int64_t leftover = *trip - (num_instances - 1) * chunk;
      LDBG(1) << " imbalanced (fallback) at " << loop.getLoc()
              << ": trip=" << *trip << ", num_instances=" << num_instances
              << ", chunks=" << chunk << "," << leftover;
    } else {
      LDBG(1) << " balanced candidate at " << loop.getLoc()
              << ": trip=" << *trip << ", num_instances=" << num_instances;
    }
  } else {
    // Dynamic trip count - balance unknown
    balance_status = BalanceStatus::kBalanceUnknown;

    LDBG(1) << " dynamic candidate at " << loop.getLoc()
            << ": trip count is dynamic"
            << ", num_instances=" << num_instances;
  }

  return Candidate{pipeline, loop,           upper_bound,
                   trip,     balance_status, std::move(applicable_groups)};
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

void rewrite(const Candidate& c) {
  mlir::scf::ForOp loop = c.loop;
  mlir::Location loc = loop.getLoc();
  mlir::Value lb = loop.getLowerBound();
  mlir::Value ub = c.upper_bound;
  mlir::Value step = loop.getStep();
  mlir::Value old_iv = loop.getInductionVar();
  mlir::Block* old_body = loop.getBody();
  const auto num_instances = c.applicable_groups.size();

  // Build ktdf.parallel at the scf.for's position. The custom builder
  // creates the body block with two index args (IV + instance id).
  mlir::OpBuilder builder(loop);
  auto parallel = mlir::ktdf::ParallelOp::create(
      builder, loc,
      /*lowerBounds=*/mlir::ValueRange{lb},
      /*upperBounds=*/mlir::ValueRange{ub},
      /*steps=*/mlir::ValueRange{step},
      /*numInstances=*/static_cast<int64_t>(num_instances));

  mlir::Block* new_body = parallel.getBody();

  // Splice the old body (excluding the trailing scf.yield terminator) into
  // the new ktdf.parallel body.
  new_body->getOperations().splice(new_body->begin(), old_body->getOperations(),
                                   old_body->begin(),
                                   std::prev(old_body->end()));

  // RAUW the original IV with the new parallel's first block argument.
  mlir::Value new_iv = parallel.getInductionVar(0);
  old_iv.replaceAllUsesWith(new_iv);

  // Append ktdf.parallel_yield as the new body's terminator.
  builder.setInsertionPointToEnd(new_body);
  mlir::ktdf::ParallelYieldOp::create(builder, loc);

  // Erase the original scf.for. Its body is now empty (only the orphaned
  // scf.yield remains, which is erased with the parent).
  loop.erase();

#ifndef NDEBUG
  if (mlir::failed(parallel.verify())) {
    llvm::report_fatal_error("ktdf.parallel verifier failed after rewrite");
  }
#endif
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

struct ParallelizeLoopsAcrossInstancesPass
    : public impl::ParallelizeLoopsAcrossInstancesPassBase<
          ParallelizeLoopsAcrossInstancesPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    // Pre-order walk over pipelines. Collect candidates first, then rewrite,
    // so the walk's iterator is not invalidated by op erasure.
    llvm::SmallVector<Candidate> candidates;
    module_op.walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          if (auto c = findCandidate(pipeline, resource_kinds)) {
            candidates.push_back(*c);
          }
        });

    for (const Candidate& c : candidates) {
      rewrite(c);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createParallelizeLoopsAcrossInstancesPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<ParallelizeLoopsAcrossInstancesPass>();
}

std::unique_ptr<mlir::Pass>
scheduler::createParallelizeLoopsAcrossInstancesPass() {
  return std::make_unique<ParallelizeLoopsAcrossInstancesPass>();
}
