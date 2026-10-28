//===-- SimplifyPipeline.cpp ------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/SimplifyPipeline.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"  // IWYU pragma: keep

#define PASS_NAME "simplify-pipeline"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Simplify Pipeline pass"),
    llvm::cl::init(false));

namespace mlir::ktdf {
#define GEN_PASS_DEF_SIMPLIFYPIPELINEPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

using namespace mlir::ktdf;

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

/// Finds the earliest stage in @p body that consumes from @p scc .
[[nodiscard]] auto findEarliestConsumer(mlir::Block* body,
                                        llvm::ArrayRef<StageOp> scc)
    -> StageOp {
  StageOp earliest;
  for (const StageGraph::Node member : scc) {
    for (const auto consumer : member.getConsumers()) {
      if (consumer->getBlock() != body || llvm::is_contained(scc, consumer)) {
        continue;
      }
      if (!earliest || consumer->isBeforeInBlock(earliest)) {
        earliest = consumer;
      }
    }
  }
  return earliest;
}

/// Moves the stages of @p scc in front of their earliest consumer, preserving
/// their relative order.
///
/// @return Whether any stage was moved.
auto moveInFrontOfConsumers(mlir::RewriterBase& rewriter, mlir::Block* body,
                            llvm::ArrayRef<StageOp> scc) -> bool {
  const auto earliest = findEarliestConsumer(body, scc);
  if (!earliest) {
    return false;
  }

  auto changed = false;
  for (auto& op : llvm::make_early_inc_range(
           llvm::make_range(std::next(earliest->getIterator()), body->end()))) {
    if (auto stage = llvm::dyn_cast<StageOp>(op);
        stage && llvm::is_contained(scc, stage)) {
      LDBG() << "moving " << mlir::OpWithFlags(stage, kSkipRegions);
      LDBG() << "  before " << mlir::OpWithFlags(earliest, kSkipRegions);
      rewriter.moveOpBefore(stage, earliest);
      changed = true;
    }
  }
  return changed;
}

/// Determines whether @p token is only used by the stages of @p pipeline .
[[nodiscard]] auto isLocalToken(mlir::Value token, PipelineOp pipeline)
    -> bool {
  return llvm::all_of(token.getUses(), [&](mlir::OpOperand& use) {
    auto stage = llvm::dyn_cast<StageOp>(use.getOwner());
    return stage && stage->getParentOp() == pipeline;
  });
}

/// Determines whether @p producer is the only stage outputting @p token .
[[nodiscard]] auto isUniqueTo(mlir::Value token, StageOp producer) -> bool {
  return llvm::all_of(token.getUses(), [&](mlir::OpOperand& use) {
    auto stage = llvm::cast<StageOp>(use.getOwner());
    return stage == producer || !stage.isOutDependency(use);
  });
}

/// Keeps the tokens of @p current that are foreign or in @p target , in order
/// and without repetition, and appends the remaining ones of @p target .
[[nodiscard]] auto mergeTokens(mlir::ValueRange current,
                               llvm::ArrayRef<mlir::Value> target,
                               PipelineOp pipeline)
    -> llvm::SmallVector<mlir::Value> {
  llvm::SetVector<mlir::Value> result;
  for (const auto token : current) {
    if (!isLocalToken(token, pipeline) || llvm::is_contained(target, token)) {
      result.insert(token);
    }
  }
  result.insert(target.begin(), target.end());
  return result.takeVector();
}

/// A stage and its consumers over local tokens.
struct LocalNode {
  unsigned index;
  llvm::SmallVector<LocalNode*> consumers;
};

}  // namespace

template <>
struct llvm::GraphTraits<LocalNode*> {
  using NodeRef = LocalNode*;
  using ChildIteratorType = llvm::SmallVector<LocalNode*>::iterator;

  [[nodiscard]] static auto getEntryNode(NodeRef node) -> NodeRef {
    return node;
  }
  [[nodiscard]] static auto child_begin(NodeRef node) -> ChildIteratorType {
    return node->consumers.begin();
  }
  [[nodiscard]] static auto child_end(NodeRef node) -> ChildIteratorType {
    return node->consumers.end();
  }
};

namespace {

/// The stages of a pipeline and their edges over local tokens.
class LocalStageGraph {
 public:
  explicit LocalStageGraph(PipelineOp pipeline) {
    for (const auto stage : pipeline.getStages()) {
      index_of_[stage] = nodes_.size();
      nodes_.push_back(LocalNode{static_cast<unsigned>(nodes_.size()), {}});
    }
    for (auto stage : pipeline.getStages()) {
      auto& node = nodes_[index_of_.at(stage)];
      for (auto& edge : llvm::make_range(ConsumerEdgeIterator::begin(stage),
                                         ConsumerEdgeIterator::end(stage))) {
        if (isLocalToken(edge.get(), pipeline)) {
          node.consumers.push_back(
              &nodes_[index_of_.at(llvm::cast<StageOp>(edge.getOwner()))]);
        }
      }
    }
  }

  LocalStageGraph(const LocalStageGraph&) = delete;
  auto operator=(const LocalStageGraph&) = delete;

  [[nodiscard]] auto getNodes() -> llvm::MutableArrayRef<LocalNode> {
    return nodes_;
  }
  [[nodiscard]] auto indexOf(StageOp stage) const -> unsigned {
    return index_of_.at(stage);
  }

 private:
  llvm::SmallVector<LocalNode> nodes_;
  llvm::DenseMap<StageOp, unsigned> index_of_;
};

/// Rewrites the local tokens of @p pipeline such that every stage with a kept
/// consumer outputs one token unique to it, which those consumers await.
///
/// @return Whether @p pipeline was modified.
auto rewriteTokens(mlir::RewriterBase& rewriter, PipelineOp pipeline,
                   llvm::function_ref<bool(StageOp, StageOp)> keep) -> bool {
  // Collect the producers of every consumer over the edges to keep.
  llvm::MapVector<StageOp, llvm::SetVector<StageOp>> producers;
  for (const auto stage : pipeline.getStages()) {
    for (auto& edge : llvm::make_range(ConsumerEdgeIterator::begin(stage),
                                       ConsumerEdgeIterator::end(stage))) {
      const auto consumer = llvm::cast<StageOp>(edge.getOwner());
      if (isLocalToken(edge.get(), pipeline) && keep(stage, consumer)) {
        producers[consumer].insert(stage);
      }
    }
  }

  // Reuse a token unique to each producer, or create one.
  std::optional<PrivateBuilder> private_builder;
  llvm::DenseMap<StageOp, mlir::Value> unique_out;
  for (const auto& entry : producers) {
    for (auto producer : entry.second) {
      if (unique_out.contains(producer)) {
        continue;
      }

      const auto outs = producer.getDependsOut();
      if (const auto it = llvm::find_if(outs,
                                        [&](mlir::Value token) {
                                          return isLocalToken(token,
                                                              pipeline) &&
                                                 isUniqueTo(token, producer);
                                        });
          it != outs.end()) {
        unique_out[producer] = *it;
        continue;
      }

      if (!private_builder) {
        private_builder.emplace(pipeline, std::nullopt, rewriter.getListener());
      }
      unique_out[producer] = private_builder->createToken(producer.getLoc());
    }
  }

  auto changed = false;
  for (auto stage : pipeline.getStages()) {
    llvm::SmallVector<mlir::Value> in_target;
    if (const auto it = producers.find(stage); it != producers.end()) {
      in_target = llvm::map_to_vector(it->second, [&](StageOp producer) {
        return unique_out.at(producer);
      });
    }
    llvm::SmallVector<mlir::Value> out_target;
    if (const auto it = unique_out.find(stage); it != unique_out.end()) {
      out_target.push_back(it->second);
    }

    const auto ins = mergeTokens(stage.getDependsIn(), in_target, pipeline);
    const auto outs = mergeTokens(stage.getDependsOut(), out_target, pipeline);
    if (llvm::equal(ins, stage.getDependsIn()) &&
        llvm::equal(outs, stage.getDependsOut())) {
      continue;
    }

    LDBG() << "rewriting tokens of " << mlir::OpWithFlags(stage, kSkipRegions);
    rewriter.modifyOpInPlace(stage, [&]() {
      stage.setDependsIn(ins);
      stage.setDependsOut(outs);
    });
    changed = true;
  }

  // Build the new tokens, and drop those left without users.
  private_builder.reset();
  if (changed) {
    PrivateBuilder::canonicalize(rewriter, pipeline);
  }
  return changed;
}

/// Moves stages in front of their consumers.
struct OrderStages : mlir::OpRewritePattern<PipelineOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(PipelineOp pipeline,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    return llvm::success(orderStages(rewriter, pipeline));
  }
};

/// Normalizes the dependency tokens of a pipeline.
struct NormalizeTokens : mlir::OpRewritePattern<PipelineOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(PipelineOp pipeline,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    return llvm::success(normalizeTokens(rewriter, pipeline));
  }
};

/// Reduces the dependency edges of a pipeline.
struct ReduceDependencies : mlir::OpRewritePattern<PipelineOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(PipelineOp pipeline,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    return llvm::success(reduceDependencies(rewriter, pipeline));
  }
};

struct SimplifyPipelinePass
    : impl::SimplifyPipelinePassBase<SimplifyPipelinePass> {
  using SimplifyPipelinePassBase::SimplifyPipelinePassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    SimplifyPipelinePassOptions options;
    options.reduceDependencies = this->reduceDependencies;
    populateSimplifyPipelinePatterns(patterns, options);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

auto mlir::ktdf::orderStages(mlir::RewriterBase& rewriter, PipelineOp pipeline)
    -> bool {
  auto* const body = pipeline.getBody();
  auto changed = false;
  if (const auto private_op = pipeline.getPrivateOp();
      private_op && &body->front() != private_op) {
    rewriter.moveOpBefore(private_op, &body->front());
    changed = true;
  }

  llvm::SetVector<StageOp> worklist(pipeline.getStages().begin(),
                                    pipeline.getStages().end());
  while (!worklist.empty()) {
    for (auto it = llvm::scc_begin(StageGraph::Node(worklist.front()));
         !it.isAtEnd(); ++it) {
      // SCCs reached from an earlier stage are already in place.
      if (!worklist.contains(it->front())) {
        continue;
      }

      changed |= moveInFrontOfConsumers(rewriter, body, *it);
      for (const auto stage : *it) {
        worklist.remove(stage);
      }
    }
  }
  return changed;
}

auto mlir::ktdf::normalizeTokens(mlir::RewriterBase& rewriter,
                                 PipelineOp pipeline) -> bool {
  return rewriteTokens(rewriter, pipeline,
                       [](StageOp, StageOp) { return true; });
}

auto mlir::ktdf::reduceDependencies(mlir::RewriterBase& rewriter,
                                    PipelineOp pipeline) -> bool {
  LocalStageGraph graph(pipeline);
  auto nodes = graph.getNodes();

  // Number the SCCs over local edges in reverse topological order. SCCs reached
  // from an earlier root were numbered as a whole, before their producers.
  constexpr auto kUnnumbered = ~0U;
  llvm::SmallVector<unsigned> scc_of(nodes.size(), kUnnumbered);
  unsigned num_sccs = 0;
  for (auto& root : nodes) {
    if (scc_of[root.index] != kUnnumbered) {
      continue;
    }
    for (auto it = llvm::scc_begin(&root); !it.isAtEnd(); ++it) {
      if (scc_of[it->front()->index] != kUnnumbered) {
        continue;
      }
      for (const auto* const node : *it) {
        scc_of[node->index] = num_sccs;
      }
      ++num_sccs;
    }
  }

  llvm::SmallVector<llvm::BitVector> successors(num_sccs,
                                                llvm::BitVector(num_sccs));
  for (const auto& node : nodes) {
    for (const auto* const consumer : node.consumers) {
      if (const auto from = scc_of[node.index], to = scc_of[consumer->index];
          from != to) {
        successors[from].set(to);
      }
    }
  }

  // An edge is implied if its target is reachable from another successor.
  llvm::SmallVector<llvm::BitVector> implied(num_sccs,
                                             llvm::BitVector(num_sccs));
  llvm::SmallVector<llvm::BitVector> reachable(num_sccs,
                                               llvm::BitVector(num_sccs));
  for (unsigned scc = 0; scc < num_sccs; ++scc) {
    for (const auto successor : successors[scc].set_bits()) {
      implied[scc] |= reachable[successor];
    }
    reachable[scc] = implied[scc];
    reachable[scc] |= successors[scc];
  }

  return rewriteTokens(rewriter, pipeline,
                       [&](StageOp producer, StageOp consumer) {
                         const auto from = scc_of[graph.indexOf(producer)];
                         const auto to = scc_of[graph.indexOf(consumer)];
                         return from == to || !implied[from].test(to);
                       });
}

void mlir::ktdf::populateSimplifyPipelinePatterns(
    mlir::RewritePatternSet& patterns,
    const SimplifyPipelinePassOptions& options) {
  patterns.add<OrderStages>(patterns.getContext());
  if (options.reduceDependencies) {
    patterns.add<ReduceDependencies>(patterns.getContext());
  } else {
    patterns.add<NormalizeTokens>(patterns.getContext());
  }
}
