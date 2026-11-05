//===-- StageCoarsening.cpp -------------------------------------*- c++ -*-===//
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
// Stage Coarsening Transformation
//
// This pass implements a stage-coarsening algorithm using the following steps:
//
// 1: Identify stage groups using loop distribution legality rules.
// 2: Construct a PipelineTree to represent the currect structure of the code
// and transform it to the desired structure (this step does not modify IR)
//   a. Distribute and sink loops into pipeline.
//   b. Further sink into stage groups containing a single stage (expand
//      buffers).
//   c. Discover and correct illegal structures iteratively.
//   d. Correct stage dependencies that cross pipeline boundaries.
// 3: Use the manipulated PipelineTree abstraction with reference to
//    operations in the input IR to generate the transformed IR.
//
// Note: Legality of stage coarsening follows loop distribution rules.
//       Data dependence analysis may be added in future.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGrouping.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/BufferExpansion.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/Materializer.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/ScopeCorrection.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Utils/PipelineTreeLegalizer.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "stage-coarsening"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Stage Coarsening pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_STAGECOARSENINGPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

//===----------------------------------------------------------------------===//
// Stage Coarsening Pass
//===----------------------------------------------------------------------===//

/// Module pass that implements the algorithm for creating hierarchical
/// pipelines
struct StageCoarseningPass
    : public ktdf::impl::StageCoarseningPassBase<StageCoarseningPass> {
  using StageCoarseningPassBase<StageCoarseningPass>::StageCoarseningPassBase;

  void runOnOperation() override;

 private:
  /// Represents a candidate for stage coarsening transformation
  struct TransformCandidate {
    llvm::SmallVector<scf::ForOp>
        loop_nest_;  // Perfectly nested loops to sink (outermost to innermost)
    ktdf::PipelineOp pipeline_;    // Target pipeline
    Operation* loop_nest_parent_;  // Parent operation of the outermost
                                   // loop (will be tree root)
  };

  /// Identify all transform candidates in the module
  /// Walks IR to find pipelines with 'n' perfectly nested loops above them
  void identifyTransformCandidates(
      ModuleOp module_op, llvm::SmallVector<TransformCandidate>& candidates);

  /// Apply the transformation algorithm to a candidate.
  /// Returns failure() if the transformation could not be applied; the caller
  /// is responsible for calling signalPassFailure().
  LogicalResult applyTransformation(TransformCandidate& candidate);

  //===--------------------------------------------------------------------===//
  // Distribute and Sink Loops into Pipeline
  //===--------------------------------------------------------------------===//

  void DistributeAndSinkIntoPipeline(
      TransformCandidate& candidate, scheduler::PipelineTree& tree,
      const ktdf::StageGroupingAnalysis& grouping,
      llvm::SmallVectorImpl<std::unique_ptr<ktdf::BufferExpansionInfo>>&
          expansion_infos);

  /// Perform buffer expansion analysis and record expansion information

  //===--------------------------------------------------------------------===//
  // Sink into Stage Groups Containing a Single Stage
  //===--------------------------------------------------------------------===//

  void SinkIntoSingleStage(TransformCandidate& candidate,
                           scheduler::PipelineTree& tree,
                           const ktdf::StageGroupingAnalysis& grouping);

  //===--------------------------------------------------------------------===//
  // Discover and Correct Illegal Structures
  //===--------------------------------------------------------------------===//

  void DiscoverAndCorrectStructure(TransformCandidate& candidate,
                                   scheduler::PipelineTree& tree);

  //===--------------------------------------------------------------------===//
  // Correct Stage Dependencies Crossing Pipeline Boundaries
  //===--------------------------------------------------------------------===//

  void CorrectStageDependencies(TransformCandidate& candidate,
                                scheduler::PipelineTree& tree);

  //===--------------------------------------------------------------------===//
  // Step 3: Generate Transformed IR from PipelineTree
  //===--------------------------------------------------------------------===//

  /// Generate the transformed IR from the manipulated PipelineTree.
  /// Returns failure() if IR generation failed; the caller is responsible for
  /// calling signalPassFailure().
  LogicalResult GenerateTransformedIR(
      TransformCandidate& candidate, scheduler::PipelineTree& tree,
      llvm::ArrayRef<const ktdf::BufferExpansionInfo*> expansion_infos);
};

//===----------------------------------------------------------------------===//
// StageCoarseningPass Implementation
//===----------------------------------------------------------------------===//

void StageCoarseningPass::runOnOperation() {
  if (DisableThisPass) return;
  LDBG(1) << "========= " PASS_NAME " =========";
  auto module_op = getOperation();

  LDBG(1) << "Loop nest depth: " << loopNestDepth;

  // Identify all transform candidates
  llvm::SmallVector<TransformCandidate> candidates;
  identifyTransformCandidates(module_op, candidates);

  if (candidates.empty()) {
    LDBG(1) << "No transform candidates found.";
    return;
  }

  LDBG(1) << "Found " << candidates.size() << " transform candidate(s)\n";

  // Apply transformation to each candidate. Stop and signal failure on the
  // first error — continuing would operate on inconsistent IR.
  for (TransformCandidate& candidate : candidates) {
    if (failed(applyTransformation(candidate))) {
      signalPassFailure();
      return;
    }
  }

  LDBG(1) << "=== Stage Coarsening Pass Complete ===\n";
}

void StageCoarseningPass::identifyTransformCandidates(
    ModuleOp module_op, llvm::SmallVector<TransformCandidate>& candidates) {
  // Walk all functions in the module
  module_op.walk([&](func::FuncOp func_op) {
    LDBG(1) << "Analyzing function: " << func_op.getName() << "";

    // Find all pipelines in this function
    func_op.walk([&](ktdf::PipelineOp pipeline) {
      LDBG(1) << "  Found pipeline, checking for loop nest above it...";

      // Walk up the parent chain to find perfectly nested loops
      Operation* current_op = pipeline.getOperation();
      llvm::SmallVector<scf::ForOp> loop_nest;

      // Collect up to 'loopNestDepth' perfectly nested loops
      for (unsigned i = 0; i < loopNestDepth; ++i) {
        Operation* parent_op = current_op->getParentOp();
        if (!parent_op) break;
        auto for_op = dyn_cast<scf::ForOp>(parent_op);
        if (!for_op) break;
        auto& body_ops = for_op.getBody()->getOperations();

        // Check if this is a perfectly nested loop
        // Allow only specific simple operations between nested loops
        bool is_perfectly_nested = true;
        int num_sibling_loops = 0;
        for (auto& op : body_ops) {
          // Skip the terminator
          if (isa<scf::YieldOp>(op)) continue;

          // In the first iteration we may encounter the pipeline we started
          // from, so just ignore it.
          if (&op == pipeline.getOperation()) continue;

          // Check if operation is a nested loop. We allow a nested loop as long
          // as it is the only child of its parent.
          if (auto inner_for = dyn_cast<scf::ForOp>(op)) {
            num_sibling_loops++;
            continue;
          }

          // Check if operation is allowed between loops using the proper
          // function
          if (!ktdf::StageCoarseningMaterializer::isOpAllowedBetweenLoops(
                  &op)) {
            is_perfectly_nested = false;
            break;
          }
        }

        if (is_perfectly_nested && num_sibling_loops <= 1) {
          loop_nest.push_back(for_op);
          current_op = for_op.getOperation();
          continue;
        }

        // Not perfectly nested, stop looking for more loops
        break;
      }

      // If we found any loops, this is a candidate
      if (!loop_nest.empty()) {
        // Reverse to get outermost-to-innermost order
        std::reverse(loop_nest.begin(), loop_nest.end());

        // Parent of outermost loop becomes tree root
        Operation* loop_nest_parent =
            loop_nest.front().getOperation()->getParentOp();

        candidates.push_back({loop_nest, pipeline, loop_nest_parent});

        LLVM_DEBUG({
          llvm::dbgs() << "    -> Found candidate with " << loop_nest.size()
                       << " nested loop(s) at ";
          scheduler::printLocation(llvm::dbgs(),
                                   loop_nest.front().getOperation());
          llvm::dbgs() << "\n";
        });
      }
    });
  });
}

LogicalResult StageCoarseningPass::applyTransformation(
    TransformCandidate& candidate) {
  LDBG(1) << "--- Applying Transformation in steps ---";
  LDBG(1) << "Loop nest depth: " << candidate.loop_nest_.size();

  // Get device and create memory tree
  auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
  auto* const device = device_manager.getOrImportDevice();
  if (!device) {
    return candidate.pipeline_->emitError(
        "Unable to import device specification for stage grouping. This could "
        "happen if the device spec file is empty or contains multiple "
        "devices");
  }

  auto& memory_tree =
      device_manager.getOrCreateView<scheduler::arch_view::MemoryTree>(*device);

  // Build pipeline tree once - it will be manipulated throughout the
  // transformation
  scheduler::PipelineTree tree;
  tree.compute(*candidate.loop_nest_parent_);
  LLVM_DEBUG({
    llvm::dbgs() << "=== Initial Pipeline Tree ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Perform stage grouping analysis once
  LDBG(1) << "\n  Performing stage grouping analysis...";
  ktdf::StageGroupingAnalysis grouping(candidate.pipeline_, memory_tree);
  LLVM_DEBUG(grouping.print(llvm::dbgs()));

  // Step 2a: Distribute and sink loops into pipeline.
  // unique_ptr elements guarantee all BufferExpansionInfo objects are freed on
  // every exit path.
  LDBG(1) << "\n--- Step 2a: Distribute and Sink into Pipeline ---";
  llvm::SmallVector<std::unique_ptr<ktdf::BufferExpansionInfo>>
      buffer_expansion_infos;
  DistributeAndSinkIntoPipeline(candidate, tree, grouping,
                                buffer_expansion_infos);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2a ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2b: Further sink into stage groups containing a single stage
  // (expand buffers)
  LDBG(1) << "\n--- Step 2b: Sink into Single-Stage Groups ---";
  SinkIntoSingleStage(candidate, tree, grouping);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2b ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2c: Discover and correct illegal structures iteratively
  LDBG(1) << "\n--- Step 2c: Discover and Correct Illegal Structures ---";
  DiscoverAndCorrectStructure(candidate, tree);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2c ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2d: Correct stage dependencies that cross pipeline boundaries
  LDBG(1) << "\n--- Step 2d: Correct Stage Dependencies ---";
  CorrectStageDependencies(candidate, tree);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2d (Final) ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 3: Generate transformed IR from the manipulated tree.
  // Collect raw const pointers for the read-only ArrayRef interface;
  // unique_ptrs in buffer_expansion_infos retain ownership and handle cleanup.
  llvm::SmallVector<const ktdf::BufferExpansionInfo*> raw_infos;
  raw_infos.reserve(buffer_expansion_infos.size());
  for (const auto& info : buffer_expansion_infos)
    raw_infos.push_back(info.get());

  return GenerateTransformedIR(candidate, tree, raw_infos);
}

void StageCoarseningPass::DistributeAndSinkIntoPipeline(
    TransformCandidate& candidate, scheduler::PipelineTree& tree,
    const ktdf::StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<std::unique_ptr<ktdf::BufferExpansionInfo>>&
        expansion_infos) {
  LDBG(1) << "Distributing loops into pipeline...";

  // Validate candidate
  if (!candidate.pipeline_ || !candidate.loop_nest_parent_) {
    LDBG(1) << "  ERROR: Invalid candidate (null pipeline or parent)";
    return;
  }

  // For each stage group, create unmaterialized loop nest and insert
  // as children of pipeline
  LDBG(1) << "\n  Distributing loop nests to stage groups...";

  // Process each group
  const llvm::SmallVector<ktdf::StageGroup>& groups = grouping.getGroups();
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const ktdf::StageGroup& group = groups[group_idx];
    const llvm::SmallVector<ktdf::StageOp>& stages = group.getStages();

    if (stages.empty()) continue;

    LDBG(1) << "  Group " << group_idx << ": Creating loop nest for "
            << stages.size() << " stage(s)";

    // Find stage nodes for this group using the operation-to-node mapping
    llvm::SmallVector<scheduler::StageNode*> stage_nodes;
    scheduler::PipelineTreeNode* pipeline_parent_node =
        tree.getNodeForOp(candidate.pipeline_.getOperation());
    assert(pipeline_parent_node && pipeline_parent_node->isPipelineNode());

    for (ktdf::StageOp stage : stages) {
      scheduler::PipelineTreeNode* node = tree.getNodeForOp(stage);
      assert(node);
      assert(node->isStageNode());
      auto* stage_node = static_cast<scheduler::StageNode*>(node);
      stage_nodes.push_back(stage_node);
    }
    assert(!stage_nodes.empty());

    // Create nested loop nodes (outermost to innermost)
    // The outermost loop will be inserted as a child of the pipeline
    // Each subsequent loop will be a child of the previous loop
    // The stages will be moved to be children of the innermost loop
    scheduler::OperationTreeNode* current_parent = pipeline_parent_node;

    for (size_t loop_idx = 0; loop_idx < candidate.loop_nest_.size();
         ++loop_idx) {
      // Create unmaterialized loop node using factory method
      scheduler::LoopNode* loop_node =
          tree.createLoopNode(nullptr);  // Unmaterialized
      loop_node->setTemplateOp(candidate.loop_nest_[loop_idx].getOperation());

      // Insert loop node as child of current parent (pipeline or outer loop)
      current_parent->insertChildNode(loop_node);

      // This loop becomes the parent for the next loop or the stages
      current_parent = loop_node;
    }

    // Move all stages in this group to be children of the innermost loop
    for (scheduler::StageNode* stage_node : stage_nodes) {
      stage_node->unlink();
      current_parent->insertChildNode(stage_node);
    }

    LDBG(1) << "    Moved " << stage_nodes.size()
            << " stage(s) under loop nest";
  }

  // Now remove the original loops from the tree
  // Note that references to their operations still exist as template ops in
  // the unmaterialized nodes
  for (scf::ForOp loop_op : candidate.loop_nest_) {
    scheduler::PipelineTreeNode* loop_node =
        tree.getNodeForOp(loop_op.getOperation());
    assert(loop_node);
    assert(loop_node->isLoopNode() && "Expected loop node");
    // Unlink only the node itself, not its subtree
    // This moves the loop's children (the pipeline) up to the loop's parent
    loop_node->unlinkNode();
    // Delete the now-unlinked loop node
    tree.deleteNode(loop_node);
  }

  ktdf::PerformBufferExpansionAnalysis(
      candidate.loop_nest_, candidate.pipeline_, grouping, expansion_infos);

  LDBG(1) << "\n  Loop distribution complete.";
}

void StageCoarseningPass::SinkIntoSingleStage(
    TransformCandidate& candidate, scheduler::PipelineTree& tree,
    const ktdf::StageGroupingAnalysis& grouping) {
  LDBG(1) << "Sinking into single-stage groups...";

  const llvm::SmallVector<ktdf::StageGroup>& groups = grouping.getGroups();

  // Find the pipeline node in the tree
  scheduler::PipelineTreeNode* pipeline_node =
      tree.getNodeForOp(candidate.pipeline_.getOperation());
  assert(pipeline_node && pipeline_node->isPipelineNode());

  // Process each stage group
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const ktdf::StageGroup& group = groups[group_idx];

    // Only process single-stage groups
    if (!group.isSingleStage()) {
      LDBG(1) << "  Group " << group_idx << ": Skipping (contains "
              << group.getStages().size() << " stages)";
      continue;
    }

    ktdf::StageOp stage = group.getFirstStage();
    LLVM_DEBUG({
      llvm::dbgs() << "  Group " << group_idx
                   << ": Processing single-stage group with stage at ";
      scheduler::printLocation(llvm::dbgs(), stage.getOperation());
      llvm::dbgs() << "\n";
    });

    // Find the stage node in the tree
    scheduler::PipelineTreeNode* stage_node =
        tree.getNodeForOp(stage.getOperation());
    assert(stage_node && stage_node->isStageNode());

    // The stage's parent should now be the innermost loop from the candidate
    scheduler::OperationTreeNode* innermost_loop = stage_node->getParentNode();
    assert(innermost_loop && "Stage should have a parent by now");

    scheduler::PipelineTreeNode* innermost_loop_pipeline =
        static_cast<scheduler::PipelineTreeNode*>(innermost_loop);
    assert(innermost_loop_pipeline->isLoopNode() &&
           "Stage's parent should be a loop node now");

    // Find the outermost loop in the nest by walking up to the pipeline
    const scheduler::PipelineTreeNode* outermost_loop =
        scheduler::PipelineTreeNode::findOutermostLoop(
            innermost_loop_pipeline,
            [](const scheduler::PipelineTreeNode* node) {
              return node->isPipelineNode();
            });
    assert(outermost_loop && "Expected to find an outermost loop");

    LDBG(1) << "    Sinking loop nest into stage";

    // Sink the loop nest into the stage:
    scheduler::PipelineTreeNode* mutable_outermost =
        const_cast<scheduler::PipelineTreeNode*>(outermost_loop);
    assert(mutable_outermost->getParentNode() == pipeline_node);
    scheduler::OperationTreeNode* insert = mutable_outermost->getPrevSibling();
    mutable_outermost->unlink();
    stage_node->unlink();
    stage_node->insertChildNode(mutable_outermost);
    pipeline_node->insertChildNode(stage_node, insert);

    LDBG(1) << "    Successfully sank loop nest into stage";
  }

  LDBG(1) << "  Sinking into single-stage groups complete.";
}

void StageCoarseningPass::DiscoverAndCorrectStructure(
    TransformCandidate& candidate, scheduler::PipelineTree& tree) {
  LDBG(1) << "Discovering and correcting illegal structures...";

  scheduler::pipeline_tree::Legalizer legalizer;

  // Find and fix all violations iteratively
  legalizer.findAndFixViolations(tree);

  LDBG(1) << "  Structure legalization complete.";
}

void StageCoarseningPass::CorrectStageDependencies(
    TransformCandidate& candidate, scheduler::PipelineTree& tree) {
  LDBG(1) << "Correcting stage dependencies...";

  // Step 1: Build pipeline-to-depth map and collect all stages.
  // Recursive DFS passes depth down so siblings always get the same value.
  llvm::DenseMap<const scheduler::PipelineNode*, unsigned> pipeline_to_depth;
  llvm::SmallVector<scheduler::StageNode*> all_stages;

  std::function<void(scheduler::OperationTreeNode*, unsigned)> buildDepthMap =
      [&](scheduler::OperationTreeNode* node, unsigned depth) {
        auto* pnode = static_cast<scheduler::PipelineTreeNode*>(node);
        if (pnode->isPipelineNode()) {
          pipeline_to_depth[static_cast<const scheduler::PipelineNode*>(
              pnode)] = depth;
          ++depth;
        } else if (pnode->isStageNode()) {
          all_stages.push_back(static_cast<scheduler::StageNode*>(pnode));
        }
        for (auto* child = node->getFirstChild(); child;
             child = child->getNextSibling())
          buildDepthMap(child, depth);
      };
  buildDepthMap(tree.getRoot(), 0);

  // Helper lambda to find a stage at the same level as the target pipeline by
  // walking up from start_stage
  auto findStageAtPipeline =
      [&](scheduler::StageNode* start_stage,
          scheduler::PipelineNode* target_pipeline) -> scheduler::StageNode* {
    // Find the stage whose containing pipeline is at the target depth
    scheduler::OperationTreeNode* result_stage =
        start_stage->findParent([&](const scheduler::OperationTreeNode* node) {
          if (!static_cast<const scheduler::PipelineTreeNode*>(node)
                   ->isStageNode()) {
            return false;
          }
          const scheduler::StageNode* stage =
              static_cast<const scheduler::StageNode*>(node);
          const scheduler::PipelineTreeNode* parent =
              static_cast<const scheduler::PipelineTreeNode*>(
                  stage->getParentNode());
          assert(parent->isPipelineNode() && "Stage parent must be a pipeline");
          const scheduler::PipelineNode* stage_pipeline =
              static_cast<const scheduler::PipelineNode*>(parent);
          return stage_pipeline == target_pipeline;
        });

    return static_cast<scheduler::StageNode*>(result_stage);
  };

  // Step 2: Process each stage and correct boundary-crossing dependencies
  for (scheduler::StageNode* source_stage : all_stages) {
    scheduler::PipelineTreeNode* source_parent =
        static_cast<scheduler::PipelineTreeNode*>(
            source_stage->getParentNode());
    assert(source_parent->isPipelineNode() &&
           "Stage parent must be a pipeline");
    scheduler::PipelineNode* source_pipeline =
        static_cast<scheduler::PipelineNode*>(source_parent);
    unsigned source_depth = pipeline_to_depth[source_pipeline];

    for (scheduler::StageNode* sink_stage : source_stage->getDependencies()) {
      scheduler::PipelineTreeNode* sink_parent =
          static_cast<scheduler::PipelineTreeNode*>(
              sink_stage->getParentNode());
      assert(sink_parent->isPipelineNode() &&
             "Stage parent must be a pipeline");
      scheduler::PipelineNode* sink_pipeline =
          static_cast<scheduler::PipelineNode*>(sink_parent);
      unsigned sink_depth = pipeline_to_depth[sink_pipeline];

      // If source and sink are in the same pipeline, nothing to do
      if (source_pipeline == sink_pipeline) continue;

      // If source is at lower depth than sink (outer -> inner)
      if (source_depth < sink_depth) {
        LDBG(1) << "    Correcting outer->inner dependency: Stage "
                << source_stage->getStageId() << " (depth " << source_depth
                << ") -> Stage " << sink_stage->getStageId() << " (depth "
                << sink_depth << ")";

        // Walk up from sink to find stage at source's depth
        scheduler::StageNode* new_sink =
            findStageAtPipeline(sink_stage, source_pipeline);
        if (new_sink) {
          LDBG(1) << "      Redirected to Stage " << new_sink->getStageId()
                  << "";
          source_stage->replaceDependency(sink_stage, new_sink);
        }
      }
      // If source is at higher depth than sink (inner -> outer)
      else if (source_depth > sink_depth) {
        LDBG(1) << "    Correcting inner->outer dependency: Stage "
                << source_stage->getStageId() << " (depth " << source_depth
                << ") -> Stage " << sink_stage->getStageId() << " (depth "
                << sink_depth << ")";

        // Walk up from source to find stage at sink's depth
        scheduler::StageNode* new_source =
            findStageAtPipeline(source_stage, sink_pipeline);
        if (new_source) {
          LDBG(1) << "      Moved dependency to Stage "
                  << new_source->getStageId() << "";

          // Add dependency to new source (if not already present)
          if (!llvm::is_contained(new_source->getDependencies(), sink_stage)) {
            new_source->addDependency(sink_stage);
          }

          // Mark dependency for removal by nullifying it
          source_stage->nullifyDependency(sink_stage);
        }
      }
    }

    // Remove all nullified dependencies
    source_stage->removeNullifiedDependencies();
  }

  LDBG(1) << "  Stage dependency correction complete.";
}

//===----------------------------------------------------------------------===//
// Step 3: Generate Transformed IR from PipelineTree
//===----------------------------------------------------------------------===//

LogicalResult StageCoarseningPass::GenerateTransformedIR(
    TransformCandidate& candidate, scheduler::PipelineTree& tree,
    llvm::ArrayRef<const ktdf::BufferExpansionInfo*> expansion_infos) {
  LDBG(1) << "\n--- Step 3: Generate Transformed IR ---";

  // Get the outer pipeline node from the tree
  scheduler::PipelineTreeNode* outer_pipeline =
      tree.getNodeForOp(candidate.pipeline_.getOperation());
  assert(outer_pipeline);
  assert(outer_pipeline->isPipelineNode());

  // Create a builder at the location of the root (ie right before outermost
  // loop in the original candidate loop nest)
  Operation* root = candidate.loop_nest_.front();
  OpBuilder builder(root);

  // Create an IRMapping to track value mappings during cloning
  IRMapping value_map;

  // Create buffer cloner if we have buffers to expand
  ktdf::BufferCloner* buffer_cloner = nullptr;
  std::optional<ktdf::StageCoarseningBufferCloner> cloner_storage;
  if (!expansion_infos.empty()) {
    LDBG(1) << "  Creating buffer cloner for " << expansion_infos.size()
            << " buffer(s)";
    cloner_storage.emplace(expansion_infos);
    buffer_cloner = &cloner_storage.value();
  }

  // Use the stage-coarsening materializer to generate IR from the tree
  ktdf::StageCoarseningMaterializer materializer(builder, value_map,
                                                 buffer_cloner);
  Operation* new_outer_pipeline = materializer.materialize(*outer_pipeline);

  // Erase the original candidate loop nest and all the old ops under it
  root->erase();

  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    os << "IR after materialization:\n";
    new_outer_pipeline->print(os);
  });

  auto new_outer_pipeline_op = dyn_cast<ktdf::PipelineOp>(new_outer_pipeline);
  if (!new_outer_pipeline_op) {
    return new_outer_pipeline->emitError(
        "materialized op is not a ktdf.pipeline");
  }

  LDBG(1) << "\n--- Cleaning Up Private Ops ---";
  mlir::IRRewriter rewriter(new_outer_pipeline_op);
  mlir::ktdf::PrivateBuilder::canonicalize(rewriter, new_outer_pipeline_op);

  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    os << "IR after private cleanup:\n";
    new_outer_pipeline->print(os);
  });

  // Fix scoping of private results that are only used in inner pipelines
  LDBG(1) << "\n--- Fixing Private Result Scoping ---";
  ktdf::ScopeCorrection scope_corrector(builder, new_outer_pipeline);
  scope_corrector.run();

  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    os << "IR after private scoping correction:\n";
    new_outer_pipeline->print(os);
  });

  LDBG(1) << "  IR generation complete.";
  return success();
}

}  // namespace

auto mlir::ktdf::createStageCoarseningPass() -> std::unique_ptr<Pass> {
  return std::make_unique<StageCoarseningPass>();
}
