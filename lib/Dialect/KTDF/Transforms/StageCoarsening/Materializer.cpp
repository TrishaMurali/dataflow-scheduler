//===-- Materializer.cpp ----------------------------------------*- c++ -*-===//
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
// Stage Coarsening Pipeline Tree Materializer Implementation
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/Materializer.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "stage-coarsening-materializer"

using namespace mlir;
using namespace mlir::ktdf;

Operation* StageCoarseningMaterializer::materialize(
    const scheduler::PipelineTreeNode& root_node) {
  // Clear any previous materialization state
  node_to_materialized_op_.clear();

  // Materialize the root and all its children
  materializeTreeNode(root_node);

  // Return the materialized operation for the root
  return getMaterializedOp(root_node);
}

Operation* StageCoarseningMaterializer::getMaterializedOp(
    const scheduler::PipelineTreeNode& node) const {
  auto it = node_to_materialized_op_.find(&node);
  return it != node_to_materialized_op_.end() ? it->second : nullptr;
}

void StageCoarseningMaterializer::materializeTreeNode(
    const scheduler::PipelineTreeNode& node) {
  // Handle different node types
  if (node.isLoopNode()) {
    const auto& loop_node = static_cast<const scheduler::LoopNode&>(node);
    materializeLoopNode(loop_node);
  } else if (node.isStageNode()) {
    const auto& stage_node = static_cast<const scheduler::StageNode&>(node);
    materializeStageNode(stage_node);
  } else if (node.isPipelineNode()) {
    const auto& pipeline_node =
        static_cast<const scheduler::PipelineNode&>(node);
    materializePipelineNode(pipeline_node);
  } else if (node.isPrivateNode()) {
    const auto& private_node = static_cast<const scheduler::PrivateNode&>(node);
    materializePrivateNode(private_node);
  } else {
    llvm_unreachable("unimplemented node type");
  }
}

void StageCoarseningMaterializer::materializeChildren(
    const scheduler::PipelineTreeNode& parent_node) {
  scheduler::OperationTreeNode* child = parent_node.getFirstChild();
  while (child) {
    materializeTreeNode(
        static_cast<const scheduler::PipelineTreeNode&>(*child));
    child = child->getNextSibling();
  }
}

void StageCoarseningMaterializer::cloneStageBody(StageOp orig_stage) {
  Block* orig_body = orig_stage.getBody();
  for (Operation& op : *orig_body) {
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    // Check if this is a DataTransferOp or IndDataTransferOp and we have a
    // clone handler
    if (auto transfer_op = dyn_cast<ktdf::DataTransferOp>(&op)) {
      if (buffer_cloner_) {
        Operation* new_op = buffer_cloner_->cloneDataTransfer(
            transfer_op, builder_, value_map_);
        if (new_op) {
          // Cloner created a replacement operation, skip default cloning
          continue;
        }
      }
    } else if (auto ind_transfer_op = dyn_cast<ktdf::IndDataTransferOp>(&op)) {
      if (buffer_cloner_) {
        Operation* new_op = buffer_cloner_->cloneIndDataTransfer(
            ind_transfer_op, builder_, value_map_);
        if (new_op) {
          // Cloner created a replacement operation, skip default cloning
          continue;
        }
      }
    }

    // Default: clone the operation
    builder_.clone(op, value_map_);
  }
}

/// Check if an operation should be cloned when materializing loop bounds.
/// Only allows specific operations that are safe to clone between loops:
bool StageCoarseningMaterializer::isOpAllowedBetweenLoops(Operation* op) {
  if (isa<ktdf::TilingDeriveSizeOp>(op)) return true;
  if (isa<ktdf::TilingLinearizeIndexOp>(op)) return true;
  return false;
}

/// Check if an operation is in the same block as the template operation.
/// If it is, we don't need to clone it as it's already available.
bool StageCoarseningMaterializer::isInSameBlock(Operation* op,
                                                Operation* template_op) {
  if (!op || !template_op) {
    return false;
  }

  return op->getBlock() == template_op->getBlock();
}

/// Helper function to materialize a value by cloning its defining operation
/// if it's not already in the value map. This handles cases where loop bounds
/// are computed by arith or affine operations between nested loops.
Value StageCoarseningMaterializer::materializeValue(Value value,
                                                    Operation* template_op) {
  // First check if the value is already mapped
  Value mapped = value_map_.lookupOrDefault(value);
  if (mapped != value)
    return mapped;  // Already mapped, return the mapped value

  // If the value is a block argument, we can't materialize it further
  if (isa<BlockArgument>(value)) return value;

  // Get the defining operation
  Operation* def_op = value.getDefiningOp();
  assert(def_op && "Value must have a defining operation");

  // If the value is a constant-like operation, we assume it's in an ancestor
  // block that dominates the current insertion point.
  if (def_op->hasTrait<OpTrait::ConstantLike>()) return value;

  // Check if the operation is in the same block as the template operation
  // If it is not, we assume it to be in an ancestor block that dominates the
  // current insertion point.
  if (!isInSameBlock(def_op, template_op)) return value;

  assert(isOpAllowedBetweenLoops(def_op) &&
         "only simple arithmetic ops created by loop tiling are allowed in "
         "between loop nests");

  // Recursively materialize all operands first
  llvm::SmallVector<Value> materialized_operands;
  for (Value operand : def_op->getOperands()) {
    materialized_operands.push_back(materializeValue(operand, template_op));
  }

  // Clone the operation with materialized operands
  IRMapping local_map;
  for (auto [orig, mat] :
       llvm::zip(def_op->getOperands(), materialized_operands)) {
    local_map.map(orig, mat);
  }

  Operation* cloned_op = builder_.clone(*def_op, local_map);

  // These operations should have exactly one result
  assert(cloned_op->getNumResults() == 1 &&
         "Expected single-result operation for loop bound computation");

  Value cloned_result = cloned_op->getResult(0);
  value_map_.map(value, cloned_result);

  return cloned_result;
}

void StageCoarseningMaterializer::materializeInterveningOps(
    const scheduler::LoopNode& loop_node) {
  Operation* template_op = loop_node.getOperationOrTemplate();
  assert(template_op && "Loop node must have either operation or template");
  auto template_for = cast<scf::ForOp>(template_op);

  // Iterate through all operations in the template loop body
  // Clone any operations that pass the isOpAllowedBetweenLoops() check
  for (Operation& op : *template_for.getBody()) {
    if (!isOpAllowedBetweenLoops(&op)) continue;
    builder_.clone(op, value_map_);
  }
}

scf::ForOp StageCoarseningMaterializer::materializeLoopNode(
    const scheduler::LoopNode& loop_node) {
  // Get the template operation to copy loop bounds from
  Operation* template_op = loop_node.getOperationOrTemplate();

  assert(template_op && "Loop node must have either operation or template");
  auto template_for = cast<scf::ForOp>(template_op);

  // Materialize the loop bounds, cloning any defining operations if needed
  Value lower_bound =
      materializeValue(template_for.getLowerBound(), template_op);
  Value upper_bound =
      materializeValue(template_for.getUpperBound(), template_op);
  Value step = materializeValue(template_for.getStep(), template_op);

  // Create the new loop with empty body
  auto new_loop = scf::ForOp::create(builder_, template_for.getLoc(),
                                     lower_bound, upper_bound, step);

  // Copy attributes from template loop to new loop
  for (NamedAttribute attr : template_for->getAttrs()) {
    new_loop->setAttr(attr.getName(), attr.getValue());
  }

  // Map the induction variable
  value_map_.map(template_for.getInductionVar(), new_loop.getInductionVar());

  // Track the materialized operation
  node_to_materialized_op_[&loop_node] = new_loop.getOperation();

  // Set insertion point inside the loop body
  builder_.setInsertionPointToStart(new_loop.getBody());

  // Materialize any intervening operations that appear in the template loop
  // body before the first child (e.g., ktdf.tiling.derive_size,
  // ktdf.tiling.linearize_index)
  materializeInterveningOps(loop_node);

  // Materialize children
  materializeChildren(loop_node);
  builder_.setInsertionPointAfter(new_loop);

  LDBG(1) << "    Created loop with IV mapped";
  return new_loop;
}

ktdf::StageOp StageCoarseningMaterializer::materializeStageNode(
    const scheduler::StageNode& stage_node) {
  // Get the original stage operation (or template for unmaterialized stages)
  Operation* orig_stage_op = stage_node.getOperationOrTemplate();

  Location loc = builder_.getUnknownLoc();
  StageOp orig_stage;

  if (orig_stage_op) {
    orig_stage = cast<ktdf::StageOp>(orig_stage_op);
    loc = orig_stage.getLoc();
  }

  LDBG(1) << "    Materializing stage " << stage_node.getStageId();

  // Build depends_in and depends_out from stage dependencies
  llvm::SmallVector<Value> depends_in;
  llvm::SmallVector<Value> depends_out;

  // For depends_out: find the token produced by this stage
  if (!stage_node.getDependencies().empty()) {
    auto it = stage_to_token_map_.find(&stage_node);
    assert(it != stage_to_token_map_.end() &&
           "expected token map to be updated for this stage");
    depends_out.push_back(it->second);
    LDBG(1) << "      Added depends_out token for stage "
            << stage_node.getStageId();
  }

  // For depends_in: find stages that this stage depends on
  // We need to look at all stages in the parent pipeline and check if this
  // stage is in their list of dependents.
  const scheduler::PipelineTreeNode* parent =
      static_cast<const scheduler::PipelineTreeNode*>(
          stage_node.getParentNode());
  assert(parent && parent->isPipelineNode());
  const scheduler::PipelineNode* pipeline_parent =
      static_cast<const scheduler::PipelineNode*>(parent);
  for (scheduler::StageNode* other_stage : pipeline_parent->getStages()) {
    if (other_stage == &stage_node) continue;
    // Check if this stage is in other_stage's dependencies
    for (scheduler::StageNode* dep : other_stage->getDependencies()) {
      if (dep != &stage_node) continue;
      // This stage depends on other_stage (other_stage unblocks this stage)
      auto it = stage_to_token_map_.find(other_stage);
      assert(it != stage_to_token_map_.end());
      depends_in.push_back(it->second);
      LDBG(1) << "      Added depends_in token for this stage from stage "
              << other_stage->getStageId();
    }
  }

  // Create the stage with proper dependencies
  auto new_stage = StageOp::create(builder_, loc, depends_in, depends_out);

  // Carry over the applicable_units attribute from the original stage. It is
  // required downstream (e.g. ParallelizeLoopsAcrossInstancesPass asserts
  // every stage has it) and the rebuilt stage would otherwise lose it.
  // For coarsened stages that have no template (no orig_stage) but enclose
  // an inner pipeline, the union is computed below after the body is
  // materialized.
  if (orig_stage) {
    if (auto applicable_units = orig_stage.getApplicableUnitsAttr()) {
      new_stage.setApplicableUnitsAttr(applicable_units);
    }
  }

  // Track the materialized operation
  node_to_materialized_op_[&stage_node] = new_stage.getOperation();

  // Set insertion point inside the stage body
  Block* stage_body = new_stage.getBody();
  builder_.setInsertionPointToStart(stage_body);

  // Check if this stage has children (loops) or if we need to clone the body
  scheduler::OperationTreeNode* first_child = stage_node.getFirstChild();

  if (first_child) {
    // Stage has children (loops), materialize them first
    LDBG(1) << "    Stage has children, materializing them";
    materializeChildren(stage_node);

    // Now clone the stage body into the innermost loop
    if (orig_stage) {
      LDBG(1) << "    Cloning stage body into innermost loop";

      // Find the innermost loop node
      const scheduler::PipelineTreeNode* first_loop =
          static_cast<const scheduler::PipelineTreeNode*>(first_child);
      assert(first_loop->isLoopNode() && "First child should be a loop");
      const scheduler::PipelineTreeNode* innermost_loop_node =
          scheduler::PipelineTreeNode::findInnermostLoop(first_loop);
      assert(innermost_loop_node);

      // Get the materialized operation for the innermost loop
      Operation* innermost_op = node_to_materialized_op_[innermost_loop_node];
      assert(innermost_op && "Innermost loop should have been materialized");

      // Set insertion point to the innermost loop body and clone stage body
      auto loop_op = cast<scf::ForOp>(innermost_op);
      // Set insertion point before the terminator (last non-terminator
      // position)
      Block* loop_body = loop_op.getBody();
      auto terminator = loop_body->getTerminator();
      builder_.setInsertionPoint(terminator);
      cloneStageBody(orig_stage);
    }
  } else if (orig_stage) {
    // Stage has no children, clone the original stage body directly
    LDBG(1) << "    Cloning original stage body";
    cloneStageBody(orig_stage);
  } else {
    llvm_unreachable("Unmaterialized stage with no children and no template");
  }

  // If the new stage has no applicable_units yet (e.g. it was synthesized by
  // coarsening with no template) but its body encloses one or more
  // ktdf.stage ops, set its applicable_units to the union of those inner
  // stages' applicable_units. ParallelizeLoopsAcrossInstances requires every
  // stage to carry the attribute. Inner stages without the attribute are
  // skipped so legacy inputs still parse cleanly.
  if (!new_stage.getApplicableUnitsAttr()) {
    llvm::SmallSetVector<mlir::Attribute, 4> union_units;
    new_stage.walk([&](ktdf::StageOp inner_stage) {
      if (inner_stage == new_stage) return;
      if (auto attr = inner_stage.getApplicableUnitsAttr()) {
        for (mlir::Attribute unit : attr.getValue()) union_units.insert(unit);
      }
    });
    if (!union_units.empty()) {
      llvm::SmallVector<Attribute> units(union_units.begin(),
                                         union_units.end());
      new_stage.setApplicableUnitsAttr(builder_.getArrayAttr(units));
    }
  }

  // Restore insertion point after the stage
  builder_.setInsertionPointAfter(new_stage);

  LDBG(1) << "    Created stage " << stage_node.getStageId();
  return new_stage;
}

ktdf::PipelineOp StageCoarseningMaterializer::materializePipelineNode(
    const scheduler::PipelineNode& pipeline_node) {
  // Get the original or template pipeline operation
  Operation* template_op = pipeline_node.getOperationOrTemplate();

  assert(template_op && "Pipeline node must have either operation or template");
  auto template_pipeline = cast<ktdf::PipelineOp>(template_op);

  // Create a new pipeline operation
  auto new_pipeline = PipelineOp::create(builder_, template_pipeline.getLoc());

  // Track the materialized operation
  node_to_materialized_op_[&pipeline_node] = new_pipeline.getOperation();

  // Set insertion point inside the pipeline body and materialize children
  Block* pipeline_body = new_pipeline.getBody();
  builder_.setInsertionPointToStart(pipeline_body);
  materializeChildren(pipeline_node);
  builder_.setInsertionPointAfter(new_pipeline);

  LDBG(1) << "    Created pipeline";
  return new_pipeline;
}

void StageCoarseningMaterializer::materializePrivateNode(
    const scheduler::PrivateNode& private_node) {
  // FIXME: For now, we clone the entire private clause into the parent
  // pipeline. This makes all SSA values from the private clause accessible to
  // all stages. In the future, we should properly handle buffer privatization
  // for nested pipelines by moving relevant allocations to the nested
  // pipeline's private clause.

  LDBG(1) << "    Materializing private clause";

  // Get the parent pipeline node to access its stages for token mapping
  const scheduler::PipelineTreeNode* parent =
      static_cast<const scheduler::PipelineTreeNode*>(
          private_node.getParentNode());
  assert(parent && parent->isPipelineNode() &&
         "Private node must have a pipeline parent");
  const scheduler::PipelineNode* pipeline_parent =
      static_cast<const scheduler::PipelineNode*>(parent);
  llvm::SmallVector<scheduler::StageNode*> stages =
      pipeline_parent->getStages();

  // Track which tokens we need to create (only for stages with dependencies)
  llvm::SmallVector<Value> new_tokens;
  llvm::SmallVector<scheduler::StageNode*> stages_with_deps;

  // Identify stages that have dependencies (need tokens)
  for (scheduler::StageNode* stage : stages) {
    if (stage->getDependencies().empty()) continue;
    stages_with_deps.push_back(stage);
  }

  Operation* orig_private_op = private_node.getOperation();
  Operation* template_op = private_node.getTemplateOp();

  // Handle unmaterialized private node with no template - create from scratch
  if (!orig_private_op && !template_op) {
    materializePrivateNodeFromScratch(stages_with_deps, new_tokens);
    return;
  }

  // Handle materialized or template-based private node
  assert(orig_private_op && "Private node must have an operation or template");
  auto orig_private = cast<ktdf::PrivateOp>(orig_private_op);
  materializePrivateNodeFromTemplate(orig_private, stages_with_deps, new_tokens,
                                     private_node);
}

void StageCoarseningMaterializer::materializePrivateNodeFromScratch(
    const llvm::SmallVector<scheduler::StageNode*>& stages_with_deps,
    llvm::SmallVector<Value>& new_tokens) {
  LDBG(1) << "    Creating private node from scratch with only tokens";

  // Create result types (only tokens - one per stage with dependencies)
  llvm::SmallVector<Type> result_types;
  Type token_type = TokenType::get(builder_.getContext());
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    result_types.push_back(token_type);
  }

  // Create the private operation with known result types
  auto new_private =
      PrivateOp::create(builder_, builder_.getUnknownLoc(), result_types);

  // Get the empty block that was created with the private op
  Block& private_body = *new_private.getBody();

  // Insert token creation operations directly into the private body
  builder_.setInsertionPointToStart(&private_body);

  // Create tokens for stages with dependencies
  for (scheduler::StageNode* stage : stages_with_deps) {
    Value new_token =
        CreateTokenOp::create(builder_, builder_.getUnknownLoc()).getResult();
    new_tokens.push_back(new_token);

    LDBG(1) << "      Created new token for stage " << stage->getStageId();
  }

  // Create yield in the private body
  PrivateYieldOp::create(builder_, builder_.getUnknownLoc(), new_tokens);

  // Build the stage-to-token mapping (all results are tokens, starting at index
  // 0)
  buildStageToTokenMapping(stages_with_deps, new_private, 0);

  // Restore insertion point
  builder_.setInsertionPointAfter(new_private);
}

void StageCoarseningMaterializer::materializePrivateNodeFromTemplate(
    PrivateOp orig_private,
    const llvm::SmallVector<scheduler::StageNode*>& stages_with_deps,
    llvm::SmallVector<Value>& new_tokens,
    const scheduler::PrivateNode& private_node) {
  Block& orig_body = *orig_private.getBody();

  // Save the current insertion point (inside the new pipeline body)
  auto saved_insertion_point = builder_.saveInsertionPoint();

  // Create a temporary block to hold operations before splicing into private op
  Block tmp_block;
  builder_.setInsertionPointToStart(&tmp_block);

  for (Operation& op : orig_body.without_terminator()) {
    if (auto alloc_op = dyn_cast<memref::AllocOp>(&op)) {
      if (buffer_cloner_) {
        memref::AllocOp new_op =
            buffer_cloner_->cloneAllocOp(alloc_op, builder_, value_map_);
        if (new_op) {
          // Cloner created a replacement operation.
          // Update value_map_ to map old result to new result so that when we
          // replace the yield operands of the private region we use the correct
          // allocations.
          value_map_.map(alloc_op.getResult(), new_op.getResult());
          continue;
        }
      }
    }

    // Default: clone the operation
    builder_.clone(op, value_map_);
  }

  // Create new ktdf.create_token ops only for stages with dependencies.
  // A separate private-op cleanup step will remove any stale cloned token
  // resources.
  for (scheduler::StageNode* stage : stages_with_deps) {
    Value new_token =
        CreateTokenOp::create(builder_, orig_private.getLoc()).getResult();
    new_tokens.push_back(new_token);

    LDBG(1) << "      Created new token for stage " << stage->getStageId();
  }

  // Get the yield operation and determine result types from mapped operands.
  auto orig_yield = cast<ktdf::PrivateYieldOp>(orig_body.getTerminator());

  llvm::SmallVector<Type> result_types;
  llvm::SmallVector<Value> new_yield_operands;

  for (Value yielded_val : orig_yield.getOperands()) {
    Value mapped_val = value_map_.lookupOrDefault(yielded_val);
    new_yield_operands.push_back(mapped_val);
    result_types.push_back(mapped_val.getType());
  }

  size_t orig_result_count = new_yield_operands.size();

  // Append the new tokens to the yield operands.
  for (Value new_token : new_tokens) {
    new_yield_operands.push_back(new_token);
    result_types.push_back(new_token.getType());
  }

  // Restore insertion point to create the private operation in the right place
  builder_.restoreInsertionPoint(saved_insertion_point);

  // Now create the private operation with correct result types
  auto new_private =
      PrivateOp::create(builder_, orig_private.getLoc(), result_types);

  // Get the empty block that was created with the private op
  Block& private_body = *new_private.getBody();

  // Splice all operations from tmp_block into the private_body
  private_body.getOperations().splice(private_body.end(),
                                      tmp_block.getOperations());

  // Create yield in the private body
  builder_.setInsertionPointToEnd(&private_body);
  PrivateYieldOp::create(builder_, orig_yield.getLoc(), new_yield_operands);

  // Map original private results to the corresponding cloned private results.
  for (auto [idx, orig_result] : llvm::enumerate(orig_private.getResults())) {
    value_map_.map(orig_result, new_private.getResult(idx));
  }

  // Build the stage-to-token mapping (new tokens start after original results).
  buildStageToTokenMapping(stages_with_deps, new_private, orig_result_count);

  // Restore insertion point
  builder_.setInsertionPointAfter(new_private);

  // Track the materialized operation
  node_to_materialized_op_[&private_node] = new_private.getOperation();
}

void StageCoarseningMaterializer::buildStageToTokenMapping(
    const llvm::SmallVector<scheduler::StageNode*>& stages_with_deps,
    PrivateOp new_private, size_t non_token_result_count) {
  // Token results come after non-token results in the new private op
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    Value token_result = new_private.getResult(non_token_result_count + i);
    stage_to_token_map_[stages_with_deps[i]] = token_result;
    LDBG(1) << "      Mapped stage " << stages_with_deps[i]->getStageId()
            << " to token result " << (non_token_result_count + i);
  }
}

// Made with Bob
