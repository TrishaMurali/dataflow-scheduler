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
// Path Expansion Pipeline Tree Materializer Implementation
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/PathExpansion/Materializer.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "path-expansion-materializer"

using namespace scheduler;

namespace scheduler {

namespace {

/// Drop unused dims/symbols from `map` and propagate constant operands so
/// that the (map, indices) pair is well-formed: indices contains only the
/// values the simplified map references. Without this, an indices vector
/// inherited from the original transfer can carry stale loop IVs and
/// duplicate constants that don't appear in the printed form (the printer
/// honors the map) but show up as real operands and break invariance/SSA
/// analyses on downstream passes.
mlir::AffineMap canonicalizeMapAndIndices(
    mlir::AffineMap map, llvm::SmallVectorImpl<mlir::Value>& indices) {
  if (!map) return map;
  mlir::affine::canonicalizeMapAndOperands(&map, &indices);
  return map;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Main Materialization Entry Point
//===----------------------------------------------------------------------===//

mlir::Operation* PathExpansionMaterializer::materialize(
    const PipelineTreeNode& root_node) {
  // Clear any previous materialization state
  node_to_materialized_op_.clear();
  stage_to_token_map_.clear();

  // Materialize the root and all its children
  materializeTreeNode(root_node);

  // Return the materialized operation for the root
  return getMaterializedOp(root_node);
}

mlir::Operation* PathExpansionMaterializer::getMaterializedOp(
    const PipelineTreeNode& node) const {
  auto it = node_to_materialized_op_.find(&node);
  return it != node_to_materialized_op_.end() ? it->second : nullptr;
}

//===----------------------------------------------------------------------===//
// Tree Node Materialization Dispatcher
//===----------------------------------------------------------------------===//

void PathExpansionMaterializer::materializeTreeNode(
    const PipelineTreeNode& node) {
  // Handle different node types
  if (node.isLoopNode()) {
    const LoopNode& loop_node = static_cast<const LoopNode&>(node);
    materializeLoopNode(loop_node);
  } else if (node.isStageNode()) {
    const StageNode& stage_node = static_cast<const StageNode&>(node);
    materializeStageNode(stage_node);
  } else if (node.isPipelineNode()) {
    const PipelineNode& pipeline_node = static_cast<const PipelineNode&>(node);
    materializePipelineNode(pipeline_node);
  } else if (node.isPrivateNode()) {
    const PrivateNode& private_node = static_cast<const PrivateNode&>(node);
    materializePrivateNode(private_node);
  } else {
    llvm_unreachable("unimplemented node type");
  }
}

void PathExpansionMaterializer::materializeChildren(
    const PipelineTreeNode& parent_node) {
  OperationTreeNode* child = parent_node.getFirstChild();
  while (child) {
    materializeTreeNode(static_cast<const PipelineTreeNode&>(*child));
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

mlir::Value PathExpansionMaterializer::materializeValue(mlir::Value value) {
  return value_map_.lookupOrDefault(value);
}

mlir::OpFoldResult PathExpansionMaterializer::materializeOpFoldResult(
    mlir::OpFoldResult foldResult) {
  if (auto value = foldResult.dyn_cast<mlir::Value>()) {
    // It's a dynamic value - materialize it through value_map_
    return materializeValue(value);
  }
  // It's a static attribute - return as-is
  return foldResult;
}

mlir::Value PathExpansionMaterializer::getPrivateResourceValue(
    const PrivateResourceSpec* spec, size_t slot_index) {
  if (spec->kind == PrivateResourceSpec::Kind::kExistingValue)
    return materializeValue(spec->existing_value);
  const PrivateResourceAllocation* alloc =
      resource_factory_.getAllocation(spec);
  assert(alloc && "Private resource should have been allocated");
  assert(slot_index < alloc->ssa_values.size() && "Slot index out of bounds");
  return alloc->ssa_values[slot_index];
}

PathExpansionMaterializer::MaterializedSide
PathExpansionMaterializer::materializeSide(const TransferSide& side,
                                           mlir::Value original_operand) {
  MaterializedSide result;
  result.value = side.resource
                     ? getPrivateResourceValue(side.resource, side.slot_index)
                     : materializeValue(original_operand);
  result.map = side.map;
  for (mlir::Value idx : side.indices)
    result.indices.push_back(materializeValue(idx));
  for (mlir::OpFoldResult size : side.sizes)
    result.sizes.push_back(materializeOpFoldResult(size));
  return result;
}

void PathExpansionMaterializer::createDataTransfer(
    mlir::Location loc, MaterializedSide source, MaterializedSide dest,
    mlir::Operation* template_op) {
  mlir::AffineMap source_map =
      canonicalizeMapAndIndices(source.map, source.indices);
  mlir::AffineMap dest_map = canonicalizeMapAndIndices(dest.map, dest.indices);
  auto transfer = mlir::ktdf::DataTransferOp::create(
      builder_, loc, source.value, source_map, source.indices, source.sizes,
      dest.value, dest_map, dest.indices, dest.sizes);
  if (template_op)
    transfer->setDiscardableAttrs(template_op->getRawDictionaryAttrs());
}

//===----------------------------------------------------------------------===//
// Loop Node Materialization
//===----------------------------------------------------------------------===//

mlir::scf::ForOp PathExpansionMaterializer::materializeLoopNode(
    const LoopNode& loop_node) {
  // Get the template operation to copy loop bounds from
  mlir::Operation* template_op = loop_node.getOperationOrTemplate();

  assert(template_op && "Loop node must have either operation or template");
  auto template_for = mlir::cast<mlir::scf::ForOp>(template_op);

  // Materialize the loop bounds
  mlir::Value lower_bound = materializeValue(template_for.getLowerBound());
  mlir::Value upper_bound = materializeValue(template_for.getUpperBound());
  mlir::Value step = materializeValue(template_for.getStep());

  // Create the new loop with empty body
  auto new_loop = mlir::scf::ForOp::create(builder_, template_for.getLoc(),
                                           lower_bound, upper_bound, step);

  // Copy attributes from template loop to new loop
  for (mlir::NamedAttribute attr : template_for->getAttrs()) {
    new_loop->setAttr(attr.getName(), attr.getValue());
  }

  // Map the induction variable
  value_map_.map(template_for.getInductionVar(), new_loop.getInductionVar());

  // Track the materialized operation
  node_to_materialized_op_[&loop_node] = new_loop.getOperation();

  // Set insertion point inside the loop body and materialize children
  builder_.setInsertionPointToStart(new_loop.getBody());
  materializeChildren(loop_node);
  builder_.setInsertionPointAfter(new_loop);

  LDBG(1) << "  Materialized loop node";
  return new_loop;
}

//===----------------------------------------------------------------------===//
// Pipeline Node Materialization
//===----------------------------------------------------------------------===//

mlir::ktdf::PipelineOp PathExpansionMaterializer::materializePipelineNode(
    const PipelineNode& pipeline_node) {
  mlir::Operation* template_op = pipeline_node.getOperationOrTemplate();
  assert(template_op && "Pipeline node must have template");

  auto template_pipeline = mlir::cast<mlir::ktdf::PipelineOp>(template_op);
  mlir::Location loc = template_pipeline.getLoc();

  // Create new pipeline operation
  auto new_pipeline = mlir::ktdf::PipelineOp::create(builder_, loc);

  // Track the materialized operation
  node_to_materialized_op_[&pipeline_node] = new_pipeline.getOperation();

  // Set insertion point inside the pipeline body
  builder_.setInsertionPointToStart(new_pipeline.getBody());

  // Materialize all children (private node first, then stages) via pre-order
  // traversal
  materializeChildren(pipeline_node);

  builder_.setInsertionPointAfter(new_pipeline);

  LDBG(1) << "  Materialized pipeline node";
  return new_pipeline;
}

//===----------------------------------------------------------------------===//
// Private Node Materialization
//===----------------------------------------------------------------------===//

void PathExpansionMaterializer::materializePrivateNode(
    const PrivateNode& private_node) {
  mlir::Operation* template_op = private_node.getOperationOrTemplate();
  assert(template_op && "Private node must have template");

  auto orig_private = mlir::cast<mlir::ktdf::PrivateOp>(template_op);
  mlir::Location loc = orig_private.getLoc();
  size_t orig_result_count = orig_private.getNumResults();

  // Get parent pipeline node
  const PipelineTreeNode* parent =
      static_cast<const PipelineTreeNode*>(private_node.getParentNode());
  assert(parent && parent->isPipelineNode() &&
         "Private node must have a pipeline parent");
  const PipelineNode* pipeline_node = static_cast<const PipelineNode*>(parent);

  // Identify stages that have dependencies (need tokens)
  llvm::SmallVector<StageNode*> stages_with_deps;
  for (StageNode* stage : pipeline_node->getStages()) {
    if (stage->getDependencies().empty()) continue;
    stages_with_deps.push_back(stage);
  }

  // Build result types for the new private operation
  size_t non_token_result_count;
  llvm::SmallVector<mlir::Type> result_types = buildPrivateResultTypes(
      orig_private, stages_with_deps, non_token_result_count);

  // Create the new private operation
  auto new_private = mlir::ktdf::PrivateOp::create(builder_, loc, result_types);

  // Set insertion point inside private body and clone original operations
  builder_.setInsertionPointToStart(new_private.getBody());
  llvm::SmallVector<mlir::Value> yield_operands =
      cloneOriginalPrivateBody(orig_private);

  // Allocate new private resources and add to yield operands
  allocateNewPrivateResources(loc, yield_operands);

  // Create tokens for stages with dependencies
  createStageTokens(loc, stages_with_deps, yield_operands);

  // Create private_yield with all operands (original + new)
  mlir::ktdf::PrivateYieldOp::create(builder_, loc, yield_operands);

  // Build stage-to-token mapping
  buildStageToTokenMapping(stages_with_deps, new_private,
                           non_token_result_count);

  // Map original private results to new private results in value_map_
  for (size_t i = 0; i < orig_result_count; ++i) {
    value_map_.map(orig_private.getResult(i), new_private.getResult(i));
  }

  // Register the private op results with the factory
  registerAllocatedResources(new_private, orig_result_count);

  // Reset insertion point to after private op
  builder_.setInsertionPointAfter(new_private);

  LDBG(1) << "  Materialized private node with " << orig_result_count
          << " original resources, " << resource_factory_.getSpecs().size()
          << " new buffers, and " << stages_with_deps.size() << " tokens";
}

void PathExpansionMaterializer::buildStageToTokenMapping(
    const llvm::SmallVector<StageNode*>& stages_with_deps,
    mlir::ktdf::PrivateOp new_private, size_t non_token_result_count) {
  // Map each stage to its token (results after non-token results)
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    StageNode* stage = stages_with_deps[i];
    mlir::Value token = new_private.getResult(non_token_result_count + i);
    stage_to_token_map_[stage] = token;
  }
}

llvm::SmallVector<mlir::Type>
PathExpansionMaterializer::buildPrivateResultTypes(
    mlir::ktdf::PrivateOp orig_private,
    const llvm::SmallVector<StageNode*>& stages_with_deps,
    size_t& out_non_token_result_count) {
  llvm::SmallVector<mlir::Type> result_types;

  // First, add all original private result types (existing buffers and FIFOs)
  size_t orig_result_count = orig_private.getNumResults();
  for (size_t i = 0; i < orig_result_count; ++i) {
    result_types.push_back(orig_private.getResult(i).getType());
  }

  // Add new private resources (buffers and FIFOs) from factory
  for (const auto& spec_ptr : resource_factory_.getSpecs()) {
    const PrivateResourceSpec& spec = *spec_ptr;
    if (spec.kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      // memory_resource is already an attribute
      mlir::MemRefType memref_type = mlir::MemRefType::get(
          spec.shape, spec.element_type, mlir::MemRefLayoutAttrInterface{},
          spec.memory_resource);
      result_types.push_back(memref_type);
      continue;
    }
    if (spec.kind != PrivateResourceSpec::Kind::kFifo) continue;
    // Add one result type per FIFO slot
    for (int64_t num_elements : spec.elements_per_slot) {
      mlir::ktdf::FifoSlotType fifo_type = mlir::ktdf::FifoSlotType::get(
          builder_.getContext(), spec.fifo_src, spec.fifo_dest, num_elements,
          spec.element_type);
      result_types.push_back(fifo_type);
    }
  }

  out_non_token_result_count = result_types.size();

  // Add tokens for stages with dependencies
  mlir::Type token_type = mlir::ktdf::TokenType::get(builder_.getContext());
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    result_types.push_back(token_type);
  }

  return result_types;
}

llvm::SmallVector<mlir::Value>
PathExpansionMaterializer::cloneOriginalPrivateBody(
    mlir::ktdf::PrivateOp orig_private) {
  // Clone operations from original private op's body (except the yield)
  mlir::Block& orig_body = *orig_private.getBody();
  for (mlir::Operation& op : orig_body.without_terminator()) {
    builder_.clone(op, value_map_);
  }

  // Get the original yield operands (now mapped to cloned values)
  auto orig_yield =
      mlir::cast<mlir::ktdf::PrivateYieldOp>(orig_body.getTerminator());
  llvm::SmallVector<mlir::Value> yield_operands;
  for (mlir::Value orig_operand : orig_yield.getOperands()) {
    yield_operands.push_back(value_map_.lookup(orig_operand));
  }

  return yield_operands;
}

void PathExpansionMaterializer::allocateNewPrivateResources(
    mlir::Location loc, llvm::SmallVector<mlir::Value>& yield_operands) {
  for (const auto& spec_ptr : resource_factory_.getSpecs()) {
    const PrivateResourceSpec* spec = spec_ptr.get();

    if (spec->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      // memory_resource is already an attribute
      mlir::MemRefType memref_type = mlir::MemRefType::get(
          spec->shape, spec->element_type, mlir::MemRefLayoutAttrInterface{},
          spec->memory_resource);

      auto alloc_op = mlir::memref::AllocOp::create(builder_, loc, memref_type);
      yield_operands.push_back(alloc_op.getResult());
      continue;
    }

    if (spec->kind != PrivateResourceSpec::Kind::kFifo) continue;

    // Create FIFO allocation with multiple slots
    llvm::SmallVector<mlir::Type> fifo_slot_types;
    for (int64_t num_elements : spec->elements_per_slot) {
      mlir::ktdf::FifoSlotType fifo_type = mlir::ktdf::FifoSlotType::get(
          builder_.getContext(), spec->fifo_src, spec->fifo_dest, num_elements,
          spec->element_type);
      fifo_slot_types.push_back(fifo_type);
    }

    llvm::SmallVector<mlir::Value> empty_operands;
    auto fifo_alloc = mlir::ktdf::FifoAllocateOp::create(
        builder_, loc, fifo_slot_types, empty_operands);

    // Add all slots to yield operands
    for (size_t slot_idx = 0; slot_idx < fifo_alloc.getNumResults();
         ++slot_idx) {
      yield_operands.push_back(fifo_alloc.getResult(slot_idx));
    }
  }
}

void PathExpansionMaterializer::createStageTokens(
    mlir::Location loc, const llvm::SmallVector<StageNode*>& stages_with_deps,
    llvm::SmallVector<mlir::Value>& yield_operands) {
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    auto token_op = mlir::ktdf::CreateTokenOp::create(builder_, loc);
    yield_operands.push_back(token_op.getResult());
  }
}

void PathExpansionMaterializer::registerAllocatedResources(
    mlir::ktdf::PrivateOp new_private, size_t orig_result_count) {
  size_t current_result_idx = orig_result_count;
  for (const auto& spec_ptr : resource_factory_.getSpecs()) {
    const PrivateResourceSpec* spec = spec_ptr.get();

    if (spec->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      // Register the private op result (not the alloc result)
      resource_factory_.registerAllocation(
          spec, {new_private.getResult(current_result_idx)});
      current_result_idx++;
      continue;
    }

    if (spec->kind != PrivateResourceSpec::Kind::kFifo) continue;

    // Register all FIFO slot results from the private op
    llvm::SmallVector<mlir::Value> slot_values;
    for (size_t slot_idx = 0; slot_idx < spec->elements_per_slot.size();
         ++slot_idx) {
      slot_values.push_back(new_private.getResult(current_result_idx));
      current_result_idx++;
    }
    resource_factory_.registerAllocation(spec, slot_values);
  }
}

//===----------------------------------------------------------------------===//
// Stage Node Materialization
//===----------------------------------------------------------------------===//

mlir::ktdf::StageOp PathExpansionMaterializer::materializeStageNode(
    const StageNode& stage_node) {
  // Look up materialization info for this stage
  auto info_it = stage_info_.find(const_cast<StageNode*>(&stage_node));

  mlir::Operation* template_op = stage_node.getOperationOrTemplate();
  mlir::Location loc =
      template_op ? template_op->getLoc() : builder_.getUnknownLoc();

  // Build depends_in and depends_out from stage dependencies
  // (same logic as stage coarsening)
  llvm::SmallVector<mlir::Value> depends_in;
  llvm::SmallVector<mlir::Value> depends_out;

  // For depends_out: find the token produced by this stage
  if (!stage_node.getDependencies().empty()) {
    auto it = stage_to_token_map_.find(const_cast<StageNode*>(&stage_node));
    assert(it != stage_to_token_map_.end() &&
           "expected token map to be updated for this stage");
    depends_out.push_back(it->second);
  }

  // For depends_in: find stages that this stage depends on
  // We need to look at all stages in the parent pipeline and check if this
  // stage is in their list of dependents.
  const PipelineTreeNode* parent =
      static_cast<const PipelineTreeNode*>(stage_node.getParentNode());
  assert(parent && parent->isPipelineNode());
  const PipelineNode* pipeline_parent =
      static_cast<const PipelineNode*>(parent);
  for (StageNode* other_stage : pipeline_parent->getStages()) {
    if (other_stage == &stage_node) continue;
    // Check if this stage is in other_stage's dependencies
    for (StageNode* dep : other_stage->getDependencies()) {
      if (dep != &stage_node) continue;
      // This stage depends on other_stage (other_stage unblocks this stage)
      auto it = stage_to_token_map_.find(other_stage);
      assert(it != stage_to_token_map_.end());
      depends_in.push_back(it->second);
    }
  }

  // Create the stage operation with proper dependencies
  auto new_stage =
      mlir::ktdf::StageOp::create(builder_, loc, depends_in, depends_out);

  // Set applicable_units: prefer planner's assigned unit, fallback to template
  // op
  if (info_it != stage_info_.end() &&
      info_it->second.applicable_unit.has_value()) {
    auto applicable_units_attr =
        builder_.getArrayAttr({*info_it->second.applicable_unit});
    new_stage.setApplicableUnitsAttr(applicable_units_attr);
  } else if (template_op) {
    // Copy applicable_units from template
    auto template_stage = mlir::cast<mlir::ktdf::StageOp>(template_op);
    if (auto applicable_units = template_stage.getApplicableUnits()) {
      new_stage.setApplicableUnitsAttr(*applicable_units);
    }
  }

  // Track the materialized operation
  node_to_materialized_op_[&stage_node] = new_stage.getOperation();

  // Set insertion point inside stage body
  builder_.setInsertionPointToStart(new_stage.getBody());

  // Materialize stage body based on kind
  if (info_it != stage_info_.end()) {
    const StageMaterializationInfo& info = info_it->second;

    switch (info.kind) {
      case StageMaterializationInfo::Kind::kPreserveOriginal:
        assert(template_op);
        cloneStageBody(mlir::cast<mlir::ktdf::StageOp>(template_op));
        break;

      case StageMaterializationInfo::Kind::kAdaptFifoKinds:
        assert(template_op);
        adaptStageBodyWithTransfers(
            mlir::cast<mlir::ktdf::StageOp>(template_op), info,
            /*handle_fifo_ops=*/true);
        break;

      case StageMaterializationInfo::Kind::kAdaptTransfer:
        assert(template_op);
        adaptStageBodyWithTransfers(
            mlir::cast<mlir::ktdf::StageOp>(template_op), info,
            /*handle_fifo_ops=*/false);
        break;

      case StageMaterializationInfo::Kind::kSyntheticTransfer:
        synthesizeTransferStage(info);
        break;
    }
    if (info.kind != StageMaterializationInfo::Kind::kSyntheticTransfer)
      appendPlannedTransfers(info);
  } else if (template_op) {
    // No info found, preserve original
    cloneStageBody(mlir::cast<mlir::ktdf::StageOp>(template_op));
  }

  builder_.setInsertionPointAfter(new_stage);

  LDBG(1) << "  Materialized stage " << stage_node.getStageId();
  return new_stage;
}

void PathExpansionMaterializer::cloneStageBody(mlir::ktdf::StageOp orig_stage) {
  mlir::Block* orig_body = orig_stage.getBody();
  for (mlir::Operation& op : *orig_body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
    builder_.clone(op, value_map_);
  }
}

bool PathExpansionMaterializer::tryAdaptDataTransferOp(
    mlir::ktdf::DataTransferOp transfer_op,
    const TransferMaterializationInfo* transfer_info) {
  if (!transfer_info) return false;

  createDataTransfer(
      transfer_op.getLoc(),
      materializeSide(transfer_info->source, transfer_op.getSource()),
      materializeSide(transfer_info->dest, transfer_op.getDestination()),
      transfer_info->template_op);

  return true;
}

bool PathExpansionMaterializer::tryAdaptIndDataTransferOp(
    mlir::ktdf::IndDataTransferOp ind_transfer_op,
    const TransferMaterializationInfo* transfer_info) {
  if (!transfer_info) return false;

  // Cloned-verbatim operands (common to both gather and scatter).
  auto materializeIfSet = [&](mlir::Value value) {
    return value ? materializeValue(value) : mlir::Value{};
  };
  mlir::Value ind_src_memref =
      materializeIfSet(ind_transfer_op.getIndSrcMemref());
  mlir::Value ind_src_index =
      materializeIfSet(ind_transfer_op.getIndSrcIndex());
  mlir::Value ind_dst_memref =
      materializeIfSet(ind_transfer_op.getIndDstMemref());
  mlir::Value ind_dst_index =
      materializeIfSet(ind_transfer_op.getIndDstIndex());

  // The planner replaces the FIFO side (dir_dst of a gather, dir_src of a
  // scatter) with a staging buffer and keeps the other side as it is.
  MaterializedSide dir_src =
      materializeSide(transfer_info->source, ind_transfer_op.getDirSrc());
  MaterializedSide dir_dst =
      materializeSide(transfer_info->dest, ind_transfer_op.getDirDst());

  auto result = mlir::ktdf::IndDataTransferOp::create(
      builder_, ind_transfer_op.getLoc(), ind_src_memref, ind_src_index,
      dir_src.value, dir_src.map, dir_src.indices, dir_src.sizes,
      ind_dst_memref, ind_dst_index, dir_dst.value, dir_dst.map,
      dir_dst.indices, dir_dst.sizes);
  result->setDiscardableAttrs(
      transfer_info->template_op->getRawDictionaryAttrs());

  return true;
}

bool PathExpansionMaterializer::tryAdaptReadFromFifoOp(
    mlir::ktdf::ReadFromFifoOp read_op,
    const TransferMaterializationInfo* transfer_info) {
  if (!transfer_info) return false;

  assert(transfer_info->source.resource &&
         "read_from_fifo should have source private resource");

  mlir::Value new_fifo_slot = getPrivateResourceValue(
      transfer_info->source.resource, transfer_info->source.slot_index);

  auto new_read = mlir::ktdf::ReadFromFifoOp::create(
      builder_, read_op.getLoc(), read_op.getResult().getType(), new_fifo_slot);
  value_map_.map(read_op.getResult(), new_read.getResult());

  return true;
}

bool PathExpansionMaterializer::tryAdaptWriteToFifoOp(
    mlir::ktdf::WriteToFifoOp write_op,
    const TransferMaterializationInfo* transfer_info) {
  if (!transfer_info) return false;

  assert(transfer_info->dest.resource &&
         "write_to_fifo should have dest private resource");

  mlir::Value new_fifo_slot = getPrivateResourceValue(
      transfer_info->dest.resource, transfer_info->dest.slot_index);

  mlir::Value data = materializeValue(write_op.getData());
  mlir::ktdf::WriteToFifoOp::create(builder_, write_op.getLoc(), data,
                                    new_fifo_slot);

  return true;
}

void PathExpansionMaterializer::adaptStageBodyWithTransfers(
    mlir::ktdf::StageOp orig_stage, const StageMaterializationInfo& info,
    bool handle_fifo_ops) {
  // Clone the stage body but adapt transfer operations to use new resources
  mlir::Block* orig_body = orig_stage.getBody();

  // Build a map from original transfer ops to their transfer info
  llvm::DenseMap<mlir::Operation*, const TransferMaterializationInfo*>
      transfer_map;
  for (const TransferMaterializationInfo* transfer_info : info.transfers) {
    if (transfer_info->template_op) {
      transfer_map[transfer_info->template_op] = transfer_info;
    }
  }

  for (mlir::Operation& op : *orig_body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;

    // Try to adapt based on operation type
    bool adapted = false;
    if (auto transfer_op = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(&op)) {
      adapted = tryAdaptDataTransferOp(transfer_op, transfer_map.lookup(&op));
    } else if (auto ind_transfer_op =
                   mlir::dyn_cast<mlir::ktdf::IndDataTransferOp>(&op)) {
      // IndDataTransferOp is a transfer-path op, not a FIFO op. It must only
      // be reached when handle_fifo_ops == false (i.e., the transfer-adapt
      // path). If called from a FIFO-handling context, transfer_map will have
      // no entry and tryAdaptIndDataTransferOp would silently clone as-is,
      // producing wrong output.
      assert(!handle_fifo_ops &&
             "IndDataTransferOp must not appear in a FIFO-handling stage");
      adapted =
          tryAdaptIndDataTransferOp(ind_transfer_op, transfer_map.lookup(&op));
    } else if (handle_fifo_ops) {
      if (auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(&op)) {
        adapted = tryAdaptReadFromFifoOp(read_op, transfer_map.lookup(&op));
      } else if (auto write_op =
                     mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(&op)) {
        adapted = tryAdaptWriteToFifoOp(write_op, transfer_map.lookup(&op));
      }
    }

    // If not adapted, clone as-is
    if (!adapted) {
      builder_.clone(op, value_map_);
    }
  }
}

void PathExpansionMaterializer::synthesizeTransferStage(
    const StageMaterializationInfo& info) {
  // Synthesize new data transfer operations based on transfer info
  for (const TransferMaterializationInfo* transfer_info : info.transfers)
    emitSyntheticTransfer(*transfer_info);
}

void PathExpansionMaterializer::appendPlannedTransfers(
    const StageMaterializationInfo& info) {
  for (const TransferMaterializationInfo* transfer_info : info.transfers)
    if (!transfer_info->template_op) emitSyntheticTransfer(*transfer_info);
}

void PathExpansionMaterializer::emitSyntheticTransfer(
    const TransferMaterializationInfo& transfer_info) {
  mlir::Location loc = transfer_info.template_op
                           ? transfer_info.template_op->getLoc()
                           : builder_.getUnknownLoc();

  assert(transfer_info.source.resource &&
         "Synthetic transfer must have source private resource");
  assert(transfer_info.dest.resource &&
         "Synthetic transfer must have dest private resource");

  createDataTransfer(loc, materializeSide(transfer_info.source, {}),
                     materializeSide(transfer_info.dest, {}),
                     transfer_info.template_op);
}

}  // namespace scheduler

// Made with Bob