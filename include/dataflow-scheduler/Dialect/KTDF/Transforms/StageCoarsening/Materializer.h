//===-- Materializer.h ------------------------------------------*- c++ -*-===//
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
// Stage Coarsening Pipeline Tree Materializer
//
// This file defines a stage-coarsening-specific materializer for generating
// MLIR IR from a PipelineTree. The materializer traverses the tree and creates
// corresponding MLIR operations (loops, pipelines, stages) while maintaining
// value mappings for SSA values.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_MATERIALIZER_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_MATERIALIZER_H_

#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::ktdf {

//===----------------------------------------------------------------------===//
// Buffer Cloner Interface
//===----------------------------------------------------------------------===//

/// Cloner interface for customizing buffer allocation and data transfer
/// operations during materialization. This allows transformations to modify
/// how buffers are allocated (e.g., expanding dimensions) and how data
/// transfers access those buffers (e.g., adjusting indices).
class BufferCloner {
 public:
  virtual ~BufferCloner() = default;

  /// Called when materializing a memref.alloc operation.
  /// The cloner can return a custom allocation operation or nullptr to use
  /// default cloning behavior.
  ///
  /// \param orig_alloc The original allocation operation from the template
  /// \param builder The builder positioned where the new alloc should be
  /// created
  /// \param value_map The current value mapping for SSA values
  /// \return A new allocation operation, or nullptr to use default behavior
  virtual memref::AllocOp cloneAllocOp(memref::AllocOp orig_alloc,
                                       OpBuilder& builder,
                                       IRMapping& value_map) = 0;

  /// Called when materializing a data transfer operation.
  /// The cloner can return a custom transfer operation or nullptr to use
  /// default cloning behavior.
  ///
  /// \param orig_transfer The original data transfer operation
  /// \param builder The builder positioned where the new transfer should be
  /// created
  /// \param value_map The current value mapping for SSA values
  /// \return A new data transfer operation, or nullptr to use default behavior
  virtual ktdf::DataTransferOp cloneDataTransfer(
      ktdf::DataTransferOp orig_transfer, OpBuilder& builder,
      IRMapping& value_map) = 0;

  /// Called when materializing an indirect data transfer operation.
  /// The cloner can return a custom indirect transfer operation or nullptr to
  /// use default cloning behavior.
  ///
  /// \param orig_transfer The original indirect data transfer operation
  /// \param builder The builder positioned where the new transfer should be
  /// created
  /// \param value_map The current value mapping for SSA values
  /// \return A new indirect data transfer operation, or nullptr to use default
  /// behavior
  virtual ktdf::IndDataTransferOp cloneIndDataTransfer(
      ktdf::IndDataTransferOp orig_transfer, OpBuilder& builder,
      IRMapping& value_map) {
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Stage Coarsening Pipeline Tree Materializer
//===----------------------------------------------------------------------===//

/// Materializer class for generating MLIR IR from a PipelineTree for the
/// stage-coarsening transform.
class StageCoarseningMaterializer {
 public:
  /// Constructor
  /// \param builder The MLIR builder for creating operations
  /// \param value_map The value mapping for SSA values
  /// \param buffer_cloner Optional cloner for customizing buffer operations
  StageCoarseningMaterializer(OpBuilder& builder, IRMapping& value_map,
                              BufferCloner* buffer_cloner = nullptr)
      : builder_(builder),
        value_map_(value_map),
        buffer_cloner_(buffer_cloner) {}

  /// Materialize the entire tree starting from the given node
  /// Returns the root operation created
  Operation* materialize(const scheduler::PipelineTreeNode& root_node);

  /// Get the materialized operation for a given tree node
  /// Returns nullptr if the node hasn't been materialized yet
  Operation* getMaterializedOp(const scheduler::PipelineTreeNode& node) const;

  /// Check if an operation is allowed between nested loops (for bound
  /// computation) Only allows: affine.apply, arith.addi, arith.subi,
  /// arith.minsi, arith.maxsi
  static bool isOpAllowedBetweenLoops(Operation* op);

 private:
  /// Check if an operation is in the same block as the template operation
  static bool isInSameBlock(Operation* op, Operation* template_op);

  /// Helper to materialize a value by cloning its defining operation if needed
  /// This handles cases where loop bounds are computed by arith/affine
  /// operations
  Value materializeValue(Value value, Operation* template_op);

  /// Materialize a loop node by creating an scf.for operation
  scf::ForOp materializeLoopNode(const scheduler::LoopNode& loop_node);

  /// Materialize a stage node by creating a ktdf.stage operation
  ktdf::StageOp materializeStageNode(const scheduler::StageNode& stage_node);

  /// Materialize a pipeline node by creating a ktdf.pipeline operation
  ktdf::PipelineOp materializePipelineNode(
      const scheduler::PipelineNode& pipeline_node);

  /// Materialize a private node by cloning its operations
  void materializePrivateNode(const scheduler::PrivateNode& private_node);

  /// Recursively materialize a tree node and its children
  void materializeTreeNode(const scheduler::PipelineTreeNode& node);

  /// Helper to materialize all children of a node
  void materializeChildren(const scheduler::PipelineTreeNode& parent_node);

  /// Helper to clone stage body operations
  void cloneStageBody(ktdf::StageOp orig_stage);

  /// Helper to materialize intervening operations between loop start and first
  /// child These are operations that appear in the template loop body before
  /// the first child operation (e.g., ktdf.tiling.derive_size,
  /// ktdf.tiling.linearize_index)
  void materializeInterveningOps(const scheduler::LoopNode& loop_node);

  /// Helper to create a private node from scratch with only tokens
  void materializePrivateNodeFromScratch(
      const SmallVector<scheduler::StageNode*>& stages_with_deps,
      SmallVector<Value>& new_tokens);

  /// Helper to materialize a private node from an existing template
  void materializePrivateNodeFromTemplate(
      ktdf::PrivateOp orig_private,
      const SmallVector<scheduler::StageNode*>& stages_with_deps,
      SmallVector<Value>& new_tokens,
      const scheduler::PrivateNode& private_node);

  /// Helper to build stage-to-token mapping
  void buildStageToTokenMapping(
      const SmallVector<scheduler::StageNode*>& stages_with_deps,
      ktdf::PrivateOp new_private, size_t non_token_result_count);

  /// MLIR builder for creating operations
  OpBuilder& builder_;

  /// Value mapping for SSA values
  IRMapping& value_map_;

  /// Optional cloner for customizing buffer operations
  BufferCloner* buffer_cloner_;

  /// Tracking map from tree nodes to their materialized operations
  DenseMap<const scheduler::PipelineTreeNode*, Operation*>
      node_to_materialized_op_;

  /// Mapping from stages to their corresponding token values (results of
  /// ktdf.private op) This is used to fill in depends_in/depends_out when
  /// materializing stages
  DenseMap<scheduler::StageNode*, Value> stage_to_token_map_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_MATERIALIZER_H_

// Made with Bob
