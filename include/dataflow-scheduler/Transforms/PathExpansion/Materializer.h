//===------------------------------------------------------------*- c++ -*-===//
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
// Path Expansion Pipeline Tree Materializer
//
// This file defines a path-expansion-specific materializer for generating
// MLIR IR from a modified PipelineTree. The materializer traverses the tree
// in pre-order, materializes the private node first to create SSA values for
// resources, then materializes each stage according to its materialization
// kind (preserve, adapt, or synthesize).
//
// This materializer is intentionally structured similarly to the stage
// coarsening materializer to enable future refactoring and code sharing.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_PATHEXPANSION_MATERIALIZER_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_PATHEXPANSION_MATERIALIZER_H_

#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

namespace scheduler {

//===----------------------------------------------------------------------===//
// Path Expansion Pipeline Tree Materializer
//===----------------------------------------------------------------------===//

/// Materializer class for generating MLIR IR from a PipelineTree for the
/// path-expansion transform.
class PathExpansionMaterializer {
 public:
  /// Constructor
  /// \param builder The MLIR builder for creating operations
  /// \param value_map The value mapping for SSA values
  /// \param stage_info Map from StageNode to materialization info
  /// \param resource_factory Factory that owns and tracks private resources
  PathExpansionMaterializer(
      mlir::OpBuilder& builder, mlir::IRMapping& value_map,
      const llvm::DenseMap<StageNode*, StageMaterializationInfo>& stage_info,
      PrivateResourceFactory& resource_factory)
      : builder_(builder),
        value_map_(value_map),
        stage_info_(stage_info),
        resource_factory_(resource_factory) {}

  /// Materialize the entire tree starting from the given node
  /// Returns the root operation created
  mlir::Operation* materialize(const PipelineTreeNode& root_node);

  /// Get the materialized operation for a given tree node
  /// Returns nullptr if the node hasn't been materialized yet
  mlir::Operation* getMaterializedOp(const PipelineTreeNode& node) const;

 private:
  /// One side of a transfer op to create: the value it reads or writes and
  /// how it addresses it, all in terms of the new IR.
  struct MaterializedSide {
    mlir::Value value;
    mlir::AffineMap map;
    llvm::SmallVector<mlir::Value> indices;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
  };

  /// Helper to materialize a value by looking it up in the value map
  mlir::Value materializeValue(mlir::Value value);

  /// Helper to materialize an OpFoldResult (either a constant attribute or SSA
  /// value)
  mlir::OpFoldResult materializeOpFoldResult(mlir::OpFoldResult foldResult);

  /// Helper to get SSA value for a private resource
  mlir::Value getPrivateResourceValue(const PrivateResourceSpec* spec,
                                      size_t slot_index);

  /// Materialize one side of a planned transfer. The side uses its private
  /// resource if it has one, and \p original_operand (an operand of the
  /// template op, may be null) otherwise.
  MaterializedSide materializeSide(const TransferSide& side,
                                   mlir::Value original_operand);

  /// Create a data_transfer from \p source to \p dest, copying the
  /// discardable attributes of \p template_op if it is set.
  void createDataTransfer(mlir::Location loc, MaterializedSide source,
                          MaterializedSide dest, mlir::Operation* template_op);

  /// Materialize a loop node by creating an scf.for operation
  mlir::scf::ForOp materializeLoopNode(const LoopNode& loop_node);

  /// Materialize a stage node by creating a ktdf.stage operation
  mlir::ktdf::StageOp materializeStageNode(const StageNode& stage_node);

  /// Materialize a pipeline node by creating a ktdf.pipeline operation
  mlir::ktdf::PipelineOp materializePipelineNode(
      const PipelineNode& pipeline_node);

  /// Materialize a private node by creating new ktdf.private with all resources
  void materializePrivateNode(const PrivateNode& private_node);

  /// Recursively materialize a tree node and its children
  void materializeTreeNode(const PipelineTreeNode& node);

  /// Helper to materialize all children of a node
  void materializeChildren(const PipelineTreeNode& parent_node);

  /// Helper to clone stage body operations (for kPreserveOriginal)
  void cloneStageBody(mlir::ktdf::StageOp orig_stage);

  /// Common helper to adapt stage body with transfer operations
  /// \param handle_fifo_ops If true, also handles ReadFromFifoOp and
  /// WriteToFifoOp
  void adaptStageBodyWithTransfers(mlir::ktdf::StageOp orig_stage,
                                   const StageMaterializationInfo& info,
                                   bool handle_fifo_ops);

  /// Helper to adapt a DataTransferOp with new resources
  /// Returns true if adapted, false if should be cloned as-is
  bool tryAdaptDataTransferOp(mlir::ktdf::DataTransferOp transfer_op,
                              const TransferMaterializationInfo* transfer_info);

  /// Helper to adapt an IndDataTransferOp, repointing the buffer-facing operand
  /// (dir_dst for gather, dir_src for scatter) to the L1 staging buffer.
  /// Returns true if adapted, false if should be cloned as-is.
  bool tryAdaptIndDataTransferOp(
      mlir::ktdf::IndDataTransferOp ind_transfer_op,
      const TransferMaterializationInfo* transfer_info);

  /// Helper to adapt a ReadFromFifoOp with new FIFO slot
  /// Returns true if adapted, false if should be cloned as-is
  bool tryAdaptReadFromFifoOp(mlir::ktdf::ReadFromFifoOp read_op,
                              const TransferMaterializationInfo* transfer_info);

  /// Helper to adapt a WriteToFifoOp with new FIFO slot
  /// Returns true if adapted, false if should be cloned as-is
  bool tryAdaptWriteToFifoOp(mlir::ktdf::WriteToFifoOp write_op,
                             const TransferMaterializationInfo* transfer_info);

  /// Helper to synthesize a new transfer stage (for kSyntheticTransfer)
  void synthesizeTransferStage(const StageMaterializationInfo& info);

  /// Emit the transfers the planner created for an original stage that also
  /// moves data for a path through its unit. These are the transfers with no
  /// template op; they go after the stage's own body.
  void appendPlannedTransfers(const StageMaterializationInfo& info);

  /// Emit one data transfer the planner created between two private
  /// resources.
  void emitSyntheticTransfer(const TransferMaterializationInfo& transfer_info);

  /// Helper to build stage-to-token mapping from private results
  void buildStageToTokenMapping(
      const llvm::SmallVector<StageNode*>& stages_with_deps,
      mlir::ktdf::PrivateOp new_private, size_t non_token_result_count);

  /// Helper to build result types for private operation
  llvm::SmallVector<mlir::Type> buildPrivateResultTypes(
      mlir::ktdf::PrivateOp orig_private,
      const llvm::SmallVector<StageNode*>& stages_with_deps,
      size_t& out_non_token_result_count);

  /// Helper to clone original private body operations
  llvm::SmallVector<mlir::Value> cloneOriginalPrivateBody(
      mlir::ktdf::PrivateOp orig_private);

  /// Helper to allocate new private resources (buffers and FIFOs)
  void allocateNewPrivateResources(
      mlir::Location loc, llvm::SmallVector<mlir::Value>& yield_operands);

  /// Helper to create tokens for stages with dependencies
  void createStageTokens(mlir::Location loc,
                         const llvm::SmallVector<StageNode*>& stages_with_deps,
                         llvm::SmallVector<mlir::Value>& yield_operands);

  /// Helper to register allocated resources with factory
  void registerAllocatedResources(mlir::ktdf::PrivateOp new_private,
                                  size_t orig_result_count);

  /// MLIR builder for creating operations
  mlir::OpBuilder& builder_;

  /// Value mapping for SSA values
  mlir::IRMapping& value_map_;

  /// Map from StageNode to materialization info
  const llvm::DenseMap<StageNode*, StageMaterializationInfo>& stage_info_;

  /// Resource factory that owns and tracks private resources
  PrivateResourceFactory& resource_factory_;

  /// Tracking map from tree nodes to their materialized operations
  llvm::DenseMap<const PipelineTreeNode*, mlir::Operation*>
      node_to_materialized_op_;

  /// Mapping from stages to their corresponding token values (results of
  /// ktdf.private op) This is used to fill in depends_in/depends_out when
  /// materializing stages
  llvm::DenseMap<StageNode*, mlir::Value> stage_to_token_map_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_PATHEXPANSION_MATERIALIZER_H_

// Made with Bob