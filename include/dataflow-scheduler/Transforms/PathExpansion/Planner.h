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
// Path Expansion Planner
//
// This file defines a lightweight planning layer for path expansion that
// operates directly on PipelineTree. The planner analyzes the original
// pipeline, compares it with the architecture graph's shortest path, and
// updates the PipelineTree in-place while storing only minimal side
// information needed by the materializer.
//
// Key design principles:
// - PipelineTree is the single source of truth for stage structure
// - Planning updates PipelineTree directly (adding/modifying nodes)
// - Side information is minimal and only for materialization needs
// - No redundant parallel abstractions to PipelineTree
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_PATHEXPANSION_PLANNER_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_PATHEXPANSION_PLANNER_H_

#include <optional>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Transforms/PathExpansion/PlannerFactories.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

//===----------------------------------------------------------------------===//
// Materialization Side Information
//===----------------------------------------------------------------------===//

/// Forward declarations for pointer-based references
struct PrivateResourceSpec;
struct PrivateResourceAllocation;

/// Specification for a new private resource to be allocated
/// Can represent either a memory buffer or a FIFO
struct PrivateResourceSpec {
  enum class Kind {
    kUnknown,
    kMemoryBuffer,   // Allocate a memref in the specified memory space
    kFifo,           // Allocate a FIFO with the specified type
    kExistingValue,  // Use a value of the original IR, allocate nothing
  };

  Kind kind = Kind::kUnknown;

  // For Memory Buffer and Existing Value:
  ResourceType memory_resource;

  // For Existing Value: the original value, mapped through the materializer.
  mlir::Value existing_value;

  // For Fifo:
  mlir::Attribute fifo_src;
  mlir::Attribute fifo_dest;
  // How many elements for each slot in the FIFO allocation.
  // Note the size of this list represents the number of slots in the
  // fifo.allocate. For example:
  //     %slot1, %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<SRC -> DST,
  //     2>, !ktdf.fifo.slot<SRC -> DST, 4>
  // \p elements_per_slot would be [2, 4]
  llvm::SmallVector<int64_t> elements_per_slot;

  // For both fifo and memory-buffers:
  llvm::SmallVector<int64_t> shape;
  mlir::Type element_type;

  void print(llvm::raw_ostream& os) const;
  void dump() const;
};

/// Represents an allocated private resource with its SSA value(s)
/// This is created during materialization and tracked by the factory
struct PrivateResourceAllocation {
  // For memory buffers: single SSA value
  // For FIFOs: one SSA value per slot
  llvm::SmallVector<mlir::Value> ssa_values;

  PrivateResourceAllocation(const PrivateResourceSpec*) {}
};

// Factory classes are declared in PlannerFactories.hpp

/// One side of a transfer, its source or its destination: what it reads or
/// writes and how it addresses it.
struct TransferSide {
  // The private resource this side uses. Null to keep the template op's
  // operand.
  const PrivateResourceSpec* resource = nullptr;
  // For a FIFO: which slot to use. Always 0 for a memory buffer.
  size_t slot_index = 0;
  // Indices are Values materialized during code generation. Sizes may be
  // static constants or dynamic SSA values.
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  // Subscript map. Copied from the template op for adapted stages, or an
  // identity map for a buffer of a synthetic transfer. Null for a FIFO.
  mlir::AffineMap map;
};

/// Information needed to materialize a transfer operation
struct TransferMaterializationInfo {
  // Template to derive from (can be null for synthetic)
  mlir::Operation* template_op = nullptr;

  // Original architecture hop being expanded
  scheduler::arch_view::RoutingGraph::EdgeInfo hop;

  // Logical endpoints of this transfer segment. When intermediate buffers split
  // a hop, these differ from getNode(hop.source/target)->resource.
  ResourceType source_resource;
  ResourceType dest_resource;

  // How the transfer fills its destination. Taken from the template op, then
  // moved to the hop that widens the data if path expansion splits the flow.
  mlir::ktdf::TransferMode transfer_mode = mlir::ktdf::TransferMode::Copy;

  // How the transfer reads its source and writes its destination.
  TransferSide source;
  TransferSide dest;

  TransferSide& side(bool source_side) { return source_side ? source : dest; }
  const TransferSide& side(bool source_side) const {
    return source_side ? source : dest;
  }

  void print(llvm::raw_ostream& os) const;
  void dump() const;

  /// Print debug information about this transfer (for LLVM_DEBUG)
  void printDebug(llvm::raw_ostream& os) const;
};

/// Side information attached to a StageNode for materialization
/// This is stored separately and looked up by the materializer
struct StageMaterializationInfo {
  enum class Kind {
    kPreserveOriginal,  // Keep original stage body unchanged (rare)
    kAdaptFifoKinds,  // Preserve body but update FIFO types (usually needed for
                      // compute stages)
    kAdaptTransfer,   // Preserve body but update transfer source or dest types
                      // (needed for transfer stages that are intermdiate-stage
                      // facing)
    kSyntheticTransfer,  // Generate new transfer stage (usually for
                         // intermediate memory ie L1)
  };

  Kind kind = Kind::kPreserveOriginal;

  // Resource associated with this stage (determined in Step 1)
  ResourceType stage_resource;  // Can be null if no resource

  // For AdaptFifoKinds, AdaptTransfer, and SyntheticTransfer:
  // Contains transfer operations with pointers to private resources
  // Pointers are owned by TransferInfoFactory
  llvm::SmallVector<TransferMaterializationInfo*, 4> transfers;

  // which unit this stage runs on
  std::optional<ResourceType> applicable_unit;

  void print(llvm::raw_ostream& os) const;
  void dump() const;
};

//===----------------------------------------------------------------------===//
// Planning Result
//===----------------------------------------------------------------------===//

/// Result of path expansion planning
/// Contains minimal side information needed by the materializer
struct PathExpansionPlan {
  // Side information for materialization (indexed by StageNode*)
  llvm::DenseMap<StageNode*, StageMaterializationInfo> stage_info;

  // Resource factory that owns all PrivateResourceSpec objects
  // and tracks their allocations during materialization
  PrivateResourceFactory resource_factory;

  // Owns all TransferMaterializationInfo objects; consumers access these
  // objects via stage_info[].transfers pointers, not through this factory
  // directly.
  TransferInfoFactory transfer_factory;

  // Whether any changes were made
  bool changed = false;

  // Default constructor
  PathExpansionPlan() = default;

  /// Get stage materialization info for a stage (with assertion)
  StageMaterializationInfo& getStageInfo(StageNode* stage) {
    assert(stage_info.count(stage) && "Stage should have materialization info");
    return stage_info[stage];
  }

  const StageMaterializationInfo& getStageInfo(StageNode* stage) const {
    assert(stage_info.count(stage) && "Stage should have materialization info");
    return stage_info.at(stage);
  }

  // Delete copy operations since PipelineTree is not copyable
  PathExpansionPlan(const PathExpansionPlan&) = delete;
  PathExpansionPlan& operator=(const PathExpansionPlan&) = delete;

  // Delete move operations since PipelineTree is not movable
  PathExpansionPlan(PathExpansionPlan&&) = delete;
  PathExpansionPlan& operator=(PathExpansionPlan&&) = delete;
};

//===----------------------------------------------------------------------===//
// Planning Functions
//===----------------------------------------------------------------------===//

/// Main planning entry point
/// Analyzes the pipeline, compares with architecture graph, and if needed:
/// 1. Updates the PipelineTree in-place (modifying stage DAG)
/// 2. Attaches materialization info to stages
/// 3. Records new private resources needed
/// Returns a unique_ptr to the plan, or nullptr on failure
std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph& routing_graph);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_PATHEXPANSION_PLANNER_H_

// Made with Bob
