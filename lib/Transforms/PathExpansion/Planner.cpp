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

#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"

#include <llvm/Support/ErrorHandling.h>

#include <tuple>
#include <vector>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "path-expansion-planner"

using namespace scheduler;

namespace scheduler {

using arch_view::RoutingGraph;
using ResourceKind = RoutingGraph::ResourceNode::ResourceKind;

namespace {

/// Helper to validate FIFO slot index assignment
/// Checks that the assigned slot index matches the original result index
/// from the template transfer operation
void validateFifoSlotIndex(mlir::Value orig_fifo_value,
                           size_t assigned_slot_idx,
                           const PrivateResourceSpec* fifo_spec) {
  if (auto orig_private_result =
          mlir::dyn_cast<mlir::OpResult>(orig_fifo_value)) {
    unsigned orig_result_idx = orig_private_result.getResultNumber();
    auto priv_op = orig_fifo_value.getDefiningOp<mlir::ktdf::PrivateOp>();
    assert(priv_op);
    mlir::ktdf::PrivateYieldOp yield = priv_op.getYieldOp();
    assert(orig_result_idx < yield.getOperands().size());
    mlir::Value fifo_slot_value = yield.getOperands()[orig_result_idx];
    mlir::OpResult fifo_alloc_result =
        mlir::dyn_cast<mlir::OpResult>(fifo_slot_value);
    assert(fifo_alloc_result);

    assert(assigned_slot_idx == fifo_alloc_result.getResultNumber() &&
           "expected result index to match our calculated fifo slot index");
  }
  assert(assigned_slot_idx < fifo_spec->elements_per_slot.size() &&
         "Slot index should be within the allocated slots for this FIFO type");
}

/// Helper to check if a stage is an intermediate (synthetic) stage
inline bool isIntermediateStage(const StageNode* stage) {
  return stage->getOperation() == nullptr;
}

//===----------------------------------------------------------------------===//
// Debug Output Helpers
//===----------------------------------------------------------------------===//

static void debugPrintStageList(llvm::raw_ostream& os,
                                llvm::ArrayRef<StageNode*> stages,
                                const PathExpansionPlan* plan,
                                llvm::StringRef step_name) {
  os << "\n=== " << step_name << " ===\n";
  os << "Total stages: " << stages.size() << "\n";
  for (StageNode* stage : stages) {
    os << "  Stage " << stage->getStageId() << ": ";

    if (isIntermediateStage(stage))
      os << "(synthetic, unmaterialized)";
    else
      os << "(original)";

    if (plan && plan->stage_info.count(stage)) {
      const StageMaterializationInfo& info = plan->stage_info.at(stage);
      info.print(os);
    }
    os << "\n";
  }
}

static void debugPrintResourceSpecs(llvm::raw_ostream& os,
                                    const PrivateResourceFactory& factory) {
  os << "Private resources:\n";
  size_t mem_count = 0, fifo_count = 0;
  for (const auto& spec_ptr : factory.getSpecs()) {
    if (spec_ptr->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      mem_count++;
    } else if (spec_ptr->kind == PrivateResourceSpec::Kind::kFifo) {
      fifo_count++;
    }
  }
  os << "  Memory buffers: " << mem_count << "\n";
  os << "  FIFOs: " << fifo_count << "\n";
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Print Functions for Data Structures
//===----------------------------------------------------------------------===//

void PrivateResourceSpec::print(llvm::raw_ostream& os) const {
  switch (kind) {
    case Kind::kMemoryBuffer:
      os << "MemoryBuffer(resource=" << memory_resource << ", shape=[";
      llvm::interleave(shape, os, [&](int64_t d) { os << d; }, "x");
      os << "], element_type=" << element_type << ")";
      break;
    case Kind::kFifo:
      os << "Fifo(src=" << fifo_src << ", dest=" << fifo_dest << ", slots=[";
      llvm::interleave(
          elements_per_slot, os, [&](int64_t e) { os << e; }, ", ");
      os << "], element_type=" << element_type << ")";
      break;
    case Kind::kExistingValue:
      os << "ExistingValue(resource=" << memory_resource
         << ", type=" << existing_value.getType() << ")";
      break;
    case Kind::kUnknown:
      os << "Unknown";
      break;
  }
}

void PrivateResourceSpec::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void TransferMaterializationInfo::print(llvm::raw_ostream& os) const {
  os << "Transfer(" << source_resource << " -> " << dest_resource;

  if (source.resource) {
    os << ", priv_source=";
    source.resource->print(os);
  }

  if (dest.resource) {
    os << ", priv_dest=";
    dest.resource->print(os);
  }

  if (template_op) {
    os << ", template=";
    printLocation(os, template_op);
  } else {
    os << ", synthetic";
  }

  os << ")";
}

void TransferMaterializationInfo::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void TransferMaterializationInfo::printDebug(llvm::raw_ostream& os) const {
  os << source_resource << " -> " << dest_resource;

  if (source.resource) {
    os << " (source: ";
    if (source.resource->kind == PrivateResourceSpec::Kind::kFifo) {
      os << "FIFO slot " << source.slot_index;
    } else {
      os << "buffer";
    }
    os << ")";
  }

  if (dest.resource) {
    os << " (dest: ";
    if (dest.resource->kind == PrivateResourceSpec::Kind::kFifo) {
      os << "FIFO slot " << dest.slot_index;
    } else {
      os << "buffer";
    }
    os << ")";
  }
}

void StageMaterializationInfo::print(llvm::raw_ostream& os) const {
  os << " kind=";
  switch (kind) {
    case StageMaterializationInfo::Kind::kPreserveOriginal:
      os << "PreserveOriginal";
      break;
    case StageMaterializationInfo::Kind::kAdaptFifoKinds:
      os << "AdaptFifoKinds";
      break;
    case StageMaterializationInfo::Kind::kAdaptTransfer:
      os << "AdaptTransfer";
      break;
    case StageMaterializationInfo::Kind::kSyntheticTransfer:
      os << "SyntheticTransfer";
      break;
  }
  os << ", applicable_unit:";
  if (applicable_unit.has_value()) {
    if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(*applicable_unit)) {
      os << str_attr.getValue();
    } else {
      os << "<unknown>";
    }
  } else {
    os << "none";
  }
  os << ", transfers=" << transfers.size();

#if 1
  os << "\nTransfers:\n";
  for (const TransferMaterializationInfo* ti : transfers) {
    ti->print(os);
    os << "\n";
  }
#endif
}
void StageMaterializationInfo::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

/// Return the memory-space attribute from val's type if it is a MemRefType
/// with a memory space, otherwise return nullptr.
static ResourceType memrefMemorySpace(mlir::Value val) {
  if (auto mt = mlir::dyn_cast<mlir::MemRefType>(val.getType()))
    return llvm::cast<ResourceType>(mt.getMemorySpace());
  return nullptr;
}

namespace {
/// What all the data transfers of a stage have in common at one end, either
/// the end they read from or the end they write to.
struct SharedEnd {
  /// The memory that every transfer uses at this end, if they all use a
  /// memref in the same memory. Null otherwise.
  ResourceType memory;
  /// The load/store unit that every transfer uses at this end, if they all
  /// use a FIFO that names the same load/store unit on this stage's side.
  /// That is the FIFO's source when the stage writes into the FIFO, and the
  /// FIFO's destination when the stage reads from it. Null otherwise.
  ResourceType unit;
  /// True if every transfer uses a memref at this end, not a FIFO.
  bool all_memref = false;
};

/// What the data transfers of one stage have in common.
struct TransferEnds {
  /// Number of data transfers in the stage.
  size_t num_transfers = 0;
  /// The end that the transfers read from.
  SharedEnd source;
  /// The end that the transfers write to.
  SharedEnd dest;
};
}  // namespace

/// Find what the data transfers in \p stage_op have in common at the end they
/// read from and at the end they write to. See TransferEnds.
///
/// A FIFO only counts when its type names a load/store unit at the stage's
/// end. Input FIFOs often name a memory there instead, such as "DDR" in
/// "DDR -> SFU". That name is only a placeholder, which path expansion
/// replaces later, so it is not trusted to tell where the stage runs.
///
/// If the stage has indirect transfers, only those are looked at. The plain
/// transfer that fills the IAB next to them would otherwise add its "IAB"
/// destination. For an indirect transfer only the data operands count, not
/// the address operands.
static TransferEnds collectTransferEnds(mlir::ktdf::StageOp stage_op,
                                        const RoutingGraph& arch_graph) {
  // The source and destination operands of every transfer.
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> operands;
  stage_op.walk([&](mlir::ktdf::IndDataTransferOp op) {
    operands.push_back({op.getDirSrc(), op.getDirDst()});
  });
  if (operands.empty()) {
    stage_op.walk([&](mlir::ktdf::DataTransferOp op) {
      operands.push_back({op.getSource(), op.getDestination()});
    });
  }

  // The load/store unit that a FIFO operand names on this stage's side, or
  // null. That is the FIFO's destination when the stage reads from it, and
  // the FIFO's source when the stage writes into it.
  auto unitOf = [&](mlir::Value val, bool stage_reads) -> ResourceType {
    auto fifo = mlir::dyn_cast<mlir::ktdf::FifoSlotType>(val.getType());
    if (!fifo) return nullptr;
    auto end = llvm::dyn_cast<ResourceType>(stage_reads ? fifo.getDest()
                                                        : fifo.getSrc());
    if (!end) return nullptr;
    // A device may have several copies of a unit, for example one per
    // sub-core. They share a name, so every node with that name must be a
    // load/store unit.
    llvm::ArrayRef<RoutingGraph::NodeId> ids =
        arch_graph.getNodeIdsForResource(end);
    if (ids.empty() || llvm::any_of(ids, [&](auto id) {
          return arch_graph.getNode(id)->kind != ResourceKind::LoadStoreUnit;
        }))
      return nullptr;
    return end;
  };

  // Work out what every transfer has in common at one end. \p pick chooses
  // the source or the destination operand, and \p stage_reads says whether
  // the stage reads from that operand.
  auto sharedEnd = [&](auto pick, bool stage_reads) {
    SharedEnd shared;
    mlir::Value first = pick(operands.front());
    shared.memory = memrefMemorySpace(first);
    shared.unit = unitOf(first, stage_reads);
    shared.all_memref = true;
    for (const auto& ops : operands) {
      mlir::Value val = pick(ops);
      if (memrefMemorySpace(val) != shared.memory) shared.memory = nullptr;
      if (unitOf(val, stage_reads) != shared.unit) shared.unit = nullptr;
      if (!mlir::isa<mlir::MemRefType>(val.getType()))
        shared.all_memref = false;
    }
    return shared;
  };

  TransferEnds result;
  result.num_transfers = operands.size();
  if (operands.empty()) return result;
  result.source = sharedEnd([](const auto& ops) { return ops.first; },
                            /*stage_reads=*/true);
  result.dest = sharedEnd([](const auto& ops) { return ops.second; },
                          /*stage_reads=*/false);
  return result;
}

//===----------------------------------------------------------------------===//
// Stage Side Classification
//===----------------------------------------------------------------------===//

namespace {
/// Where an original stage sits relative to the compute stages of the
/// pipeline. Load-side stages bring data towards the compute stages and
/// store-side stages carry results away from them.
enum class StageSide { kLoad, kCompute, kStore };
}  // namespace

using StageSideMap = llvm::DenseMap<StageNode*, StageSide>;

/// Return true if \p stage is pinned to a compute unit in the input IR.
///
/// Only stages whose applicable unit is a Compute node in the routing graph
/// count. A stage pinned to a load/store unit is still an ordinary load-side
/// or store-side stage.
static bool isComputeAnchor(StageNode* stage, const RoutingGraph& arch_graph) {
  auto stage_op =
      mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
  if (!stage_op) return false;
  std::optional<mlir::ArrayAttr> units = stage_op.getApplicableUnits();
  if (!units || units->size() != 1) return false;
  auto unit = llvm::dyn_cast<ResourceType>(units->getValue()[0]);
  if (!unit) return false;
  auto node = arch_graph.getNode(unit);
  return node && node->kind == ResourceKind::Compute;
}

/// Decide, for every original stage, whether it is on the load side, is a
/// compute stage, or is on the store side. Fails if the stage graph does not
/// have a shape that path expansion can handle.
///
/// The stages do not have to form a single chain. Each side may branch and
/// merge freely, for example two load stages feeding one compute stage. What
/// matters is that every stage is clearly before or clearly after the compute
/// stages:
///
///  - There must be at least one compute stage.
///  - Every other stage must be upstream of a compute stage (load side) or
///    downstream of one (store side), directly or through other stages.
///  - No stage may be both upstream and downstream of compute stages, since
///    that would place it between two of them. Compute stages themselves may
///    feed each other directly or be unconnected.
///
/// How it works: walk the stages in topological order, marking as downstream
/// every stage fed by a compute stage or by a stage already marked
/// downstream. Then walk in reverse order to mark the upstream stages the
/// same way. Every non-compute stage must end up with exactly one mark.
static llvm::FailureOr<StageSideMap> classifyStageSides(
    llvm::ArrayRef<StageNode*> sorted_stages, const RoutingGraph& arch_graph) {
  llvm::SmallPtrSet<StageNode*, 4> anchors;
  for (StageNode* stage : sorted_stages)
    if (isComputeAnchor(stage, arch_graph)) anchors.insert(stage);
  if (anchors.empty()) {
    LDBG(1) << "Pipeline has no stage pinned to a compute unit";
    return mlir::failure();
  }

  // Forward walk: a stage comes after a compute stage if any of its
  // predecessors is a compute stage or comes after one.
  llvm::SmallPtrSet<StageNode*, 8> after_compute;
  for (StageNode* stage : sorted_stages) {
    if (!anchors.contains(stage) && !after_compute.contains(stage)) continue;
    for (StageNode* succ : stage->getDependencies()) after_compute.insert(succ);
  }

  // Backward walk: a stage comes before a compute stage if any of its
  // successors is a compute stage or comes before one.
  llvm::SmallPtrSet<StageNode*, 8> before_compute;
  for (StageNode* stage : llvm::reverse(sorted_stages)) {
    if (llvm::any_of(stage->getDependencies(), [&](StageNode* succ) {
          return anchors.contains(succ) || before_compute.contains(succ);
        }))
      before_compute.insert(stage);
  }

  StageSideMap sides;
  for (StageNode* stage : sorted_stages) {
    if (anchors.contains(stage)) {
      sides[stage] = StageSide::kCompute;
      continue;
    }
    bool is_before = before_compute.contains(stage);
    bool is_after = after_compute.contains(stage);
    if (is_before && is_after) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " sits between two compute stages";
      return mlir::failure();
    }
    if (!is_before && !is_after) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " is not connected to any compute stage";
      return mlir::failure();
    }
    sides[stage] = is_before ? StageSide::kLoad : StageSide::kStore;
  }
  return sides;
}

//===----------------------------------------------------------------------===//
// Main Planning Functions
//===----------------------------------------------------------------------===//

/// Topologically sort the pipeline stages, check that the stage graph has a
/// shape path expansion can handle, and work out which side of the compute
/// stages each stage is on. See classifyStageSides() for the rules.
static llvm::FailureOr<llvm::SmallVector<StageNode*>> sortAndValidateStages(
    PipelineTree& tree, PipelineNode* pipeline, const RoutingGraph& arch_graph,
    StageSideMap& sides) {
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(sorted_stages_or)) {
    LDBG(1) << "Failed to perform topological sort (cycle detected)";
    return mlir::failure();
  }

  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  llvm::FailureOr<StageSideMap> sides_or =
      classifyStageSides(sorted_stages, arch_graph);
  if (mlir::failed(sides_or)) {
    LDBG(1) << "Stages are not all clearly before or after the compute stages";
    return mlir::failure();
  }
  sides = std::move(*sides_or);

  return sorted_stages;
}

/// Return the memory that a load/store unit works on: the memory it reads
/// from for a load-side stage, or the memory it writes to for a store-side
/// stage. Returns null unless there is exactly one such memory.
///
/// A device may have several copies of the unit, for example one per
/// sub-core. All copies must work on the same memory.
static ResourceType memoryOfUnit(ResourceType unit, StageSide side,
                                 const RoutingGraph& arch_graph) {
  llvm::SmallSetVector<ResourceType, 2> memories;
  auto consider = [&](RoutingGraph::NodeId node_id) {
    auto node = arch_graph.getNode(node_id);
    if (node && node->kind == ResourceKind::Memory)
      memories.insert(node->resource);
  };
  for (RoutingGraph::NodeId unit_id : arch_graph.getNodeIdsForResource(unit)) {
    if (side == StageSide::kLoad) {
      // Memories with a datapath into the unit.
      for (RoutingGraph::NodeId node_id : arch_graph.getAllNodeIds())
        if (llvm::is_contained(arch_graph.getNeighbors(node_id), unit_id))
          consider(node_id);
    } else {
      // Memories the unit has a datapath into.
      for (RoutingGraph::NodeId node_id : arch_graph.getNeighbors(unit_id))
        consider(node_id);
    }
  }
  return memories.size() == 1 ? memories.front() : nullptr;
}

/// Work out the stage_resource of every original stage, and for some stages
/// also the unit they run on.
///
/// The stage_resource is the memory that a stage's load/store unit works on.
/// Path expansion uses it to find which memories and units the data passes
/// through between stages.
///
/// A stage may have several data transfers. Either they all read from the
/// same place, or they all write to the same place; the other ends may
/// differ. A place is either a memory, for a memref, or a load/store unit
/// named in a FIFO type. For example, a stage may copy DDR into two FIFOs,
/// or write into two FIFOs of type "L0LU -> PT" and "L0LU -> SFU", which
/// both start at L0LU.
///
/// Each stage is decided on its own. A stage pinned to a unit in the input IR
/// uses that unit as its resource. For any other stage, two things are worked
/// out separately:
///
///  - The memory. If all transfers read from one memory, or all write to one
///    memory, that memory is the resource. If both are true, as in a single
///    DDR -> L1 copy, the side of the pipeline decides: a load-side stage
///    uses the memory it reads from, and a store-side stage the memory it
///    writes to.
///  - The unit. If all transfers use FIFOs that name the same load/store unit
///    on this stage's side, the stage runs on that unit. The stage's side of
///    a FIFO is the FIFO's source when the stage writes into it, and the
///    FIFO's destination when the stage reads from it.
///
/// A stage may have a memory, a unit, or both. With only a unit, the resource
/// is the memory that unit works on. With both, that memory must be the same
/// as the one found from the memrefs. For example, a stage that reads an L1
/// memref into FIFOs of type "L1LU -> SFU" has memory L1 and unit L1LU,
/// which agree because L1LU reads from L1.
///
/// A stage that has neither a memory nor a unit is rejected.
///
/// Because the side of a stage comes from \p sides, the rules never look at
/// neighbouring stages. That is why the order of stages within a side does
/// not matter here.
///
/// A stage with no predecessors must read from memory, because nothing in the
/// pipeline could fill a FIFO it reads from. Likewise, a stage with no
/// successors must write to memory. Both are asserted.
static mlir::LogicalResult assignOriginalStageResources(
    llvm::ArrayRef<StageNode*> sorted_stages, const StageSideMap& sides,
    const RoutingGraph& arch_graph, PathExpansionPlan* plan) {
  // Stages that some other stage feeds into.
  llvm::SmallPtrSet<StageNode*, 8> has_predecessor;
  for (StageNode* stage : sorted_stages)
    has_predecessor.insert(stage->getDependencies().begin(),
                           stage->getDependencies().end());

  // stage_info entries are default-constructed on first access (operator[]).
  for (StageNode* stage : sorted_stages) {
    StageMaterializationInfo& info = plan->stage_info[stage];

    auto stage_op =
        mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
    assert(stage_op && "expected operation to be valid for original stages");

    // A pinned stage: the unit is already known from the input IR.
    if (auto units = stage_op.getApplicableUnits()) {
      assert(units->size() == 1 &&
             "path expansion currently does not handle nested pipelines with "
             "multi-unit stages");
      info.stage_resource = llvm::cast<ResourceType>(units->getValue()[0]);
      continue;
    }

    TransferEnds ends = collectTransferEnds(stage_op, arch_graph);
    assert(ends.num_transfers > 0 &&
           "Unable to determine stage_resource for original stage: it has no "
           "data transfer");
    assert((has_predecessor.contains(stage) || ends.source.all_memref) &&
           "A stage with no predecessors must have a memref source on every "
           "data_transfer");
    assert((!stage->getDependencies().empty() || ends.dest.all_memref) &&
           "A stage with no successors must have a memref dest on every "
           "data_transfer");

    // Compute stages are always pinned, so they were handled by rule 1.
    StageSide side = sides.at(stage);
    assert(side != StageSide::kCompute &&
           "Compute stages should be pinned to a unit");

    // The memory shared by all transfers, if any.
    ResourceType memory;
    if (ends.source.memory && ends.dest.memory)
      memory = side == StageSide::kLoad ? ends.source.memory : ends.dest.memory;
    else
      memory = ends.source.memory ? ends.source.memory : ends.dest.memory;

    // The load/store unit shared by all transfers, if any.
    ResourceType unit = ends.source.unit ? ends.source.unit : ends.dest.unit;
    if (ends.source.unit && ends.dest.unit &&
        ends.source.unit != ends.dest.unit)
      return stage_op.emitError(
                 "path-expansion: the stage's FIFOs place it "
                 "on two different units, ")
             << ends.source.unit << " and " << ends.dest.unit;

    if (!memory && !unit)
      return stage_op.emitError(
          "path-expansion: the data transfers of a stage must all read from "
          "the same place or all write to the same place");

    if (unit) {
      ResourceType unit_memory = memoryOfUnit(unit, side, arch_graph);
      if (!unit_memory)
        return stage_op.emitError(
                   "path-expansion: cannot tell which memory unit ")
               << unit << " works on for this stage";
      if (memory && memory != unit_memory)
        return stage_op.emitError("path-expansion: the stage uses memory ")
               << memory << ", but its FIFOs place it on unit " << unit
               << ", which works on " << unit_memory;
      memory = unit_memory;
      info.applicable_unit = unit;
    }
    info.stage_resource = memory;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Data Flows Between Stages
//===----------------------------------------------------------------------===//

namespace {
/// One end (producer or consumer) of a FIFO slot inside an original stage.
struct FifoEndpoint {
  mlir::Operation* op = nullptr;
  StageNode* stage = nullptr;
  // Whether the FIFO slot is the op's source operand (always true for
  // read_from_fifo, always false for write_to_fifo).
  bool fifo_is_source = false;
};

/// The producer and consumer of every FIFO slot used by the original stages.
struct FifoEndpoints {
  llvm::MapVector<mlir::Value, FifoEndpoint> producers;
  llvm::MapVector<mlir::Value, FifoEndpoint> consumers;
  // FIFO slots used by an indirect data transfer.
  llvm::SmallPtrSet<mlir::Value, 4> indirect_slots;
  // The first op found to be a second producer or consumer of a FIFO slot,
  // and whether it is a producer. Path expansion cannot handle such a slot.
  std::optional<std::pair<mlir::Operation*, bool>> duplicate;
};
}  // namespace

/// Find the op that writes and the op that reads each FIFO slot used by the
/// original stages. Only the first writer and the first reader of a slot are
/// kept. A slot with more than one is noted in FifoEndpoints::duplicate,
/// which is only an error if the pipeline needs expanding.
static FifoEndpoints collectFifoEndpoints(llvm::ArrayRef<StageNode*> stages) {
  FifoEndpoints endpoints;
  for (StageNode* stage : stages) {
    if (isIntermediateStage(stage)) continue;
    auto stage_op = mlir::cast<mlir::ktdf::StageOp>(stage->getOperation());

    auto record = [&](llvm::MapVector<mlir::Value, FifoEndpoint>& ends,
                      mlir::Value fifo_slot, mlir::Operation* op,
                      bool fifo_is_source) {
      if (!ends.insert({fifo_slot, {op, stage, fifo_is_source}}).second &&
          !endpoints.duplicate)
        endpoints.duplicate = {op, &ends == &endpoints.producers};
    };

    stage_op.walk([&](mlir::Operation* op) {
      if (auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op))
        return record(endpoints.consumers, read_op.getFifoSlot(), op, true);
      if (auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op))
        return record(endpoints.producers, write_op.getFifoSlot(), op, false);
      if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
        if (mlir::isa<mlir::ktdf::FifoSlotType>(transfer.getSource().getType()))
          record(endpoints.consumers, transfer.getSource(), op, true);
        if (mlir::isa<mlir::ktdf::FifoSlotType>(
                transfer.getDestination().getType()))
          record(endpoints.producers, transfer.getDestination(), op, false);
        return;
      }
      if (auto ind_transfer =
              mlir::dyn_cast<mlir::ktdf::IndDataTransferOp>(op)) {
        bool is_gather = ind_transfer.isGather();
        mlir::Value fifo_side =
            is_gather ? ind_transfer.getDirDst() : ind_transfer.getDirSrc();
        if (!mlir::isa<mlir::ktdf::FifoSlotType>(fifo_side.getType())) return;
        endpoints.indirect_slots.insert(fifo_side);
        if (is_gather)
          record(endpoints.producers, fifo_side, op, false);
        else
          record(endpoints.consumers, fifo_side, op, true);
      }
    });
  }
  return endpoints;
}

/// Return the fifo.allocate result that \p fifo_slot (a ktdf.private result)
/// is yielded from, or a null result if it cannot be traced.
static mlir::OpResult getFifoAllocateResult(mlir::Value fifo_slot) {
  auto private_result = mlir::dyn_cast<mlir::OpResult>(fifo_slot);
  if (!private_result) return {};
  auto priv_op =
      mlir::dyn_cast<mlir::ktdf::PrivateOp>(private_result.getOwner());
  if (!priv_op) return {};
  mlir::Value yielded =
      priv_op.getYieldOp().getOperands()[private_result.getResultNumber()];
  auto alloc_result = mlir::dyn_cast<mlir::OpResult>(yielded);
  if (!alloc_result ||
      !mlir::isa<mlir::ktdf::FifoAllocateOp>(alloc_result.getOwner()))
    return {};
  return alloc_result;
}

namespace {
/// What holds a flow's data at one boundary (see DataFlow): a buffer, FIFO
/// slot or original memref, and how a transfer addresses it.
using Carrier = TransferSide;

/// Data that one original stage passes to another, and the route it takes.
///
/// The data is carried by a FIFO slot or a memref, the carrier. One op of the
/// producer stage writes the carrier, one op of the consumer stage reads it,
/// and the producer stage has a dependency on the consumer stage.
///
/// The route is a routing-graph path from the producer's resource to the
/// consumer's resource. Step 1 splits the middle of the path into hops and
/// gives each hop to the stage that runs on its unit, or creates a new
/// (synthetic) stage for it. Those hop stages form the flow's chain.
///
/// Each stage on the route, producer -> chain -> consumer, works on one
/// memory through one unit: its stage_resource and its applicable unit. On
/// the path the two are next to each other. A load unit reads its memory, so
/// the memory comes first, as in L1 -> L1LU. A store unit writes its memory,
/// so the unit comes first, as in L1SU -> L1. A compute stage, or a hop on a
/// unit with no memory, is just its unit. At the two ends of the path a stage
/// may appear only in part, since the path starts at the producer's resource
/// and ends at the consumer's resource.
///
/// A boundary is where the data leaves one stage on the route and enters the
/// next. Something must hold the data there:
///  - a buffer in the receiving stage's memory, if that stage loads from it;
///  - a buffer in the sending stage's memory, if that stage stores into it;
///  - otherwise the two units are next to each other, and a FIFO from the
///    sending unit to the receiving unit.
/// With n hop stages there are n + 1 boundaries.
///
/// For example, take DDR -> MNILU -> L1 -> L1LU -> SFU from a load stage to a
/// compute stage. The load stage is DDR read by MNILU, the hop stage is L1
/// read by L1LU, and the compute stage is SFU. Boundary 0, from the load
/// stage to the hop, is held in an L1 buffer, because the hop loads from L1.
/// Boundary 1, from the hop to SFU, is held in an "L1LU" -> "SFU" FIFO.
struct DataFlow {
  mlir::Value carrier;
  mlir::Operation* producer_op = nullptr;
  mlir::Operation* consumer_op = nullptr;
  StageNode* producer = nullptr;
  StageNode* consumer = nullptr;

  // How the producer op fills the carrier: Copy unless it is a data transfer
  // that splats or pads. Once the flow has hops, the mode belongs to the first
  // transfer that writes a FIFO (see assignTransferMode).
  mlir::ktdf::TransferMode transfer_mode = mlir::ktdf::TransferMode::Copy;

  RoutingGraph::Path path;
  // Last path index that is part of the producer.
  size_t producer_end = 0;
  // First path index that is part of the consumer.
  size_t consumer_begin = 0;
  // Load/store units the path gives the producer and the consumer, if any.
  ResourceType producer_unit;
  ResourceType consumer_unit;

  // The hop stages, in data-flow order. Set by Step 1.
  llvm::SmallVector<StageNode*, 2> chain;
  // The boundaries of the route. boundaries[k] is the path index of the
  // first node of the stage that receives the data at boundary k. Boundary 0
  // is from the producer to chain.front(), and boundary chain.size() is from
  // chain.back() to the consumer. Empty when the chain is empty. Set by
  // Step 1.
  llvm::SmallVector<size_t, 3> boundaries;
  // What carries the data across each boundary. Steps 2 and 3 fill it in.
  llvm::SmallVector<Carrier, 3> carriers;

  bool carrierIsMemref() const {
    return mlir::isa<mlir::MemRefType>(carrier.getType());
  }

  /// The range [middleBegin(), middleEnd()) of path indices that need hop
  /// stages. It is empty when the two stages own the whole path.
  size_t middleBegin() const { return producer_end + 1; }
  size_t middleEnd() const { return consumer_begin; }
};
}  // namespace

/// Return the routing-graph node at index \p idx of \p flow's path.
static RoutingGraph::ResourceNode pathNode(const RoutingGraph& arch_graph,
                                           const DataFlow& flow, size_t idx) {
  std::optional<RoutingGraph::ResourceNode> node =
      arch_graph.getNode(flow.path[idx]);
  assert(node && "node in path must exist in graph");
  return *node;
}

/// Find the data each original stage passes to the stages it has a
/// dependency on. See DataFlow.
///
/// A FIFO slot is a carrier when its producer and its consumer are both in
/// the pipeline. A memref is a carrier when, within the pipeline, exactly one
/// data transfer writes it, exactly one data transfer in another stage reads
/// it, and nothing else uses it. Other memrefs, such as the pipeline's inputs
/// and outputs, are not passed between stages. A dependency that no carrier
/// follows only orders the two stages, and gets no flow.
///
/// How it works: first record every use of every memref in the pipeline.
/// Then walk the ops of each stage in order, and pair each carrier an op
/// writes with the op that reads it. The flows are grouped by producer, in
/// topological order, and then by consumer, in the order of the producer's
/// dependencies. This keeps new stages and transfers in a stable order.
static std::vector<DataFlow> collectDataFlows(
    llvm::ArrayRef<StageNode*> sorted_stages, const FifoEndpoints& endpoints) {
  using OpInStage = std::pair<mlir::Operation*, StageNode*>;
  struct MemrefUses {
    llvm::SmallVector<OpInStage, 1> writers;
    llvm::SmallVector<OpInStage, 1> readers;
    bool other_use = false;
  };
  llvm::DenseMap<mlir::Value, MemrefUses> memref_uses;
  for (StageNode* stage : sorted_stages) {
    stage->getOperation()->walk([&](mlir::Operation* op) {
      auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op);
      for (mlir::Value value : op->getOperands()) {
        if (!mlir::isa<mlir::MemRefType>(value.getType())) continue;
        MemrefUses& uses = memref_uses[value];
        if (transfer && value == transfer.getDestination())
          uses.writers.push_back({op, stage});
        else if (transfer && value == transfer.getSource())
          uses.readers.push_back({op, stage});
        else
          uses.other_use = true;
      }
    });
  }

  // The op that reads \p carrier, written by \p writer, and its stage.
  auto findReader = [&](mlir::Value carrier,
                        mlir::Operation* writer) -> OpInStage {
    if (mlir::isa<mlir::ktdf::FifoSlotType>(carrier.getType())) {
      auto producer_it = endpoints.producers.find(carrier);
      auto consumer_it = endpoints.consumers.find(carrier);
      if (producer_it == endpoints.producers.end() ||
          producer_it->second.op != writer ||
          consumer_it == endpoints.consumers.end())
        return {};
      return {consumer_it->second.op, consumer_it->second.stage};
    }
    auto uses_it = memref_uses.find(carrier);
    if (uses_it == memref_uses.end()) return {};
    const MemrefUses& uses = uses_it->second;
    if (uses.writers.size() != 1 || uses.readers.size() != 1 || uses.other_use)
      return {};
    return uses.readers.front();
  };

  std::vector<DataFlow> flows;
  for (StageNode* producer : sorted_stages) {
    // The flows out of this stage, in the order of the ops that write them.
    std::vector<DataFlow> outgoing;
    producer->getOperation()->walk([&](mlir::Operation* op) {
      mlir::Value written;
      if (auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op))
        written = write_op.getFifoSlot();
      else if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op))
        written = transfer.getDestination();
      else if (auto ind_transfer =
                   mlir::dyn_cast<mlir::ktdf::IndDataTransferOp>(op))
        if (ind_transfer.isGather()) written = ind_transfer.getDirDst();
      if (!written) return;

      auto [reader, consumer] = findReader(written, op);
      if (!consumer || consumer == producer) return;
      DataFlow flow;
      flow.carrier = written;
      flow.producer_op = op;
      flow.consumer_op = reader;
      flow.producer = producer;
      flow.consumer = consumer;
      if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op))
        flow.transfer_mode = transfer.getTransferMode();
      outgoing.push_back(std::move(flow));
    });

    llvm::SmallPtrSet<StageNode*, 4> seen;
    for (StageNode* consumer : producer->getDependencies()) {
      if (!seen.insert(consumer).second) continue;
      for (const DataFlow& flow : outgoing)
        if (flow.consumer == consumer) flows.push_back(flow);
    }
  }
  return flows;
}

/// Find the routing-graph path of every flow, and work out which of its nodes
/// are part of the producer and which are part of the consumer.
///
/// The path starts at the producer's resource and ends at the consumer's
/// resource. A load-side producer's memory is read by a load unit, which
/// comes right after the memory on the path; that unit is part of the
/// producer too. A store-side consumer's memory is written by a store unit,
/// which comes right before the memory; that unit is part of the consumer
/// too. These units later become the stages' applicable units. See DataFlow
/// for how stages, memories and units line up on a path.
///
/// For example, take a load stage reading DDR that feeds a compute stage on
/// SFU, with the path DDR -> MNILU -> L1 -> L1LU -> SFU. The load stage is
/// DDR read by MNILU, the compute stage is SFU, and L1 -> L1LU in the middle
/// is a hop that needs a stage on L1LU.
///
/// The path depends only on the resources of the two stages, so all flows
/// between the same two stages get the same path. Fails if some flow has no
/// path in the routing graph.
static mlir::LogicalResult routeDataFlows(llvm::MutableArrayRef<DataFlow> flows,
                                          const StageSideMap& sides,
                                          const RoutingGraph& arch_graph,
                                          const PathExpansionPlan* plan) {
  auto nodeAt = [&](const DataFlow& flow, size_t idx) {
    return pathNode(arch_graph, flow, idx);
  };

  for (DataFlow& flow : flows) {
    std::optional<RoutingGraph::Path> path = arch_graph.findShortestPath(
        arch_graph.getNodeIdForResource(
            plan->stage_info.at(flow.producer).stage_resource),
        arch_graph.getNodeIdForResource(
            plan->stage_info.at(flow.consumer).stage_resource));
    if (!path || path->empty()) {
      LDBG(1) << "No routing path from stage " << flow.producer->getStageId()
              << " to stage " << flow.consumer->getStageId();
      return mlir::failure();
    }
    flow.path = std::move(*path);

    size_t last = flow.path.size() - 1;
    flow.consumer_begin = last;

    // The load unit right after a load-side producer's memory is part of the
    // producer, unless that unit is the consumer's own resource.
    if (sides.at(flow.producer) == StageSide::kLoad && last > 1 &&
        nodeAt(flow, 0).kind == ResourceKind::Memory &&
        nodeAt(flow, 1).kind == ResourceKind::LoadStoreUnit) {
      flow.producer_end = 1;
      flow.producer_unit = nodeAt(flow, 1).resource;
    }

    // The store unit right before a store-side consumer's memory is part of
    // the consumer, unless it is already part of the producer.
    if (sides.at(flow.consumer) == StageSide::kStore &&
        last > flow.producer_end + 1 &&
        nodeAt(flow, last).kind == ResourceKind::Memory &&
        nodeAt(flow, last - 1).kind == ResourceKind::LoadStoreUnit) {
      flow.consumer_begin = last - 1;
      flow.consumer_unit = nodeAt(flow, last - 1).resource;
    }
  }
  return mlir::success();
}

/// Return true if path expansion has to add stages: some path passes through
/// a memory or compute unit that is part of neither the producer nor the
/// consumer. A load/store unit with no stage does not count on its own.
static bool needsExpansion(llvm::ArrayRef<DataFlow> flows,
                           const RoutingGraph& arch_graph) {
  for (const DataFlow& flow : flows) {
    for (size_t i = flow.middleBegin(); i < flow.middleEnd(); ++i)
      if (pathNode(arch_graph, flow, i).kind != ResourceKind::LoadStoreUnit)
        return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Step 1: Add Stages Along Each Path
//===----------------------------------------------------------------------===//

/// Set the applicable unit of every original stage, and check that no two of
/// them run on the same unit. A pipeline can run only one stage on each unit,
/// so an input where two stages end up on the same unit is invalid.
///
/// A stage pinned to a compute or load/store unit runs on that unit. Any other
/// stage gets the load/store unit its flows give it (see routeDataFlows). A
/// stage with several flows on the same side must get the same unit from all
/// of them.
static mlir::LogicalResult assignOriginalStageUnits(
    llvm::ArrayRef<DataFlow> flows, llvm::ArrayRef<StageNode*> sorted_stages,
    const RoutingGraph& arch_graph, PathExpansionPlan* plan) {
  // A stage whose resource is a compute or load/store unit runs on it.
  for (StageNode* stage : sorted_stages) {
    StageMaterializationInfo& info = plan->getStageInfo(stage);
    auto node = arch_graph.getNode(info.stage_resource);
    if (node && node->kind != ResourceKind::Memory)
      info.applicable_unit = info.stage_resource;
  }

  auto setUnit = [&](StageNode* stage,
                     ResourceType unit) -> mlir::LogicalResult {
    if (!unit) return mlir::success();
    std::optional<ResourceType>& current =
        plan->getStageInfo(stage).applicable_unit;
    if (current && *current != unit)
      return stage->getOperation()->emitError(
                 "path-expansion: stage is reached through two different "
                 "units, ")
             << *current << " and " << unit;
    current = unit;
    return mlir::success();
  };
  for (const DataFlow& flow : flows) {
    if (mlir::failed(setUnit(flow.producer, flow.producer_unit)) ||
        mlir::failed(setUnit(flow.consumer, flow.consumer_unit)))
      return mlir::failure();
  }

  llvm::DenseMap<ResourceType, StageNode*> unit_owner;
  for (StageNode* stage : sorted_stages) {
    std::optional<ResourceType> unit =
        plan->getStageInfo(stage).applicable_unit;
    if (!unit) continue;
    auto [it, inserted] = unit_owner.try_emplace(*unit, stage);
    if (inserted) continue;
    mlir::InFlightDiagnostic diag =
        stage->getOperation()->emitError("path-expansion: stage runs on unit ")
        << *unit << ", which another stage of the pipeline already runs on";
    diag.attachNote(it->second->getOperation()->getLoc())
        << "other stage on unit " << *unit;
    return diag;
  }
  return mlir::success();
}

/// Make each flow that got hop stages pass through them: the dependency
/// producer -> consumer becomes producer -> hop stages -> consumer. The
/// direct dependency is kept only if some other flow between the same two
/// stages needs no hops.
static void rewireDependencies(llvm::ArrayRef<DataFlow> flows) {
  llvm::MapVector<std::pair<StageNode*, StageNode*>,
                  llvm::SmallVector<const DataFlow*, 2>>
      flows_by_pair;
  for (const DataFlow& flow : flows)
    flows_by_pair[{flow.producer, flow.consumer}].push_back(&flow);

  auto addDependency = [](StageNode* from, StageNode* to) {
    if (!llvm::is_contained(from->getDependencies(), to))
      from->addDependency(to);
  };
  for (const auto& [pair, pair_flows] : flows_by_pair) {
    auto [producer, consumer] = pair;
    auto hasChain = [](const DataFlow* flow) { return !flow->chain.empty(); };
    if (llvm::none_of(pair_flows, hasChain)) continue;
    if (llvm::all_of(pair_flows, hasChain)) {
      producer->nullifyDependency(consumer);
      producer->removeNullifiedDependencies();
    }
    for (const DataFlow* flow : pair_flows) {
      if (!hasChain(flow)) continue;
      addDependency(producer, flow->chain.front());
      for (size_t k = 0; k + 1 < flow->chain.size(); ++k)
        addDependency(flow->chain[k], flow->chain[k + 1]);
      addDependency(flow->chain.back(), consumer);
    }
  }
}

/// Step 1: Add the stages that move data through memories and units that have
/// no stage of their own on a path.
///
/// How it works, for each flow:
///  1. The middle of the path is split into hops. A memory followed by the
///     load unit that reads it, or a store unit followed by the memory it
///     writes, is one hop that moves data through that memory. A load/store
///     unit on its own is a hop with no memory. The first path index of each
///     hop and of the consumer is recorded as a boundary (see DataFlow).
///  2. Each hop is given to the stage that already runs on its unit, so that
///     no unit ever runs two stages. Only if no stage runs on the unit yet is
///     a new stage created for it. The existing stage may be an original
///     stage or a new stage made for another flow, and it must use the same
///     memory as the hop.
///
/// Then the dependencies are rewired through the hop stages (see
/// rewireDependencies).
///
/// Giving a hop to an existing stage makes that stage depend on the producer.
/// If the existing stage is upstream of the producer, that creates a cycle;
/// the caller detects it when it sorts the stages again.
///
/// New stages are placed in the pipeline right after their producer (after
/// any new stages already placed there), which keeps the printed order close
/// to the data-flow order.
static mlir::LogicalResult expandDataFlows(
    PipelineTree& tree, PipelineNode* pipeline,
    llvm::MutableArrayRef<DataFlow> flows,
    llvm::ArrayRef<StageNode*> sorted_stages, const RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  // The stage running on each unit. assignOriginalStageUnits() made sure the
  // original stages all run on different units.
  llvm::DenseMap<ResourceType, StageNode*> unit_owner;
  for (StageNode* stage : sorted_stages)
    if (std::optional<ResourceType> unit =
            plan->getStageInfo(stage).applicable_unit)
      unit_owner[*unit] = stage;

  // The stage after which the next new stage following a producer goes.
  llvm::DenseMap<StageNode*, StageNode*> insert_after;

  for (DataFlow& flow : flows) {
    llvm::SmallVector<StageNode*, 2>& chain = flow.chain;
    StageNode*& last_inserted =
        insert_after.try_emplace(flow.producer, flow.producer).first->second;

    // Find or create the stage for the hop through memory \p memory (may be
    // null) on unit \p unit.
    auto addHop = [&](ResourceType memory,
                      ResourceType unit) -> mlir::LogicalResult {
      auto owner_it = unit_owner.find(unit);
      if (owner_it == unit_owner.end()) {
        StageNode* synth = tree.createStageNode(nullptr, next_stage_id++);
        StageMaterializationInfo info;
        info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
        info.stage_resource = memory;
        info.applicable_unit = unit;
        plan->stage_info[synth] = info;
        pipeline->insertChildNode(synth, last_inserted);
        last_inserted = synth;
        unit_owner[unit] = synth;
        chain.push_back(synth);
        return mlir::success();
      }

      StageNode* owner = owner_it->second;
      if (owner == flow.producer || owner == flow.consumer ||
          llvm::is_contained(chain, owner)) {
        LDBG(1) << "Path from stage " << flow.producer->getStageId()
                << " to stage " << flow.consumer->getStageId()
                << " passes through unit " << unit
                << " twice, but a unit can run only one stage";
        return mlir::failure();
      }
      if (plan->getStageInfo(owner).stage_resource != memory) {
        LDBG(1) << "Path from stage " << flow.producer->getStageId()
                << " to stage " << flow.consumer->getStageId() << " needs unit "
                << unit << " for memory " << memory << ", but stage "
                << owner->getStageId() << " already uses it for memory "
                << plan->getStageInfo(owner).stage_resource;
        return mlir::failure();
      }
      chain.push_back(owner);
      return mlir::success();
    };

    size_t end = flow.middleEnd();
    flow.boundaries.push_back(flow.middleBegin());
    for (size_t i = flow.middleBegin(); i < end;) {
      RoutingGraph::ResourceNode node = pathNode(arch_graph, flow, i);
      std::optional<RoutingGraph::ResourceNode> next;
      if (i + 1 < end) next = pathNode(arch_graph, flow, i + 1);

      // The hop starting at path index i: its memory (may be null), its unit,
      // and the number of path nodes it covers.
      ResourceType memory, unit;
      size_t width = 1;
      switch (node.kind) {
        case ResourceKind::Memory:
          // A memory with no stage must be followed by the unit that reads
          // it.
          if (!next || next->kind != ResourceKind::LoadStoreUnit) {
            LDBG(1) << "Memory " << node.resource << " between stage "
                    << flow.producer->getStageId() << " and stage "
                    << flow.consumer->getStageId()
                    << " has no load unit after it";
            return mlir::failure();
          }
          memory = node.resource;
          unit = next->resource;
          width = 2;
          break;
        case ResourceKind::LoadStoreUnit:
          unit = node.resource;
          if (next && next->kind == ResourceKind::Memory) {
            memory = next->resource;
            width = 2;
          }
          break;
        case ResourceKind::Compute:
          // Hop stages only move data between memories through load/store
          // units. Passing data through a compute unit would need a compute
          // stage that relays it, which path expansion does not create.
          LDBG(1) << "Path from stage " << flow.producer->getStageId()
                  << " to stage " << flow.consumer->getStageId()
                  << " passes through compute unit " << node.resource
                  << ", which has no stage";
          return mlir::failure();
      }
      if (mlir::failed(addHop(memory, unit))) return mlir::failure();
      i += width;
      flow.boundaries.push_back(i);
    }
    if (chain.empty()) flow.boundaries.clear();
  }

  rewireDependencies(flows);
  return mlir::success();
}

/// Put the pipeline's stages in an order where every stage comes after the
/// stages it depends on, changing the current order as little as possible.
///
/// Giving a hop to an existing stage can make it depend on a stage placed
/// after it. The materializer emits stages in pipeline order, so this keeps
/// the output in data-flow order. How it works: repeatedly take the first
/// stage, in the current order, whose predecessors have all been placed.
/// Fails if the dependencies form a cycle.
static mlir::LogicalResult orderStagesByDependencies(PipelineNode* pipeline) {
  llvm::SmallVector<StageNode*> stages = pipeline->getStages();
  if (stages.empty()) return mlir::success();

  llvm::DenseMap<StageNode*, unsigned> unplaced_preds;
  for (StageNode* stage : stages)
    for (StageNode* succ : stage->getDependencies()) ++unplaced_preds[succ];

  llvm::SmallVector<StageNode*> ordered;
  llvm::SmallPtrSet<StageNode*, 8> placed;
  while (ordered.size() < stages.size()) {
    auto next = llvm::find_if(stages, [&](StageNode* stage) {
      return !placed.contains(stage) && unplaced_preds.lookup(stage) == 0;
    });
    if (next == stages.end()) return mlir::failure();
    placed.insert(*next);
    ordered.push_back(*next);
    for (StageNode* succ : (*next)->getDependencies()) --unplaced_preds[succ];
  }
  if (ordered == stages) return mlir::success();

  // Re-link the stages in the new order where the first one used to be.
  OperationTreeNode* pos = stages.front()->getPrevSibling();
  for (StageNode* stage : stages) stage->unlink();
  for (StageNode* stage : ordered) {
    if (pos)
      pipeline->insertChildNode(stage, pos);
    else
      pipeline->insertAsFirstChild(stage);
    pos = stage;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 2: Adapt the Ends of Each Flow
//===----------------------------------------------------------------------===//

namespace {
/// Where a flow's data is held at one boundary (see DataFlow): a buffer in a
/// memory, or a FIFO between two units. Exactly one of memory and
/// fifo_src/fifo_dest is set.
struct BoundaryPlace {
  ResourceType memory;
  ResourceType fifo_src;
  ResourceType fifo_dest;
};

/// The shape and element type of the data a flow moves.
struct Tile {
  llvm::SmallVector<int64_t> shape;
  mlir::Type element_type;

  int64_t numElements() const {
    int64_t n = 1;
    for (int64_t dim : shape) n *= dim;
    return n;
  }
};
}  // namespace

/// Return where \p flow's data is held at boundary \p k. See DataFlow.
///
/// How it works: the receiving stage starts at path index boundaries[k], and
/// the sending stage ends just before it. If the receiving stage starts with
/// its memory, it loads from that memory, and the data is held in a buffer
/// there. Otherwise, if the sending stage ends with its memory, it stores
/// into that memory, and the data is held in a buffer there. Otherwise the
/// sending stage ends with its unit and the receiving stage starts with its
/// unit, and the data is held in a FIFO between the two.
static BoundaryPlace boundaryPlace(const DataFlow& flow, size_t k,
                                   const RoutingGraph& arch_graph) {
  size_t idx = flow.boundaries[k];
  RoutingGraph::ResourceNode sender_last = pathNode(arch_graph, flow, idx - 1);
  RoutingGraph::ResourceNode receiver_first = pathNode(arch_graph, flow, idx);
  if (receiver_first.kind == ResourceKind::Memory)
    return {receiver_first.resource, {}, {}};
  if (sender_last.kind == ResourceKind::Memory)
    return {sender_last.resource, {}, {}};
  return {{}, sender_last.resource, receiver_first.resource};
}

/// Return the shape of the data that \p op moves on one side, the side that
/// path expansion is about to replace (its source if \p replaces_source).
/// Fails if the shape is not static.
///
/// For a data transfer, the shape is taken from its memref side, since the
/// FIFO side may give only a number of elements. For an indirect transfer,
/// it is taken from the side being replaced.
static llvm::FailureOr<Tile> tileOf(mlir::Operation* op, bool replaces_source) {
  mlir::Value side;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
    bool use_source = replaces_source;
    mlir::Value replaced =
        replaces_source ? transfer.getSource() : transfer.getDestination();
    if (!mlir::isa<mlir::MemRefType>(replaced.getType()))
      use_source = !use_source;
    side = use_source ? transfer.getSource() : transfer.getDestination();
    sizes = use_source ? transfer.getMixedSourceSizes()
                       : transfer.getMixedDestSizes();
  } else {
    auto ind_transfer = mlir::cast<mlir::ktdf::IndDataTransferOp>(op);
    side =
        replaces_source ? ind_transfer.getDirSrc() : ind_transfer.getDirDst();
    sizes = replaces_source ? ind_transfer.getMixedDirSrcSizes()
                            : ind_transfer.getMixedDirDstSizes();
  }

  Tile tile;
  if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(side.getType()))
    tile.element_type = memref_type.getElementType();
  else
    tile.element_type =
        mlir::cast<mlir::ktdf::FifoSlotType>(side.getType()).getElementType();
  for (mlir::OpFoldResult size : sizes) {
    std::optional<int64_t> dim = mlir::getConstantIntValue(size);
    if (!dim) return mlir::failure();
    tile.shape.push_back(*dim);
  }
  return tile;
}

/// Return the shape of the data \p flow moves, for buffers and FIFOs between
/// two hop stages. It comes from the data transfer at the producer end, or
/// else at the consumer end. If both ends are FIFO ops, it is the size of the
/// FIFO slot.
static llvm::FailureOr<Tile> flowTile(const DataFlow& flow) {
  auto isTransfer = [](mlir::Operation* op) {
    return mlir::isa<mlir::ktdf::DataTransferOp, mlir::ktdf::IndDataTransferOp>(
        op);
  };
  if (isTransfer(flow.producer_op))
    return tileOf(flow.producer_op, /*replaces_source=*/false);
  if (isTransfer(flow.consumer_op))
    return tileOf(flow.consumer_op, /*replaces_source=*/true);
  auto fifo_type = mlir::cast<mlir::ktdf::FifoSlotType>(flow.carrier.getType());
  if (fifo_type.isDynamicNumElements()) return mlir::failure();
  return Tile{{fifo_type.getStaticNumElements()}, fifo_type.getElementType()};
}

namespace {
/// Creates the FIFOs that carry flows across boundaries.
///
/// The FIFOs of flows whose carrier is a FIFO slot are grouped like the
/// original slots: slots of one fifo.allocate that cross the same pair of
/// units go into one new fifo.allocate, in the same order. A flow whose
/// carrier is a memref gets a FIFO of its own.
class BoundaryFifos {
 public:
  explicit BoundaryFifos(PathExpansionPlan* plan) : plan_(plan) {}

  /// Return the FIFO and slot that carry \p flow from \p src to \p dest. If
  /// \p mirror_slot is set, the slot must have the same index as the
  /// carrier's slot in its original fifo.allocate.
  std::pair<PrivateResourceSpec*, size_t> get(const DataFlow& flow,
                                              ResourceType src,
                                              ResourceType dest,
                                              const Tile& tile,
                                              bool mirror_slot) {
    if (flow.carrierIsMemref())
      return {plan_->resource_factory.createFifo(
                  src, dest, {tile.numElements()}, tile.element_type),
              0};

    auto fifo_type =
        mlir::cast<mlir::ktdf::FifoSlotType>(flow.carrier.getType());
    assert(!fifo_type.isDynamicNumElements() &&
           "Dynamic FIFO sizes not supported in path expansion");
    int64_t num_elements = fifo_type.getStaticNumElements();
    mlir::OpResult alloc_result = getFifoAllocateResult(flow.carrier);
    Key key = {alloc_result ? alloc_result.getOwner() : nullptr, src, dest};
    PrivateResourceSpec*& spec = specs_[key];
    if (!spec)
      spec = plan_->resource_factory.createFifo(src, dest, {num_elements},
                                                fifo_type.getElementType());
    else
      spec->elements_per_slot.push_back(num_elements);
    size_t slot = next_slot_[key]++;
    if (mirror_slot) validateFifoSlotIndex(flow.carrier, slot, spec);
    return {spec, slot};
  }

 private:
  using Key = std::tuple<mlir::Operation*, mlir::Attribute, mlir::Attribute>;
  PathExpansionPlan* plan_;
  llvm::DenseMap<Key, PrivateResourceSpec*> specs_;
  llvm::DenseMap<Key, size_t> next_slot_;
};
}  // namespace

/// Return the transfer registered for \p op in \p info, or nullptr.
static TransferMaterializationInfo* findTransferForOp(
    StageMaterializationInfo& info, mlir::Operation* op) {
  for (TransferMaterializationInfo* transfer : info.transfers)
    if (transfer->template_op == op) return transfer;
  return nullptr;
}

/// Step 2: Make the ops at both ends of every flow that got hop stages
/// exchange data with those hop stages, and record what carries the data
/// across the first and the last boundary of each such flow.
///
/// The producer op must now write into the first hop stage and the consumer
/// op must read from the last one. At each end, boundaryPlace() says what
/// carries the data:
///   - A buffer in the memory there. A data transfer then uses a new buffer
///     in that memory instead of its original operand. If the carrier is a
///     memref already in that memory, the op is left as it is and the hop
///     stage uses the memref directly.
///   - A FIFO between two units. A data transfer, read_from_fifo or
///     write_to_fifo then uses a new FIFO between those units.
/// A read_from_fifo or write_to_fifo cannot use a buffer, and an indirect
/// transfer is only adapted to use a buffer; anything else fails.
///
/// An op may be an end of more than one flow, for example a data transfer
/// that reads a memref from one stage and writes a FIFO to another. It gets
/// one transfer, with each side adapted for its own flow.
///
/// How it works: walk the original stages in order, and the ops of each
/// stage in order, indirect transfers first, then data transfers, then FIFO
/// ops. Each op that is an end of a flow with hops is adapted. Walking in
/// this order keeps the new buffers and FIFOs in a stable order.
static mlir::LogicalResult adaptFlowEnds(
    llvm::ArrayRef<StageNode*> expanded_stages,
    llvm::MutableArrayRef<DataFlow> flows, const RoutingGraph& arch_graph,
    PathExpansionPlan* plan, BoundaryFifos& fifos) {
  // The flow ends of every op: the flow, and whether the op is its producer.
  llvm::DenseMap<mlir::Operation*,
                 llvm::SmallVector<std::pair<DataFlow*, bool>, 2>>
      op_ends;
  for (DataFlow& flow : flows) {
    if (flow.chain.empty()) continue;
    flow.carriers.resize(flow.chain.size() + 1);
    op_ends[flow.consumer_op].push_back({&flow, false});
    op_ends[flow.producer_op].push_back({&flow, true});
  }

  // Adapt the side of \p op that faces a hop stage of \p flow.
  auto adaptEnd = [&](StageNode* stage, mlir::Operation* op, DataFlow& flow,
                      bool is_producer) -> mlir::LogicalResult {
    StageMaterializationInfo& stage_info = plan->getStageInfo(stage);
    size_t k = is_producer ? 0 : flow.chain.size();
    // The op reads the carrier at the consumer end and writes it at the
    // producer end.
    bool source_side = !is_producer;
    BoundaryPlace place = boundaryPlace(flow, k, arch_graph);
    Carrier& carrier = flow.carriers[k];

    // A memref already in the right memory is used as it is.
    if (flow.carrierIsMemref() && place.memory &&
        memrefMemorySpace(flow.carrier) == place.memory) {
      carrier = TransferInfoFactory::describeTemplateSide(op, source_side);
      carrier.resource =
          plan->resource_factory.createExistingValue(flow.carrier);
      return mlir::success();
    }

    bool is_fifo_op =
        mlir::isa<mlir::ktdf::ReadFromFifoOp, mlir::ktdf::WriteToFifoOp>(op);
    bool is_ind = mlir::isa<mlir::ktdf::IndDataTransferOp>(op);
    if ((is_fifo_op && place.memory) || (is_ind && !place.memory)) {
      LDBG(1) << "Stage " << stage->getStageId() << ": cannot make "
              << op->getName() << " exchange data with a hop stage through "
              << (place.memory ? "memory " : "a FIFO from ")
              << (place.memory ? place.memory : place.fifo_src);
      return mlir::failure();
    }

    // The ends of the edge this transfer implements; only for debug output.
    ResourceType here = stage_info.stage_resource;
    ResourceType there = place.memory  ? place.memory
                         : is_producer ? place.fifo_dest
                                       : place.fifo_src;
    ResourceType source_resource = is_producer ? here : there;
    ResourceType dest_resource = is_producer ? there : here;
    RoutingGraph::EdgeInfo edge{flow.path[flow.boundaries[k] - 1],
                                flow.path[flow.boundaries[k]], 1};

    if (is_fifo_op) {
      auto [spec, slot] = fifos.get(flow, place.fifo_src, place.fifo_dest,
                                    Tile{}, /*mirror_slot=*/true);
      TransferMaterializationInfo* transfer =
          plan->transfer_factory.createFromFifoOp(op, edge, source_resource,
                                                  dest_resource, spec, slot,
                                                  /*is_read=*/source_side);
      stage_info.transfers.push_back(transfer);
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
      carrier = transfer->side(source_side);
      LDBG(1) << "  Stage " << stage->getStageId() << ": " << op->getName()
              << " - FIFO src=" << place.fifo_src
              << ", dest=" << place.fifo_dest << ", slot " << slot << "\n";
      return mlir::success();
    }

    llvm::FailureOr<Tile> tile = tileOf(op, source_side);
    if (mlir::failed(tile)) {
      LDBG(1) << "Stage " << stage->getStageId()
              << ": transfer has a dynamic size";
      return mlir::failure();
    }
    PrivateResourceSpec* spec;
    size_t slot = 0;
    if (place.memory) {
      spec = plan->resource_factory.createMemoryBuffer(
          place.memory, tile->shape, tile->element_type);
    } else {
      std::tie(spec, slot) = fifos.get(flow, place.fifo_src, place.fifo_dest,
                                       *tile, /*mirror_slot=*/true);
    }

    TransferMaterializationInfo* transfer = findTransferForOp(stage_info, op);
    if (!transfer) {
      transfer = plan->transfer_factory.createFromTemplate(
          op, edge, source_resource, dest_resource);
      stage_info.transfers.push_back(transfer);
    }
    mlir::OpBuilder builder(op->getContext());
    plan->transfer_factory.retargetSide(transfer, source_side, spec, slot,
                                        builder);
    if (stage_info.kind == StageMaterializationInfo::Kind::kPreserveOriginal)
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptTransfer;
    carrier = transfer->side(source_side);
    return mlir::success();
  };

  for (StageNode* stage : expanded_stages) {
    if (isIntermediateStage(stage)) continue;
    auto stage_op = mlir::cast<mlir::ktdf::StageOp>(stage->getOperation());

    llvm::SmallVector<mlir::Operation*> ops;
    stage_op.walk([&](mlir::ktdf::IndDataTransferOp op) { ops.push_back(op); });
    stage_op.walk([&](mlir::ktdf::DataTransferOp op) { ops.push_back(op); });
    stage_op.walk([&](mlir::Operation* op) {
      if (mlir::isa<mlir::ktdf::ReadFromFifoOp, mlir::ktdf::WriteToFifoOp>(op))
        ops.push_back(op);
    });

    for (mlir::Operation* op : ops) {
      auto ends_it = op_ends.find(op);
      if (ends_it == op_ends.end()) continue;
      for (auto [flow, is_producer] : ends_it->second)
        if (mlir::failed(adaptEnd(stage, op, *flow, is_producer)))
          return mlir::failure();
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 2b: Retype FIFOs Between Original Stages
//===----------------------------------------------------------------------===//

/// Return true if \p transfer replaces the FIFO operand of \p endpoint. A
/// transfer on a DataTransferOp may only replace its memref side, in which
/// case the FIFO operand is still the original one.
static bool transferRewritesFifo(const TransferMaterializationInfo* transfer,
                                 const FifoEndpoint& endpoint) {
  if (!transfer) return false;
  return transfer->side(endpoint.fifo_is_source).resource != nullptr;
}

/// Step 2b: Step 2 only rewrites FIFOs that touch a synthetic stage. A FIFO
/// whose producer and consumer are both original stages keeps the endpoints
/// from the input IR (e.g. memory-level "L1" -> "SFU"), even though Step 1 has
/// assigned an applicable unit to both stages. Retype every such FIFO to
/// (producer.applicable_unit -> consumer.applicable_unit) and register
/// transfers on both ends so the producer and consumer move to the new FIFO
/// together.
///
/// The new allocation mirrors the original fifo.allocate slot for slot, since
/// slot order is significant for FIFO semantics.
static mlir::LogicalResult retypeFifosBetweenOriginalStages(
    const FifoEndpoints& endpoints, const RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  const llvm::MapVector<mlir::Value, FifoEndpoint>& producers =
      endpoints.producers;
  const llvm::MapVector<mlir::Value, FifoEndpoint>& consumers =
      endpoints.consumers;
  const llvm::SmallPtrSet<mlir::Value, 4>& indirect_slots =
      endpoints.indirect_slots;

  // Classify each slot and group the ones needing a new type by the
  // fifo.allocate that defines them.
  using Endpoints = std::pair<mlir::Attribute, mlir::Attribute>;
  struct SlotToRetype {
    mlir::Value fifo_slot;
    unsigned slot_idx;
    FifoEndpoint producer;
    FifoEndpoint consumer;
    Endpoints expected;
  };
  struct AllocToRetype {
    llvm::SmallVector<SlotToRetype> slots;
    bool needs_retype = false;
  };
  llvm::MapVector<mlir::Operation*, AllocToRetype> allocs;

  for (auto& [fifo_slot, producer] : producers) {
    auto consumer_it = consumers.find(fifo_slot);
    // A slot with only one end in this pipeline has nothing to keep in sync.
    if (consumer_it == consumers.end()) continue;
    const FifoEndpoint& consumer = consumer_it->second;

    bool producer_rewritten = transferRewritesFifo(
        findTransferForOp(plan->stage_info[producer.stage], producer.op),
        producer);
    bool consumer_rewritten = transferRewritesFifo(
        findTransferForOp(plan->stage_info[consumer.stage], consumer.op),
        consumer);
    assert(producer_rewritten == consumer_rewritten &&
           "FIFO slot was rewritten on only one end; producer and consumer "
           "would use different FIFOs");
    // Both ends were already moved off this FIFO by Step 2.
    if (producer_rewritten) continue;

    std::optional<ResourceType> src_unit =
        plan->stage_info[producer.stage].applicable_unit;
    std::optional<ResourceType> dest_unit =
        plan->stage_info[consumer.stage].applicable_unit;
    if (!src_unit || !dest_unit) continue;

    auto fifo_type = mlir::cast<mlir::ktdf::FifoSlotType>(fifo_slot.getType());
    Endpoints expected = {*src_unit, *dest_unit};
    bool needs_retype =
        expected != Endpoints{fifo_type.getSrc(), fifo_type.getDest()};

    mlir::OpResult alloc_result = getFifoAllocateResult(fifo_slot);
    if (!alloc_result) {
      if (!needs_retype) continue;
      return producer.op->emitError(
          "path-expansion: cannot trace FIFO slot to its fifo.allocate");
    }
    if (needs_retype && indirect_slots.contains(fifo_slot))
      return producer.op->emitError(
          "path-expansion: retyping a FIFO used by an indirect data transfer "
          "is not supported");

    AllocToRetype& alloc = allocs[alloc_result.getOwner()];
    alloc.slots.push_back({fifo_slot, alloc_result.getResultNumber(), producer,
                           consumer, expected});
    alloc.needs_retype |= needs_retype;
  }

  for (auto& [alloc_op, alloc] : allocs) {
    if (!alloc.needs_retype) continue;

    // Every slot of the allocation must still be on the original FIFO, and
    // all of them must map to the same new endpoints; otherwise the new
    // allocation cannot mirror the original one.
    Endpoints first_expected = alloc.slots.front().expected;
    if (alloc.slots.size() != alloc_op->getNumResults() ||
        llvm::any_of(alloc.slots, [&](const SlotToRetype& slot) {
          return slot.expected != first_expected;
        }))
      return alloc_op->emitError(
          "path-expansion: cannot retype a fifo.allocate whose slots do not "
          "all connect the same pair of units");

    auto fifo_src =
        llvm::cast<ResourceType>(alloc.slots.front().expected.first);
    auto fifo_dest =
        llvm::cast<ResourceType>(alloc.slots.front().expected.second);
    llvm::SmallVector<int64_t> elements_per_slot;
    mlir::Type element_type;
    for (mlir::Type result_type : alloc_op->getResultTypes()) {
      auto fifo_type = mlir::cast<mlir::ktdf::FifoSlotType>(result_type);
      assert(!fifo_type.isDynamicNumElements() &&
             "Dynamic FIFO sizes not supported in path expansion");
      assert((!element_type || element_type == fifo_type.getElementType()) &&
             "Expected all slots of a fifo.allocate to share an element type");
      element_type = fifo_type.getElementType();
      elements_per_slot.push_back(fifo_type.getStaticNumElements());
    }
    PrivateResourceSpec* fifo_spec = plan->resource_factory.createFifo(
        fifo_src, fifo_dest, elements_per_slot, element_type);

    for (const SlotToRetype& slot : alloc.slots) {
      validateFifoSlotIndex(slot.fifo_slot, slot.slot_idx, fifo_spec);

      StageMaterializationInfo& producer_info =
          plan->stage_info[slot.producer.stage];
      StageMaterializationInfo& consumer_info =
          plan->stage_info[slot.consumer.stage];

      // Use a dummy edge when no direct edge exists between the stage
      // resources (they usually go through LS-unit nodes).
      RoutingGraph::NodeId src_node_id =
          arch_graph.getNodeIdForResource(producer_info.stage_resource);
      RoutingGraph::NodeId dst_node_id =
          arch_graph.getNodeIdForResource(consumer_info.stage_resource);
      RoutingGraph::EdgeInfo edge{src_node_id, dst_node_id, 1};
      if (auto edge_opt = arch_graph.getEdgeInfo(src_node_id, dst_node_id))
        edge = *edge_opt;

      // Point the FIFO operand of one end at the new slot. A DataTransferOp
      // may already have a transfer that only replaced its memref side; reuse
      // it so the op still has a single transfer.
      auto retargetEndpoint = [&](const FifoEndpoint& endpoint,
                                  StageMaterializationInfo& info,
                                  bool is_consumer) {
        ResourceType source_resource =
            is_consumer ? fifo_src : info.stage_resource;
        ResourceType dest_resource =
            is_consumer ? info.stage_resource : fifo_dest;

        TransferMaterializationInfo* transfer =
            findTransferForOp(info, endpoint.op);
        if (!transfer) {
          transfer =
              mlir::isa<mlir::ktdf::DataTransferOp>(endpoint.op)
                  ? plan->transfer_factory.createFromTemplate(
                        endpoint.op, edge, source_resource, dest_resource)
                  : plan->transfer_factory.createFromFifoOp(
                        endpoint.op, edge, source_resource, dest_resource,
                        fifo_spec, slot.slot_idx, /*is_read=*/is_consumer);
          info.transfers.push_back(transfer);
        }
        TransferSide& fifo_side = transfer->side(/*source_side=*/is_consumer);
        fifo_side.resource = fifo_spec;
        fifo_side.slot_index = slot.slot_idx;

        // The materializer only rewrites read_from_fifo / write_to_fifo in
        // kAdaptFifoKinds stages; DataTransferOps are rewritten in either
        // adapting kind.
        bool is_fifo_op = !mlir::isa<mlir::ktdf::DataTransferOp>(endpoint.op);
        if (info.kind == StageMaterializationInfo::Kind::kPreserveOriginal ||
            is_fifo_op)
          info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
      };

      retargetEndpoint(slot.producer, producer_info, /*is_consumer=*/false);
      retargetEndpoint(slot.consumer, consumer_info, /*is_consumer=*/true);

      LDBG(1) << "  Retyped FIFO slot " << slot.slot_idx << " between stage "
              << slot.producer.stage->getStageId() << " and stage "
              << slot.consumer.stage->getStageId() << " to " << fifo_src
              << " -> " << fifo_dest << "\n";
    }
  }

  return mlir::success();
}

/// Hand \p flow's transfer mode (splat or pad) to the transfer that now fills
/// the destination the mode was meant for.
///
/// A splat or pad is the load from memory into a FIFO, so it belongs to the
/// first of \p transfers, in data-flow order, that writes a FIFO. Everything
/// before it only moves data between memories, at the width of the producer's
/// memref side, and copies. The producer op's transfer, \p producer_transfer
/// (may be null), starts out with the op's own mode and gives it up if it no
/// longer writes a FIFO. Leaves the modes as planned if no transfer does.
static void assignTransferMode(
    const DataFlow& flow,
    llvm::ArrayRef<TransferMaterializationInfo*> transfers,
    TransferMaterializationInfo* producer_transfer) {
  if (flow.transfer_mode == mlir::ktdf::TransferMode::Copy) return;

  for (TransferMaterializationInfo* transfer : transfers) {
    if (!transfer->dest.resource ||
        transfer->dest.resource->kind != PrivateResourceSpec::Kind::kFifo)
      continue;
    if (producer_transfer && producer_transfer != transfer)
      producer_transfer->transfer_mode = mlir::ktdf::TransferMode::Copy;
    transfer->transfer_mode = flow.transfer_mode;
    return;
  }
}

//===----------------------------------------------------------------------===//
// Step 3: Populate Hop Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 3: Give each hop stage the transfers that move data through it.
///
/// Step 2 recorded what carries each flow across its first boundary (out of
/// the producer) and its last boundary (into the consumer). Here each hop
/// stage on the flow's chain gets a transfer from the carrier before it to
/// the carrier after it. A new stage gets only these transfers. An original
/// stage that took on a hop keeps its own body and gets these transfers as
/// well. A hop stage on several flows gets one transfer per flow.
///
/// When two hop stages are next to each other, nothing carries the data
/// between them yet. boundaryPlace() says what should: a buffer in the memory
/// there, or a FIFO between the two units. It is created here, with the
/// shape of the data the flow moves (see flowTile).
///
/// How it works: walk the flows in order. For each flow with hops, create the
/// carriers of its inner boundaries, then one transfer for each hop stage.
/// Finally, give the flow's transfer mode to the first transfer that writes a
/// FIFO (see assignTransferMode).
static mlir::LogicalResult populateHopStageTransfers(
    llvm::MutableArrayRef<DataFlow> flows, const RoutingGraph& arch_graph,
    PathExpansionPlan* plan, BoundaryFifos& fifos) {
  for (DataFlow& flow : flows) {
    if (flow.chain.empty()) continue;
    mlir::Operation* context_op = flow.producer->getOperation();
    mlir::OpBuilder builder(context_op->getContext());
    llvm::FailureOr<Tile> tile = flowTile(flow);

    // Carriers between two hop stages.
    if (flow.chain.size() > 1 && mlir::failed(tile)) {
      LDBG(1) << "Flow from stage " << flow.producer->getStageId()
              << " to stage " << flow.consumer->getStageId()
              << " moves data of a dynamic size";
      return mlir::failure();
    }
    for (size_t k = 1; k < flow.chain.size(); ++k) {
      BoundaryPlace place = boundaryPlace(flow, k, arch_graph);
      PrivateResourceSpec* spec;
      size_t slot = 0;
      if (place.memory)
        spec = plan->resource_factory.createMemoryBuffer(
            place.memory, tile->shape, tile->element_type);
      else
        std::tie(spec, slot) = fifos.get(flow, place.fifo_src, place.fifo_dest,
                                         *tile, /*mirror_slot=*/false);
      flow.carriers[k] = TransferInfoFactory::describePrivateSide(
          spec, slot, context_op, builder);
    }

    // The transfers that move the flow's data, in data-flow order, starting
    // with the producer op's own transfer if it was adapted.
    llvm::SmallVector<TransferMaterializationInfo*> flow_transfers;
    TransferMaterializationInfo* producer_transfer =
        findTransferForOp(plan->getStageInfo(flow.producer), flow.producer_op);
    if (producer_transfer) flow_transfers.push_back(producer_transfer);

    for (size_t k = 0; k < flow.chain.size(); ++k) {
      const Carrier& source = flow.carriers[k];
      const Carrier& dest = flow.carriers[k + 1];
      assert(source.resource && dest.resource &&
             "Every boundary of a flow with hops should have a carrier");

      // The stage ends this transfer connects; only for debug output.
      size_t source_idx = flow.boundaries[k] - 1;
      size_t dest_idx = flow.boundaries[k + 1];
      RoutingGraph::ResourceNode source_node =
          pathNode(arch_graph, flow, source_idx);
      RoutingGraph::ResourceNode dest_node =
          pathNode(arch_graph, flow, dest_idx);
      RoutingGraph::EdgeInfo edge{flow.path[source_idx], flow.path[dest_idx],
                                  1};

      TransferMaterializationInfo* transfer =
          plan->transfer_factory.createSynthetic(edge, source_node.resource,
                                                 dest_node.resource, source,
                                                 dest, builder.getContext());
      plan->getStageInfo(flow.chain[k]).transfers.push_back(transfer);
      flow_transfers.push_back(transfer);

      LDBG(1) << "  Stage " << flow.chain[k]->getStageId()
              << ": created transfer " << source_node.resource << " -> "
              << dest_node.resource << " for the flow from stage "
              << flow.producer->getStageId() << " to stage "
              << flow.consumer->getStageId() << "\n";
    }

    assignTransferMode(flow, flow_transfers, producer_transfer);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Orchestration
//===----------------------------------------------------------------------===//

static void debugPrintInitialPlannerState(
    PipelineNode* pipeline, llvm::ArrayRef<StageNode*> sorted_stages) {
  llvm::dbgs() << "Original PipelineTree before expansion:\n";
  pipeline->print(llvm::dbgs());
  llvm::dbgs() << "\nOriginal topological order: ";
  for (StageNode* stage : sorted_stages) {
    llvm::dbgs() << "Stage " << stage->getStageId() << " ";
  }
  llvm::dbgs() << "\n\n";
}

static void debugPrintDataFlows(llvm::ArrayRef<DataFlow> flows) {
  for (const DataFlow& flow : flows) {
    llvm::dbgs() << "Flow from stage " << flow.producer->getStageId()
                 << " to stage " << flow.consumer->getStageId() << " via "
                 << flow.carrier.getType() << ", path (node IDs): ";
    for (auto node_id : flow.path) llvm::dbgs() << node_id << " ";
    llvm::dbgs() << "\n";
  }
}

static void debugPrintPlanningSummary(
    llvm::ArrayRef<StageNode*> expanded_stages, const PathExpansionPlan* plan) {
  llvm::dbgs() << "\n=== Planning Complete ===\n";
  llvm::dbgs() << "Total stages: " << expanded_stages.size() << "\n";
  llvm::dbgs() << "Total private resources: "
               << plan->resource_factory.getSpecs().size() << "\n";
  debugPrintResourceSpecs(llvm::dbgs(), plan->resource_factory);
}

/// Plan the expansion of a pipeline that needs it. Step 1 adds the hop stages
/// to the PipelineTree and rewires the dependencies. Steps 2, 2b and 3 record
/// in \p plan how each stage must be rewritten: the buffers and FIFOs to
/// create, and the transfers to adapt or add. Nothing is changed in the IR
/// here; the materializer does that afterwards from the tree and the plan.
static mlir::LogicalResult buildExpansionPlan(
    PipelineTree& tree, PipelineNode* pipeline,
    llvm::MutableArrayRef<DataFlow> flows, const FifoEndpoints& fifo_endpoints,
    llvm::ArrayRef<StageNode*> sorted_stages, const RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  if (fifo_endpoints.duplicate) {
    auto [op, is_producer] = *fifo_endpoints.duplicate;
    op->emitError("path-expansion: FIFO slot has more than one ")
        << (is_producer ? "producer" : "consumer");
    return mlir::failure();
  }

  // STEP 1: Add hop stages along each flow
  LDBG(1) << "\n=== Step 1: Add Stages Along Each Path ===\n";
  if (mlir::failed(expandDataFlows(tree, pipeline, flows, sorted_stages,
                                   arch_graph, plan, next_stage_id))) {
    return mlir::failure();
  }
  if (mlir::failed(orderStagesByDependencies(pipeline))) {
    LDBG(1) << "Giving hops to existing stages created a dependency cycle";
    return mlir::failure();
  }
  llvm::FailureOr<llvm::SmallVector<StageNode*>> expanded_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(expanded_stages_or)) {
    return mlir::failure();
  }
  llvm::SmallVector<StageNode*> expanded_stages = *expanded_stages_or;

  // Each unit must still run at most one stage.
  llvm::DenseSet<ResourceType> used_units;
  for (StageNode* stage : expanded_stages) {
    std::optional<ResourceType> unit =
        plan->getStageInfo(stage).applicable_unit;
    if (unit && !used_units.insert(*unit).second) {
      LDBG(1) << "Unit " << *unit << " runs more than one stage";
      return mlir::failure();
    }
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 1");
  });

  // STEP 2: Adapt the ops at both ends of each flow that got hops
  LDBG(1) << "\n=== Step 2: Adapt the Ends of Each Flow ===\n";
  BoundaryFifos fifos(plan);
  if (mlir::failed(
          adaptFlowEnds(expanded_stages, flows, arch_graph, plan, fifos))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 2");
  });

  // STEP 2b: Retype FIFOs whose producer and consumer are both original stages
  LDBG(1) << "\n=== Step 2b: Retype FIFOs Between Original Stages ===\n";
  if (mlir::failed(
          retypeFifosBetweenOriginalStages(fifo_endpoints, arch_graph, plan))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 2b");
  });

  // STEP 3: Populate hop stage transfers
  LDBG(1) << "\n=== Step 3: Populate Hop Stage Transfers ===\n";
  if (mlir::failed(populateHopStageTransfers(flows, arch_graph, plan, fifos))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 3");
  });
  LLVM_DEBUG(debugPrintPlanningSummary(expanded_stages, plan));

  return mlir::success();
}

std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const RoutingGraph& arch_graph) {
  auto plan = std::make_unique<PathExpansionPlan>();
  plan->changed = false;

  // PREP 1: Sort and validate pipeline stages, and find each stage's side
  // (load, compute or store).
  StageSideMap sides;
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      sortAndValidateStages(tree, pipeline, arch_graph, sides);
  if (mlir::failed(sorted_stages_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  LLVM_DEBUG(debugPrintInitialPlannerState(pipeline, sorted_stages));

  // PREP 2: Assign resources
  if (mlir::failed(assignOriginalStageResources(sorted_stages, sides,
                                                arch_graph, plan.get()))) {
    return nullptr;
  }

  // PREP 3: Find the data passed between stages and the route of each
  FifoEndpoints fifo_endpoints = collectFifoEndpoints(sorted_stages);
  std::vector<DataFlow> flows = collectDataFlows(sorted_stages, fifo_endpoints);
  if (mlir::failed(routeDataFlows(flows, sides, arch_graph, plan.get()))) {
    LDBG(1) << "No resource path found\n";
    return nullptr;
  }

  LLVM_DEBUG(debugPrintDataFlows(flows));

  // PREP 4: Assign units to the original stages; each must get its own
  if (mlir::failed(assignOriginalStageUnits(flows, sorted_stages, arch_graph,
                                            plan.get()))) {
    return nullptr;
  }

  // PREP 5: Check whether path expansion is needed
  if (!needsExpansion(flows, arch_graph)) {
    plan->changed = false;
    LDBG(1) << "Pipeline already legal";
    return plan;
  }

  plan->changed = true;

  int next_stage_id = static_cast<int>(sorted_stages.size());
  if (mlir::failed(buildExpansionPlan(tree, pipeline, flows, fifo_endpoints,
                                      sorted_stages, arch_graph, plan.get(),
                                      next_stage_id))) {
    return nullptr;
  }

  return plan;
}

}  // namespace scheduler

// Made with Bob
