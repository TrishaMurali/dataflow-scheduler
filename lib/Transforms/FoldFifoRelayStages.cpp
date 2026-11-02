//===-- FoldFifoRelayStages.cpp ---------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/SimplifyPipeline.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep

#define PASS_NAME "fold-fifo-relay-stages"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Fold FIFO Relay Stages pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_FOLDFIFORELAYSTAGESPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

/// Discardable attributes that block the fold of a relay when both of its
/// transfers carry them with different values.
constexpr llvm::StringLiteral kNonMergeableAttrs[] = {
    "dataflow_scheduler.throttle",
};

/// A transfer from memory into a FIFO slot, in the producer, and the transfer
/// out of that slot into memory, in the consumer.
struct Relay {
  mlir::ktdf::DataTransferOp in;
  mlir::ktdf::DataTransferOp out;
  mlir::ktdf::StageOp producer;
  mlir::ktdf::StageOp consumer;
};

/// Where one end of a transfer is: a memory, or the resource that a FIFO
/// names at the stage's end of it.
struct Place {
  mlir::Attribute resource;
  bool is_memory = false;

  friend auto operator==(const Place& lhs, const Place& rhs) -> bool {
    return lhs.resource == rhs.resource && lhs.is_memory == rhs.is_memory;
  }
};

/// The places a transfer reads from and writes to.
struct TransferEnds {
  Place source;
  Place dest;
};

/// An end that all transfers of a stage share. Path expansion places the
/// stage on it, so no two stages should share one.
struct Anchor {
  mlir::Attribute resource;
  bool is_source = false;

  friend auto operator==(const Anchor& lhs, const Anchor& rhs) -> bool {
    return lhs.resource == rhs.resource && lhs.is_source == rhs.is_source;
  }
};

/// Returns the transfers in @p stage , or std::nullopt if it holds anything
/// else.
[[nodiscard]] auto getTransfers(mlir::ktdf::StageOp stage)
    -> std::optional<llvm::SmallVector<mlir::ktdf::DataTransferOp>> {
  llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
  for (auto& op : *stage.getBody()) {
    auto transfer = llvm::dyn_cast<mlir::ktdf::DataTransferOp>(op);
    if (!transfer) {
      return std::nullopt;
    }
    transfers.push_back(transfer);
  }
  if (transfers.empty()) {
    return std::nullopt;
  }
  return transfers;
}

[[nodiscard]] auto getMemorySpace(mlir::Value memref) -> mlir::Attribute {
  return llvm::cast<mlir::MemRefType>(memref.getType()).getMemorySpace();
}

/// Returns the place of @p value , which a stage reads from if
/// @p stage_reads and writes to otherwise. For a memref that is its memory.
/// For a FIFO slot it is the resource named at the stage's end: the slot's
/// destination when the stage reads from it, its source when the stage writes.
[[nodiscard]] auto getPlace(mlir::Value value, bool stage_reads) -> Place {
  if (auto slot = llvm::dyn_cast<mlir::ktdf::FifoSlotType>(value.getType())) {
    return {stage_reads ? slot.getDest() : slot.getSrc(), false};
  }
  return {getMemorySpace(value), true};
}

[[nodiscard]] auto getEnds(mlir::ktdf::DataTransferOp transfer)
    -> TransferEnds {
  return {getPlace(transfer.getSource(), /*stage_reads=*/true),
          getPlace(transfer.getDestination(), /*stage_reads=*/false)};
}

/// Returns the anchors of a stage whose transfers have @p ends .
///
/// An anchor is an end that all transfers share. If the sources and the
/// destinations are both shared, a memory wins over a FIFO, and if they are of
/// the same kind, both count. Returns std::nullopt if neither end is shared.
[[nodiscard]] auto findAnchors(llvm::ArrayRef<TransferEnds> ends)
    -> std::optional<llvm::SmallVector<Anchor, 2>> {
  assert(!ends.empty() && "expected a stage with transfers");
  const auto shared = [&](auto pick) -> std::optional<Place> {
    const Place first = pick(ends.front());
    if (llvm::all_of(ends, [&](const TransferEnds& transfer) {
          return pick(transfer) == first;
        })) {
      return first;
    }
    return std::nullopt;
  };
  const auto source =
      shared([](const TransferEnds& transfer) { return transfer.source; });
  const auto dest =
      shared([](const TransferEnds& transfer) { return transfer.dest; });

  llvm::SmallVector<Anchor, 2> anchors;
  if (source && (!dest || source->is_memory || !dest->is_memory)) {
    anchors.push_back({source->resource, /*is_source=*/true});
  }
  if (dest && (!source || dest->is_memory || !source->is_memory)) {
    anchors.push_back({dest->resource, /*is_source=*/false});
  }
  if (anchors.empty()) {
    return std::nullopt;
  }
  return anchors;
}

/// Returns the anchors of @p stage , or none if it holds anything but
/// transfers or its transfers share no end.
[[nodiscard]] auto getAnchors(mlir::ktdf::StageOp stage)
    -> llvm::SmallVector<Anchor, 2> {
  const auto transfers = getTransfers(stage);
  if (!transfers) {
    return {};
  }
  const auto ends = llvm::map_to_vector(*transfers, getEnds);
  return findAnchors(ends).value_or(llvm::SmallVector<Anchor, 2>{});
}

/// Merges the discardable attributes of @p in and @p out , preferring those of
/// @p in . Returns std::nullopt if a non-mergeable attribute conflicts.
[[nodiscard]] auto mergeAttrs(mlir::ktdf::DataTransferOp in,
                              mlir::ktdf::DataTransferOp out)
    -> std::optional<mlir::NamedAttrList> {
  for (const auto name : kNonMergeableAttrs) {
    const auto in_attr = in->getDiscardableAttr(name);
    const auto out_attr = out->getDiscardableAttr(name);
    if (in_attr && out_attr && in_attr != out_attr) {
      return std::nullopt;
    }
  }

  mlir::NamedAttrList merged(in->getDiscardableAttrDictionary());
  for (const auto attr : out->getDiscardableAttrs()) {
    if (!merged.get(attr.getName())) {
      merged.append(attr);
    }
  }
  return merged;
}

/// Merges the applicable units of @p target and @p donor . Returns
/// std::nullopt if both are set and do not intersect.
[[nodiscard]] auto mergeUnits(mlir::ktdf::StageOp target,
                              mlir::ktdf::StageOp donor)
    -> std::optional<mlir::ArrayAttr> {
  const auto target_units = target.getApplicableUnitsAttr();
  const auto donor_units = donor.getApplicableUnitsAttr();
  if (!target_units || !donor_units) {
    return target_units ? target_units : donor_units;
  }

  llvm::SmallVector<mlir::Attribute> common;
  for (const auto unit : target_units) {
    if (llvm::is_contained(donor_units.getValue(), unit)) {
      common.push_back(unit);
    }
  }
  if (common.empty()) {
    return std::nullopt;
  }
  return mlir::ArrayAttr::get(target.getContext(), common);
}

/// Returns the relay that starts at @p in : a transfer from memory into a FIFO
/// slot that has no other users than one transfer, in another stage, out of it
/// into memory. Both stages must hold nothing but transfers.
[[nodiscard]] auto findRelay(mlir::ktdf::DataTransferOp in)
    -> std::optional<Relay> {
  if (!in.isSourceMemRef() || !in.isDestFifo()) {
    return std::nullopt;
  }
  auto slot = in.getDestination();
  if (!slot.hasNUses(2)) {
    return std::nullopt;
  }
  mlir::ktdf::DataTransferOp out;
  for (auto* user : slot.getUsers()) {
    if (user != in) {
      out = llvm::dyn_cast<mlir::ktdf::DataTransferOp>(user);
    }
  }
  if (!out || out.getSource() != slot || !out.isDestMemRef()) {
    return std::nullopt;
  }

  auto producer = llvm::dyn_cast<mlir::ktdf::StageOp>(in->getParentOp());
  auto consumer = llvm::dyn_cast<mlir::ktdf::StageOp>(out->getParentOp());
  if (!producer || !consumer || producer == consumer ||
      producer->getBlock() != consumer->getBlock() || !getTransfers(producer) ||
      !getTransfers(consumer)) {
    return std::nullopt;
  }
  return Relay{in, out, producer, consumer};
}

/// Checks what @p relay needs to fold, whichever stage it folds into, and
/// returns the attributes of the folded transfer.
///
/// The memories must differ, so that the destination cannot alias the source
/// and the folded transfer can run in any order with the others in its stage.
/// What goes into the slot must be what comes out of it.
[[nodiscard]] auto checkRelay(Relay relay)
    -> std::optional<mlir::NamedAttrList> {
  const auto reject = [&](llvm::StringRef reason) {
    LDBG() << "not folding " << relay.in << ": " << reason;
    return std::nullopt;
  };

  if (getMemorySpace(relay.in.getSource()) ==
      getMemorySpace(relay.out.getDestination())) {
    return reject("source and destination memories are the same");
  }

  const auto pushed = relay.in.getMixedDestSizes();
  const auto popped = relay.out.getMixedSourceSizes();
  if (pushed.size() != popped.size() ||
      !llvm::all_of(llvm::zip_equal(pushed, popped), [](auto sizes) {
        return mlir::isEqualConstantIntOrValue(std::get<0>(sizes),
                                               std::get<1>(sizes));
      })) {
    return reject("FIFO slot sizes do not match");
  }

  auto attrs = mergeAttrs(relay.in, relay.out);
  if (!attrs) {
    return reject("transfers carry conflicting attributes");
  }
  return attrs;
}

/// Checks whether @p relay can fold into its producer if @p into_producer ,
/// and into its consumer otherwise. That stage is the target, which gets the
/// folded transfer. The other is the donor, which loses its half of the relay.
/// Returns the applicable units of the target after the fold.
///
/// - Tokens: stages that await the producer, other than the consumer, rely on
///   the producer having read the source, so the read cannot move to the
///   consumer. Stages that await the consumer rely on the destination being
///   written, so the write can only move to the producer if the consumer
///   awaits it.
/// - Cycles: the target also awaits what the donor awaits, so no stage that
///   the donor awaits, other than the target, may depend on the target.
/// - Anchors: the transfers of the target, and of the donor if any are left,
///   must still share an end. An anchor that a stage gains must not be one
///   that another stage already has.
/// - Units: the applicable units of the two stages must intersect.
[[nodiscard]] auto canFold(Relay relay, bool into_producer)
    -> std::optional<mlir::ArrayAttr> {
  auto target = into_producer ? relay.producer : relay.consumer;
  auto donor = into_producer ? relay.consumer : relay.producer;
  auto target_half = into_producer ? relay.in : relay.out;
  auto donor_half = into_producer ? relay.out : relay.in;
  const auto reject = [&](llvm::StringRef reason) {
    LDBG() << "not folding " << relay.in << " into the "
           << (into_producer ? "producer" : "consumer") << ": " << reason;
    return std::nullopt;
  };

  const mlir::ktdf::StageGraph::Node producer = relay.producer;
  const mlir::ktdf::StageGraph::Node consumer = relay.consumer;
  if (into_producer) {
    if (!consumer.getConsumers().empty() &&
        !llvm::is_contained(consumer.getProducers(), relay.producer)) {
      return reject("consumer is awaited but does not await the producer");
    }
  } else if (llvm::any_of(producer.getConsumers(), [&](auto awaiter) {
               return awaiter != relay.consumer;
             })) {
    return reject("producer token is awaited by another stage");
  }

  if (llvm::any_of(mlir::ktdf::StageGraph::Node(donor).getProducers(),
                   [&](mlir::ktdf::StageGraph::Node awaited) {
                     return awaited != target && awaited.dependsOn(target);
                   })) {
    return reject("dependencies would form a cycle");
  }

  // The ends of the transfers of the target and the donor after the fold.
  const TransferEnds folded{getEnds(relay.in).source, getEnds(relay.out).dest};
  const auto target_transfers = *getTransfers(target);
  const auto donor_transfers = *getTransfers(donor);
  llvm::SmallVector<TransferEnds> target_ends;
  for (auto transfer : target_transfers) {
    target_ends.push_back(transfer == target_half ? folded : getEnds(transfer));
  }
  llvm::SmallVector<TransferEnds> donor_ends;
  for (auto transfer : donor_transfers) {
    if (transfer != donor_half) {
      donor_ends.push_back(getEnds(transfer));
    }
  }

  const auto target_anchors = findAnchors(target_ends);
  if (!target_anchors) {
    return reject("target transfers would share no end");
  }
  llvm::SmallVector<Anchor, 2> donor_anchors;
  if (!donor_ends.empty()) {
    auto anchors = findAnchors(donor_ends);
    if (!anchors) {
      return reject("donor transfers would share no end");
    }
    donor_anchors = std::move(*anchors);
  }

  llvm::SmallVector<Anchor> other_anchors;
  for (auto stage : target->getBlock()->getOps<mlir::ktdf::StageOp>()) {
    if (stage != target && stage != donor) {
      llvm::append_range(other_anchors, getAnchors(stage));
    }
  }
  const auto gainsTakenAnchor = [&](mlir::ktdf::StageOp stage,
                                    llvm::ArrayRef<Anchor> after,
                                    llvm::ArrayRef<Anchor> peer) {
    const auto before = getAnchors(stage);
    return llvm::any_of(after, [&](const Anchor& anchor) {
      return !llvm::is_contained(before, anchor) &&
             (llvm::is_contained(other_anchors, anchor) ||
              llvm::is_contained(peer, anchor));
    });
  };
  if (gainsTakenAnchor(target, *target_anchors, donor_anchors) ||
      gainsTakenAnchor(donor, donor_anchors, *target_anchors)) {
    return reject("a stage would share its anchor with another stage");
  }

  const auto units = mergeUnits(target, donor);
  if (!units) {
    return reject("applicable units do not intersect");
  }
  return units;
}

/// Folds @p relay into one memory-to-memory transfer in its producer if
/// @p into_producer , and in its consumer otherwise. See canFold.
///
/// The folded transfer takes the place of the target's half of the relay, and
/// the target also awaits what the donor awaits. If the donor is left empty, it
/// is erased, and the stages that awaited it await what it awaited instead.
/// Repeated and unused tokens are left for normalizeTokens.
void foldRelay(mlir::RewriterBase& rewriter, Relay relay, bool into_producer,
               const mlir::NamedAttrList& attrs, mlir::ArrayAttr units) {
  auto target = into_producer ? relay.producer : relay.consumer;
  auto donor = into_producer ? relay.consumer : relay.producer;
  auto target_half = into_producer ? relay.in : relay.out;

  LDBG() << "folding " << relay.in << " into the "
         << (into_producer ? "producer " : "consumer ")
         << mlir::OpWithFlags(target, kSkipRegions);

  rewriter.setInsertionPoint(target_half);
  auto transfer = mlir::ktdf::DataTransferOp::create(
      rewriter, rewriter.getFusedLoc({relay.in.getLoc(), relay.out.getLoc()}),
      relay.in.getSource(), relay.in.getSourceMap().value_or(mlir::AffineMap{}),
      relay.in.getSourceIndices(), relay.in.getMixedSourceSizes(),
      relay.out.getDestination(),
      relay.out.getDestMap().value_or(mlir::AffineMap{}),
      relay.out.getDestIndices(), relay.out.getMixedDestSizes());
  transfer->setDiscardableAttrs(attrs.getDictionary(rewriter.getContext()));
  rewriter.eraseOp(relay.in);
  rewriter.eraseOp(relay.out);

  // The target must not await its own tokens.
  auto depends_in = llvm::to_vector(target.getDependsIn());
  for (auto token : donor.getDependsIn()) {
    if (!llvm::is_contained(target.getDependsOut(), token)) {
      depends_in.push_back(token);
    }
  }
  rewriter.modifyOpInPlace(target, [&]() {
    target.setDependsIn(depends_in);
    if (units) {
      target.setApplicableUnitsAttr(units);
    } else {
      target.removeApplicableUnitsAttr();
    }
  });

  if (!donor.getBody()->empty()) {
    return;
  }
  for (auto awaiter :
       llvm::to_vector(mlir::ktdf::StageGraph::Node(donor).getConsumers())) {
    rewriter.modifyOpInPlace(awaiter, [&]() {
      for (auto token : donor.getDependsIn()) {
        awaiter.addInDependency(
            llvm::cast<mlir::TypedValue<mlir::ktdf::TokenType>>(token));
      }
    });
  }
  rewriter.eraseOp(donor);
}

/// Folds the first relay in @p pipeline that can fold, in stage order.
///
/// A relay first tries the stage whose fold leaves the other one empty, then
/// the producer, then the consumer.
///
/// @return Whether a relay was folded.
auto foldFirstRelay(mlir::RewriterBase& rewriter,
                    mlir::ktdf::PipelineOp pipeline) -> bool {
  for (auto stage : pipeline.getStages()) {
    const auto transfers = getTransfers(stage);
    if (!transfers) {
      continue;
    }
    for (auto in : *transfers) {
      const auto relay = findRelay(in);
      if (!relay) {
        continue;
      }
      const auto attrs = checkRelay(*relay);
      if (!attrs) {
        continue;
      }

      const auto empties_producer = transfers->size() == 1;
      const auto empties_consumer = getTransfers(relay->consumer)->size() == 1;
      const auto consumer_first = empties_producer && !empties_consumer;
      for (const auto into_producer : {!consumer_first, consumer_first}) {
        if (const auto units = canFold(*relay, into_producer)) {
          foldRelay(rewriter, *relay, into_producer, *attrs, *units);
          return true;
        }
      }
    }
  }
  return false;
}

/// Drops the slots without uses from @p alloc , unless it has no slots in use
/// at all, in which case it is left for erasure.
void shrinkFifoAllocation(mlir::RewriterBase& rewriter,
                          mlir::ktdf::FifoAllocateOp alloc) {
  if (alloc->use_empty() ||
      llvm::none_of(alloc.getSlots(),
                    [](mlir::Value slot) { return slot.use_empty(); })) {
    return;
  }

  llvm::SmallVector<mlir::Value> kept_slots;
  llvm::SmallVector<mlir::Type> kept_types;
  llvm::SmallVector<mlir::Value> kept_sizes;
  auto dynamic_sizes = alloc.getDynamicSizes().begin();
  for (auto slot : alloc.getSlots()) {
    const auto type = llvm::cast<mlir::ktdf::FifoSlotType>(slot.getType());
    const auto size =
        type.isDynamicNumElements() ? *dynamic_sizes++ : mlir::Value{};
    if (slot.use_empty()) {
      continue;
    }
    kept_slots.push_back(slot);
    kept_types.push_back(type);
    if (size) {
      kept_sizes.push_back(size);
    }
  }

  rewriter.setInsertionPoint(alloc);
  auto shrunk = mlir::ktdf::FifoAllocateOp::create(rewriter, alloc.getLoc(),
                                                   kept_types, kept_sizes);
  shrunk->setDiscardableAttrs(alloc->getDiscardableAttrDictionary());
  rewriter.replaceAllUsesWith(kept_slots, shrunk.getSlots());
  rewriter.eraseOp(alloc);
}

struct FoldFifoRelayStagesPass
    : impl::FoldFifoRelayStagesPassBase<FoldFifoRelayStagesPass> {
  using FoldFifoRelayStagesPassBase::FoldFifoRelayStagesPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    mlir::IRRewriter rewriter(&getContext());
    getOperation()->walk([&](mlir::ktdf::PipelineOp pipeline) {
      // Each fold removes a relay, so this ends.
      auto changed = false;
      while (foldFirstRelay(rewriter, pipeline)) {
        ++num_folded;
        changed = true;
      }
      if (!changed) {
        return;
      }

      // A fold into a producer can make it await a stage that comes later.
      mlir::ktdf::orderStages(rewriter, pipeline);
      // Drop the repeated tokens, and the slots and tokens left without users.
      mlir::ktdf::normalizeTokens(rewriter, pipeline);
      mlir::ktdf::PrivateBuilder::canonicalize(rewriter, pipeline);
      if (auto private_op = pipeline.getPrivateOp(); private_op) {
        for (auto alloc : llvm::to_vector(
                 private_op.getBody()->getOps<mlir::ktdf::FifoAllocateOp>())) {
          shrinkFifoAllocation(rewriter, alloc);
        }
      }
    });
  }
};

}  // namespace
