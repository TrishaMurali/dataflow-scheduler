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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LogicalMemoryViewBuilder.h"

#include <mlir/IR/Matchers.h>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/SymbolicStartAddress.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"

#define DEBUG_TYPE "logical-memory-view-builder"

using namespace scheduler;

namespace {

/// Returns the memory space attribute from a memref type's memory space
/// encoding, or std::nullopt if the type has no memory space attribute.
std::optional<mlir::Attribute> getMemorySpaceAttr(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref) return std::nullopt;
  auto space_attr = memref.getMemorySpace();
  if (!space_attr) return std::nullopt;
  return space_attr;
}

/// Returns true if the only uses of `val` are memref.dealloc ops.
bool onlyDeallocUses(mlir::Value val) {
  return llvm::all_of(val.getUsers(), [](mlir::Operation* user) {
    return mlir::isa<mlir::memref::DeallocOp>(user);
  });
}

/// Phase 1: Walk all program_unit bodies in `func`.
/// - Collect memory space attributes used by Source A chains,
///   Source A' (ktdp_lowering.construct_memory_view), and Source B casts.
/// - Prune Source B casts whose only uses are memref.dealloc (delete cast +
///   deallocs). Returns the set of needed memory space attributes.
llvm::SetVector<ResourceType> discoverAndPrune(mlir::func::FuncOp func) {
  llvm::SetVector<ResourceType> needed;

  func.walk([&](mlir::dataflow::ProgramUnitOp pu) {
    llvm::SmallVector<mlir::Operation*> to_prune;

    pu.getRegion().front().walk([&](mlir::Operation* op) {
      // Source A: ktdp.construct_memory_view — get memory space from
      // downstream memory_space_cast or from result type for direct users.
      if (auto cmv = mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(op)) {
        for (mlir::Operation* user : cmv.getResult().getUsers()) {
          if (auto msc =
                  mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(user)) {
            if (auto ms = getMemorySpaceAttr(msc.getDest().getType()))
              needed.insert(llvm::cast<ResourceType>(*ms));
          } else if (!mlir::isa<mlir::memref::ReinterpretCastOp>(user)) {
            if (auto ms = getMemorySpaceAttr(cmv.getResult().getType()))
              needed.insert(llvm::cast<ResourceType>(*ms));
          }
        }
        return;
      }

      // Source A': ktdp_lowering.construct_memory_view — memory space is
      // encoded directly on the result type (no downstream cast chain).
      if (auto cmv =
              mlir::dyn_cast<mlir::ktdp_lowering::ConstructMemoryViewOp>(op)) {
        if (auto ms = getMemorySpaceAttr(cmv.getResult().getType()))
          needed.insert(llvm::cast<ResourceType>(*ms));
        return;
      }

      // Source B: unrealized_conversion_cast index -> memref<..., memory_space>
      if (auto ucc = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
        if (ucc.getInputs().size() != 1 ||
            !mlir::isa<mlir::IndexType>(ucc.getInputs()[0].getType()))
          return;
        if (ucc.getOutputs().size() != 1) return;
        auto ms = getMemorySpaceAttr(ucc.getOutputs()[0].getType());
        if (!ms) return;

        if (onlyDeallocUses(ucc.getOutputs()[0])) {
          to_prune.push_back(op);
        } else {
          needed.insert(llvm::cast<ResourceType>(*ms));
        }
      }
    });

    // Prune: delete deallocs then the cast.
    for (mlir::Operation* op : to_prune) {
      auto ucc = mlir::cast<mlir::UnrealizedConversionCastOp>(op);
      llvm::SmallVector<mlir::Operation*> deallocs(
          ucc.getOutputs()[0].getUsers().begin(),
          ucc.getOutputs()[0].getUsers().end());
      for (mlir::Operation* d : deallocs) d->erase();
      op->erase();
      LDBG(1) << "  Pruned dealloc-only cast";
    }
  });

  return needed;
}

/// Phase 3a: resolve from_unit for each needed memory space inside a
/// program_unit. Global spaces (DDR) and below-scratchpad spaces (register
/// files) are looked up directly from memory_unit_ssa at key -1. Per-core
/// spaces (L1) get a uniform map + query emitted inside the body.
mlir::LogicalResult buildResolvedUnits(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::SetVector<ResourceType>& needed_spaces,
    const MemoryUnitSSAMap& memory_unit_ssa,
    const scheduler::arch_view::MemoryTree& memory_tree,
    const SchedulerExtContext& ext_ctx,
    llvm::DenseMap<ResourceType, mlir::Value>& resolved_units,
    mlir::OpBuilder& builder) {
  llvm::SetVector<ResourceType> per_core;
  for (auto ms : needed_spaces) {
    if (memory_tree.isGlobalMemory(ms) || memory_tree.isBelowScratchPad(ms)) {
      mlir::Value unit = memory_unit_ssa.lookup({ms, -1});
      if (!unit) return pu.emitError("global memory unit SSA not found");
      resolved_units[ms] = unit;
    } else if (memory_tree.isPerCoreScratchPadMemory(ms)) {
      per_core.insert(ms);
    }
  }

  if (per_core.empty()) return mlir::success();

  UniformInfra infra(pu->getParentOfType<mlir::func::FuncOp>());
  return infra.buildMemoryUniformMaps(pu, per_core, memory_unit_ssa, ext_ctx,
                                      resolved_units, builder);
}

/// Build a 1-D linearization affine map from static strides.
/// E.g., strides [4096, 4096, 64, 1] → (d0,d1,d2,d3) -> (d0*4096 + d1*4096 +
/// d2*64 + d3)
mlir::AffineMap buildLinearizationMap(mlir::MLIRContext* ctx,
                                      llvm::ArrayRef<int64_t> strides) {
  unsigned rank = strides.size();
  mlir::AffineExpr sum = mlir::getAffineConstantExpr(0, ctx);
  for (unsigned i = 0; i < rank; ++i) {
    sum = sum + mlir::getAffineDimExpr(i, ctx) *
                    mlir::getAffineConstantExpr(strides[i], ctx);
  }
  return mlir::AffineMap::get(rank, 0, sum, ctx);
}

/// The size in bytes of the word the units \p pu runs on address in, when they
/// reach \p memory_space.
///
/// Addressing is in words: a unit whose word is 128 bytes reads word 1 to reach
/// byte 128. The device says the size per memory space, under the load feature
/// of a unit that loads and the store feature of one that stores, and a unit
/// that says nothing addresses in bytes.
///
/// The units of one program unit are instances of one kind on different cores:
/// the load units of a load stage, the store units of a store stage. So the
/// kind of the first answers for all of them, and a set that disagreed would
/// mean one address in two granularities, which is reported.
mlir::FailureOr<int64_t> wordSizeOf(
    mlir::dataflow::ProgramUnitOp pu, ResourceType memory_space,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  std::optional<int64_t> word_size;
  for (mlir::Value unit : pu.getUnits()) {
    auto get_unit = unit.getDefiningOp<mlir::dataflow::GetUnitOp>();
    if (!get_unit)
      return pu.emitError("program_unit operand is not a dataflow.get_unit");
    auto type = get_unit->getAttrOfType<mlir::StringAttr>("type");
    if (!type)
      return get_unit->emitError("dataflow.get_unit has no 'type' attribute");

    // The kind as the device spells it, which is upper case; the unit ops carry
    // it lower case.
    auto kind = mlir::StringAttr::get(pu.getContext(), type.getValue().upper());

    int64_t of_this_unit = 1;
    if (auto load =
            resource_kinds.getFeature<mlir::ktdf_arch::feature::Load>(kind)) {
      of_this_unit = static_cast<int64_t>(load.getWordSize(memory_space));
    } else if (auto store =
                   resource_kinds.getFeature<mlir::ktdf_arch::feature::Store>(
                       kind)) {
      of_this_unit = static_cast<int64_t>(store.getWordSize(memory_space));
    }

    if (word_size && *word_size != of_this_unit) {
      return pu.emitError()
             << "the units of one program unit address " << memory_space
             << " in different word sizes (" << *word_size << " and "
             << of_this_unit << "), so one address would need two symbols";
    }
    word_size = of_this_unit;
  }
  return word_size.value_or(1);
}

/// Walks forward from @p view_like through single-use ViewLikeOpInterface ops.
/// Collects the ops of the chain into @p intermediates , head first, and
/// captures its first reinterpret_cast, if any, into @p rc for offset
/// extraction. Returns the tail of the chain.
mlir::Value walkViewChain(
    mlir::ViewLikeOpInterface view_like,
    llvm::SmallVectorImpl<mlir::Operation*>& intermediates,
    mlir::memref::ReinterpretCastOp& rc) {
  rc =
      mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(view_like.getOperation());
  mlir::Value cursor = view_like.getViewDest();
  intermediates.push_back(view_like.getOperation());
  while (cursor.hasOneUse()) {
    mlir::Operation* user = *cursor.getUsers().begin();
    auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user);
    if (!view)
      break;
    else if (!rc)
      rc = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(user);
    intermediates.push_back(user);
    cursor = view.getViewDest();
  }
  return cursor;
}

/// Phase 3b: replace Source A chains with get_logical_memory_view.
/// Emits the view op and RAUWs the chain tail inline; does not populate
/// `replacements` (Source A handles its own erasure).
///
/// A single ktdp.construct_memory_view may feed multiple ViewLikeOpInterface
/// chains (one per access-tile offset computed from the same base view). Each
/// chain is walked generically via ViewLikeOpInterface; a reinterpret_cast is
/// captured opportunistically for offset extraction if present (the MLIR
/// canonicalizer may fold it away when its offset is zero). All chains are
/// replaced before the cmv is erased to avoid dangling uses.
mlir::LogicalResult replaceSourceAChains(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::DenseMap<ResourceType, mlir::Value>& resolved_units,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    llvm::DenseMap<mlir::Value, mlir::Value>& replacements,
    SymbolAllocator& symbols, mlir::OpBuilder& definitions,
    mlir::OpBuilder& builder) {
  auto* ctx = pu.getContext();

  llvm::SmallVector<mlir::ktdp::ConstructMemoryViewOp> chains;
  pu.getRegion().front().walk(
      [&](mlir::ktdp::ConstructMemoryViewOp cmv) { chains.push_back(cmv); });

  for (auto cmv : chains) {
    // Validate static sizes.
    for (int64_t s : cmv.getStaticSizes()) {
      if (mlir::ShapedType::isDynamic(s))
        return cmv.emitError(
            "construct_memory_view: dynamic sizes not supported");
    }

    // Validate static strides.
    auto static_strides = cmv.getStaticStrides();
    for (int64_t s : static_strides) {
      if (mlir::ShapedType::isDynamic(s))
        return cmv.emitError(
            "construct_memory_view: dynamic strides not supported");
    }

    // Build layout map from static strides (shared by all chains off this cmv).
    auto layout_map = buildLinearizationMap(ctx, static_strides);
    auto src_type = mlir::cast<mlir::MemRefType>(cmv.getResult().getType());
    auto plain_type =
        mlir::MemRefType::get(src_type.getShape(), src_type.getElementType());

    // Lower a single view use chain (or direct use if cursor ==
    // cmv.getResult()).
    auto lowerViewChain =
        [&](mlir::Value cursor, llvm::ArrayRef<mlir::Operation*> intermediates,
            mlir::memref::ReinterpretCastOp rc) -> mlir::LogicalResult {
      // Emitting the view, once the start address is settled -- shared by the
      // symbolic path and the constant one, which arrive at that address in
      // quite different ways but say the same thing with it.
      auto emitView = [&](mlir::Value start_address) -> mlir::LogicalResult {
        auto ms = getMemorySpaceAttr(cursor.getType());
        if (!ms)
          return cmv.emitError(
              "construct_memory_view: no memory space found in chain");
        mlir::Value unit = resolved_units.lookup(llvm::cast<ResourceType>(*ms));
        if (!unit) return cmv.emitError("no resolved unit for memory space");

        // Emit get_logical_memory_view with plain result type (no memory space,
        // no strided layout). The builder is already positioned after the last
        // chain op (and after any offset arithmetic just emitted), so no
        // setInsertionPoint needed here.
        auto view_op = mlir::dataflow::GetLogicalMemoryViewOp::create(
            builder, cmv.getLoc(), plain_type, unit, start_address,
            mlir::AffineMapAttr::get(layout_map));

        // cursor is the tail of the chain, replace all its users with the new
        // view.
        cursor.replaceAllUsesWith(view_op.getData());

        // Erase the now-dead chain tail-to-head. msc, rc and any other
        // intermediates are all in the vector; reverse order handles
        // dependencies.
        for (auto* op : llvm::reverse(intermediates)) op->erase();
        return mlir::success();
      };

      // Whether this tensor's address is computed from one of the run's inputs
      // rather than from constants the compiler picked. If it is, no value
      // stands for it here and the address is a symbol: see
      // SymbolicStartAddress.h.
      mlir::FailureOr<std::optional<SymbolicAddress>> taken =
          takeSymbolicAddressApart(cmv.getOffset(), pu, symbols);
      if (mlir::failed(taken)) return mlir::failure();

      mlir::OpFoldResult reinterpret_offset;
      if (rc) {
        reinterpret_offset = rc.getConstifiedMixedOffset();
        builder.setInsertionPointAfter(rc);
      } else {
        // The canonicalizer only folds away a reinterpret_cast when its offset
        // is 0, so the absence of rc guarantees the original offset was zero.
        reinterpret_offset = mlir::OpFoldResult(builder.getIndexAttr(0));
        builder.setInsertionPointAfter(cursor.getDefiningOp());
      }

      // A symbolic address is emitted whole: one symbol where every grid
      // element reads the same address, one per element gathered into a uniform
      // map where they do not. A displacement is a term of what each symbol is
      // computed from rather than arithmetic at the view, because what code
      // generation writes over has to be the operand itself. It is a term per
      // grid element where the access tile read out of the view is the grid
      // element's own slab of it, which is the other way of saying an address
      // that varies with the grid.
      if (taken->has_value()) {
        mlir::FailureOr<Displacement> displacement =
            takeDisplacementApart(reinterpret_offset, pu);
        if (mlir::failed(displacement)) return mlir::failure();

        // The offset came off a memref, so it counts elements; the address it
        // displaces is one of the run's inputs, which is a byte address (that
        // is why emitSymbolicStartAddress divides by the word size at the end).
        // Convert, or every grid element but the first reads at
        // element_size-th of the distance it should.
        //
        // Only on this path: the constant-address path below adds the same
        // offset to an address the compiler picked, which is itself in
        // elements, so the two agree there and must not be converted.
        displacement->scaleBy(*tryGetSizeInBytes(src_type.getElementType()));

        auto memory_space = getMemorySpaceAttr(cursor.getType());
        if (!memory_space)
          return cmv.emitError(
              "construct_memory_view: no memory space found in chain");
        mlir::FailureOr<int64_t> word_size = wordSizeOf(
            pu, llvm::cast<ResourceType>(*memory_space), resource_kinds);
        if (mlir::failed(word_size)) return mlir::failure();

        mlir::FailureOr<mlir::Value> symbolic = emitSymbolicStartAddress(
            **taken, *displacement, *word_size, pu, symbols, builder,
            definitions, cmv.getLoc());
        if (mlir::failed(symbolic)) return mlir::failure();

        return emitView(*symbolic);
      }

      // Compute start_address = base_addr + reinterpret_offset, where the
      // reinterpret offset may be a static constant OR a dynamic SSA value
      // (e.g. a per-compute-tile offset). getConstifiedMixedOffset() yields an
      // IntegerAttr for a static offset or the SSA Value for a dynamic one.
      mlir::Value start_address = cmv.getOffset();

      if (auto offset_attr =
              mlir::dyn_cast<mlir::Attribute>(reinterpret_offset)) {
        // Static offset: add a constant, skipping the no-op zero case.
        int64_t reinterpret_offset_val =
            llvm::cast<mlir::IntegerAttr>(offset_attr).getInt();
        if (reinterpret_offset_val != 0) {
          mlir::Value offset_cst = mlir::arith::ConstantIndexOp::create(
              builder, cmv.getLoc(), reinterpret_offset_val);
          start_address = mlir::arith::AddIOp::create(
              builder, cmv.getLoc(), start_address, offset_cst);
        }
      } else {
        // Non-constant offset: getConstifiedMixedOffset() returns an SSA Value
        // only when the offset cannot be folded to a constant (a foldable
        // operand would have been promoted to an IntegerAttr above). Add the
        // runtime value directly.
        auto offset_val = llvm::cast<mlir::Value>(reinterpret_offset);
        start_address = mlir::arith::AddIOp::create(builder, cmv.getLoc(),
                                                    start_address, offset_val);

        // The offset can reach a run input, though, and then the sum is no more
        // a number than an address taken from an input is. It becomes a symbol
        // the same way, over the sum in bytes: base and offset both count
        // elements, while a symbol's definition is divided by the word size.
        mlir::Value element_size = mlir::arith::ConstantIndexOp::create(
            builder, cmv.getLoc(),
            *tryGetSizeInBytes(src_type.getElementType()));
        mlir::Value in_bytes = mlir::arith::MulIOp::create(
            builder, cmv.getLoc(), start_address, element_size);

        // Over the whole address, not the offset alone: what the symbol stands
        // for is where the tile starts, and the base is part of that.
        mlir::FailureOr<std::optional<SymbolicAddress>> displaced =
            takeSymbolicAddressApart(in_bytes, pu, symbols);
        if (mlir::failed(displaced)) return mlir::failure();

        if (displaced->has_value()) {
          auto memory_space = getMemorySpaceAttr(cursor.getType());
          if (!memory_space)
            return cmv.emitError(
                "construct_memory_view: no memory space found in chain");
          mlir::FailureOr<int64_t> word_size = wordSizeOf(
              pu, llvm::cast<ResourceType>(*memory_space), resource_kinds);
          if (mlir::failed(word_size)) return mlir::failure();

          // No displacement: the offset is in the expression, and the grid
          // element's share of it is a per-grid term there, so the symbols
          // still come out one per element.
          mlir::FailureOr<mlir::Value> symbolic = emitSymbolicStartAddress(
              **displaced, Displacement(), *word_size, pu, symbols, builder,
              definitions, cmv.getLoc());
          if (mlir::failed(symbolic)) return mlir::failure();
          if (mlir::failed(emitView(*symbolic))) return mlir::failure();

          // After the view, which erases the chain holding the last use of the
          // arithmetic the address was read out of.
          eraseDeadAncestorOps({in_bytes.getDefiningOp()});
          return mlir::success();
        }

        // No run input in it, so the sum stands as the address and the
        // conversion to bytes is undone. Erased here rather than through
        // eraseDeadAncestorOps, which would take the sum with it: the view
        // below is still to read it, so with in_bytes gone it would look dead.
        in_bytes.getDefiningOp()->erase();
        element_size.getDefiningOp()->erase();
      }

      return emitView(start_address);
    };

    // Collect all ViewLikeOpInterface users up front so we can iterate safely
    // while modifying the use-list.
    llvm::SmallVector<mlir::ViewLikeOpInterface> users;
    for (auto* user : cmv.getResult().getUsers()) {
      if (auto viewLikeOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user))
        users.push_back(viewLikeOp);
    }

    for (auto viewLikeOp : users) {
      // Walk forward through ViewLikeOpInterface ops, capturing rc if present
      // for offset extraction.
      mlir::memref::ReinterpretCastOp rc;
      llvm::SmallVector<mlir::Operation*> intermediates;
      mlir::Value cursor = walkViewChain(viewLikeOp, intermediates, rc);
      if (mlir::failed(lowerViewChain(cursor, intermediates, rc)))
        return mlir::failure();
    }

    // Direct users get the view directly using the result type's memory space.
    if (!cmv.getResult().use_empty()) {
      if (mlir::failed(lowerViewChain(cmv.getResult(), /*intermediates=*/{},
                                      /*rc=*/nullptr)))
        return mlir::failure();
    }

    // All chains sourced from this cmv have been replaced; erase it now.
    cmv.erase();
  }
  return mlir::success();
}

/// Replace all ktdp_lowering.construct_memory_view ops inside \p pu with
/// dataflow.get_logical_memory_view. Unlike the ktdp variant the memory space
/// is encoded directly on the result type. The result may be used directly, or
/// through a ViewLikeOpInterface chain (e.g. the reinterpret_cast and
/// memref.cast that bufferization puts on an IAB). Each such chain is replaced
/// by its own view, displaced by the chain's reinterpret_cast offset.
static mlir::LogicalResult replaceLoweringConstructMemoryViewOps(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::DenseMap<ResourceType, mlir::Value>& resolved_units,
    mlir::OpBuilder& builder) {
  auto* ctx = pu.getContext();

  llvm::SmallVector<mlir::ktdp_lowering::ConstructMemoryViewOp> cmvs;
  pu.getRegion().front().walk(
      [&](mlir::ktdp_lowering::ConstructMemoryViewOp cmv) {
        cmvs.push_back(cmv);
      });

  for (auto cmv : cmvs) {
    for (int64_t s : cmv.getStaticSizes()) {
      if (mlir::ShapedType::isDynamic(s))
        return cmv.emitError(
            "ktdp_lowering.construct_memory_view: dynamic sizes not supported");
    }
    for (int64_t s : cmv.getStaticStrides()) {
      if (mlir::ShapedType::isDynamic(s))
        return cmv.emitError(
            "ktdp_lowering.construct_memory_view: dynamic strides not "
            "supported");
    }

    auto src_type = mlir::cast<mlir::MemRefType>(cmv.getResult().getType());
    auto ms = getMemorySpaceAttr(src_type);
    if (!ms)
      return cmv.emitError(
          "ktdp_lowering.construct_memory_view: no memory space on result "
          "type");
    mlir::Value unit = resolved_units.lookup(llvm::cast<ResourceType>(*ms));
    if (!unit)
      return cmv.emitError(
          "ktdp_lowering.construct_memory_view: no resolved unit for memory "
          "space");

    auto layout_map = buildLinearizationMap(ctx, cmv.getStaticStrides());
    auto plain_type =
        mlir::MemRefType::get(src_type.getShape(), src_type.getElementType());

    llvm::SmallVector<mlir::ViewLikeOpInterface> chains;
    for (auto* user : cmv.getResult().getUsers()) {
      if (auto view_like = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user))
        chains.push_back(view_like);
    }

    for (auto view_like : chains) {
      mlir::memref::ReinterpretCastOp rc;
      llvm::SmallVector<mlir::Operation*> intermediates;
      mlir::Value cursor = walkViewChain(view_like, intermediates, rc);

      // start_address = base_addr + reinterpret_offset, both in elements.
      builder.setInsertionPointAfter(cursor.getDefiningOp());
      mlir::Value start_address = cmv.getOffset();
      if (rc) {
        const auto offset = rc.getConstifiedMixedOffset();
        if (auto offset_val = llvm::dyn_cast<mlir::Value>(offset)) {
          start_address = mlir::arith::AddIOp::create(
              builder, cmv.getLoc(), start_address, offset_val);
        } else if (const int64_t offset_cst =
                       llvm::cast<mlir::IntegerAttr>(
                           llvm::cast<mlir::Attribute>(offset))
                           .getInt();
                   offset_cst != 0) {
          start_address = mlir::arith::AddIOp::create(
              builder, cmv.getLoc(), start_address,
              mlir::arith::ConstantIndexOp::create(builder, cmv.getLoc(),
                                                   offset_cst));
        }
      }

      auto view_op = mlir::dataflow::GetLogicalMemoryViewOp::create(
          builder, cmv.getLoc(), plain_type, unit, start_address,
          mlir::AffineMapAttr::get(layout_map));
      cursor.replaceAllUsesWith(view_op.getData());
      for (auto* op : llvm::reverse(intermediates)) op->erase();
    }

    // Direct users get the undisplaced view.
    if (!cmv.getResult().use_empty()) {
      builder.setInsertionPointAfter(cmv);
      auto view_op = mlir::dataflow::GetLogicalMemoryViewOp::create(
          builder, cmv.getLoc(), plain_type, unit, cmv.getOffset(),
          mlir::AffineMapAttr::get(layout_map));
      cmv.getResult().replaceAllUsesWith(view_op.getData());
    }
    cmv.erase();
  }
  return mlir::success();
}

/// Phase 3c: replace Source B unrealized_conversion_casts with
/// get_logical_memory_view. Dealloc-only casts were pruned in Phase 1.
mlir::LogicalResult replaceSourceBCasts(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::DenseMap<ResourceType, mlir::Value>& resolved_units,
    llvm::DenseMap<mlir::Value, mlir::Value>& replacements,
    mlir::OpBuilder& builder) {
  auto* ctx = pu.getContext();

  llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
  pu.getRegion().front().walk([&](mlir::UnrealizedConversionCastOp ucc) {
    if (ucc.getInputs().size() != 1 ||
        !mlir::isa<mlir::IndexType>(ucc.getInputs()[0].getType()))
      return;
    if (ucc.getOutputs().size() != 1) return;
    if (!getMemorySpaceAttr(ucc.getOutputs()[0].getType())) return;
    casts.push_back(ucc);
  });

  for (auto ucc : casts) {
    auto result_type =
        mlir::cast<mlir::MemRefType>(ucc.getOutputs()[0].getType());
    auto shape = result_type.getShape();

    // Assert fully static shape.
    for (int64_t dim : shape) {
      if (mlir::ShapedType::isDynamic(dim))
        return ucc.emitError(
            "unrealized_conversion_cast: dynamic shape not supported");
    }

    // Synthesize contiguous row-major strides from shape.
    llvm::SmallVector<int64_t> strides(shape.size());
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
    auto layout_map = buildLinearizationMap(ctx, strides);

    // Get from_unit.
    auto ms = getMemorySpaceAttr(result_type);
    if (!ms)
      return ucc.emitError("unrealized_conversion_cast: no ktdf memory space");
    mlir::Value from_unit =
        resolved_units.lookup(llvm::cast<ResourceType>(*ms));
    if (!from_unit) return ucc.emitError("no resolved unit for memory space");

    auto plain_type =
        mlir::MemRefType::get(shape, result_type.getElementType());

    builder.setInsertionPoint(ucc);

    // Address assignment materializes the address in elements, which is what
    // the view's start address counts.
    mlir::Value addr = ucc.getInputs()[0];

    auto view_op = mlir::dataflow::GetLogicalMemoryViewOp::create(
        builder, ucc.getLoc(), plain_type, from_unit, addr,
        mlir::AffineMapAttr::get(layout_map));

    // Delete remaining dealloc users (mixed-use casts that survived Phase 1).
    llvm::SmallVector<mlir::Operation*> deallocs;
    for (auto* user : ucc.getOutputs()[0].getUsers())
      if (mlir::isa<mlir::memref::DeallocOp>(user)) deallocs.push_back(user);
    for (auto* d : deallocs) d->erase();

    // If the ucc result flows through a memref.cast (e.g., dynamizing dims for
    // select_memref), RAUW the cast's result with the view now and erase the
    // cast so that propagateTypes only ever sees (ind_)data_transfer or
    // select_memref as consumers of ucc_result. Also update select_memref
    // result types whose operand type changed.
    mlir::Value ucc_result = ucc.getOutputs()[0];
    for (auto* user : llvm::make_early_inc_range(ucc_result.getUsers())) {
      if (auto mc = mlir::dyn_cast<mlir::memref::CastOp>(user)) {
        // Update select_memref result types before RAUW so we can still find
        // them via mc.getDest() users.
        for (auto* mc_user : mc.getDest().getUsers()) {
          if (auto sel = mlir::dyn_cast<mlir::ktdf::SelectMemrefOp>(mc_user))
            sel.getResult().setType(
                mlir::cast<mlir::MemRefType>(view_op.getData().getType()));
        }
        mc.getDest().replaceAllUsesWith(view_op.getData());
        mc.erase();
        break;
      }
    }
    replacements[ucc_result] = view_op.getData();
  }
  return mlir::success();
}

/// Phase 3d: RAUW old values with new get_logical_memory_view results and
/// propagate plain-memref types through select_memref and
/// (ind_)data_transfer.
mlir::LogicalResult propagateTypes(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::DenseMap<mlir::Value, mlir::Value>& replacements,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  for (auto& [old_val, new_val] : replacements) {
    llvm::SmallVector<mlir::Operation*> users(old_val.getUsers().begin(),
                                              old_val.getUsers().end());
    for (auto* user : users) {
      if (mlir::isa<mlir::ktdf::DataTransferOp, mlir::ktdf::IndDataTransferOp>(
              user)) {
        user->replaceUsesOfWith(old_val, new_val);
      } else if (auto sel = mlir::dyn_cast<mlir::ktdf::SelectMemrefOp>(user)) {
        sel->replaceUsesOfWith(old_val, new_val);
        // All operands of select_memref have the same element type; use the
        // new value's plain type for the result.
        sel.getResult().setType(
            mlir::cast<mlir::MemRefType>(new_val.getType()));
        for (auto* sel_user : sel.getResult().getUsers()) {
          if (!mlir::isa<mlir::ktdf::DataTransferOp,
                         mlir::ktdf::IndDataTransferOp>(sel_user))
            return sel.emitError(
                "select_memref result used by unexpected op; expected "
                "data_transfer or ind_data_transfer");
        }
      } else {
        // Sub-scratchpad spaces (e.g. SFU_REG) are register-file buffers
        // consumed directly by compute ops (linalg.*, write_to_fifo, …).
        // Any consumer is valid; just swap the operand.
        auto ms = getMemorySpaceAttr(old_val.getType());
        if (ms &&
            memory_tree.isBelowScratchPad(llvm::cast<ResourceType>(*ms))) {
          user->replaceUsesOfWith(old_val, new_val);
        } else {
          return pu.emitError(
              "unexpected consumer of memory view; expected "
              "data_transfer, ind_data_transfer or select_memref");
        }
      }
    }
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::buildLogicalMemoryViews(
    mlir::func::FuncOp func,
    const scheduler::arch_view::MemoryTree& memory_tree,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    const SchedulerExtContext& ext_ctx, SymbolAllocator& symbols) {
  LDBG(1) << "buildLogicalMemoryViews on " << func.getName();

  // Phase 1: discover needed memory spaces and prune dealloc-only casts.
  auto needed_spaces = discoverAndPrune(func);
  if (needed_spaces.empty()) {
    LDBG(1) << "  No memory spaces found; skipping";
    return mlir::success();
  }

  // Phase 2: extract grid size, then materialize memory-space get_unit ops
  // at func-entry (after existing compute get_unit ops).
  int grid_size = 0;
  if (mlir::failed(extractGridSize(func, grid_size))) return mlir::failure();

  // Find insertion point: after last GetUnitOp in entry block, or at block
  // start if none exist.
  mlir::Block& entry = func.getBody().front();
  mlir::OpBuilder builder(func.getContext());
  mlir::Operation* last_get_unit = nullptr;
  for (mlir::Operation& op : entry) {
    if (mlir::isa<mlir::dataflow::GetUnitOp>(op)) last_get_unit = &op;
  }
  if (last_get_unit)
    builder.setInsertionPointAfter(last_get_unit);
  else
    builder.setInsertionPointToStart(&entry);

  UnitMaterializer materializer(func);
  MemoryUnitSSAMap memory_unit_ssa;
  if (mlir::failed(materializer.materializeMemoryUnits(
          needed_spaces, grid_size, memory_tree, memory_unit_ssa, builder)))
    return mlir::failure();

  // Where each symbol's definition is emitted: at function level, not inside a
  // program unit. A unit's region is one grid element's program, while a symbol
  // belongs to the run as a whole. Nothing reads the values emitted here; they
  // exist so that whatever resolves the symbols can read each definition back
  // out of the IR.
  mlir::OpBuilder definitions(func.getContext());
  definitions.setInsertionPoint(&entry, builder.getInsertionPoint());

  // Which input each of this function's arguments is, declared here rather than
  // when the symbols were numbered. Declaring it earlier would have put it in
  // the work the program units were built from, and each unit would have got a
  // copy of it.
  symbols.declareInputsIn(func, definitions);

  // Phase 3: per-program_unit rewrites.
  llvm::SmallVector<mlir::dataflow::ProgramUnitOp> program_units;
  func.walk([&](mlir::dataflow::ProgramUnitOp pu) {
    program_units.push_back(pu);
    return mlir::WalkResult::skip();
  });

  for (auto pu : program_units) {
    mlir::Block& pu_body = pu.getRegion().front();
    builder.setInsertionPointToStart(&pu_body);

    // Phase 3a: build resolved_units map (uniform maps for per-core, direct
    // lookup for global).
    llvm::DenseMap<ResourceType, mlir::Value> resolved_units;
    if (mlir::failed(buildResolvedUnits(pu, needed_spaces, memory_unit_ssa,
                                        memory_tree, ext_ctx, resolved_units,
                                        builder)))
      return mlir::failure();

    llvm::DenseMap<mlir::Value, mlir::Value> replacements;

    // Phase 3b: Source A chains (ktdp.construct_memory_view).
    if (mlir::failed(replaceSourceAChains(pu, resolved_units, resource_kinds,
                                          replacements, symbols, definitions,
                                          builder)))
      return mlir::failure();

    // Phase 3b': ktdp_lowering.construct_memory_view (IAB and similar).
    if (mlir::failed(
            replaceLoweringConstructMemoryViewOps(pu, resolved_units, builder)))
      return mlir::failure();

    // Phase 3c: Source B casts.
    if (mlir::failed(
            replaceSourceBCasts(pu, resolved_units, replacements, builder)))
      return mlir::failure();

    // Phase 3d: RAUW + type propagation.
    if (mlir::failed(propagateTypes(pu, replacements, memory_tree)))
      return mlir::failure();

    // Erase original Source A chain ops and Source B casts (now dead).
    for (auto& [old_val, new_val] : replacements) {
      if (old_val.use_empty()) old_val.getDefiningOp()->erase();
    }
    (void)pu_body;
  }

  return mlir::success();
}
