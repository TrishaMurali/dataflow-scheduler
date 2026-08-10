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

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
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
/// - Collect memory space attributes used by Source A chains and Source B
/// casts.
/// - Prune Source B casts whose only uses are memref.dealloc (delete cast +
/// deallocs). Returns the set of needed memory space attributes.
llvm::SetVector<ResourceType> discoverAndPrune(mlir::func::FuncOp func) {
  llvm::SetVector<ResourceType> needed;

  func.walk([&](mlir::dataflow::ProgramUnitOp pu) {
    llvm::SmallVector<mlir::Operation*> to_prune;

    pu.getRegion().front().walk([&](mlir::Operation* op) {
      // Source A: ktdp.construct_memory_view — get memory space from the
      // downstream memory_space_cast result type.
      if (auto cmv = mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(op)) {
        for (mlir::Operation* user : cmv.getResult().getUsers()) {
          if (auto msc =
                  mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(user)) {
            if (auto ms = getMemorySpaceAttr(msc.getDest().getType()))
              needed.insert(*ms);
          }
        }
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
          needed.insert(*ms);
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
      auto it = memory_unit_ssa.find({ms, -1});
      if (it == memory_unit_ssa.end())
        return pu.emitError("global memory unit SSA not found");
      resolved_units[ms] = it->second;
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
    llvm::DenseMap<mlir::Value, mlir::Value>& replacements,
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

    // Collect all ViewLikeOpInterface users up front so we can iterate safely
    // while modifying the use-list.
    llvm::SmallVector<mlir::ViewLikeOpInterface> users;
    for (auto* user : cmv.getResult().getUsers()) {
      if (auto viewLikeOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user))
        users.push_back(viewLikeOp);
    }
    if (users.empty())
      return cmv.emitError(
          "construct_memory_view: no ViewLikeOpInterface users found");

    for (auto viewLikeOp : users) {
      // Walk forward through ViewLikeOpInterface ops, capturing rc if present
      // for offset extraction.
      mlir::memref::ReinterpretCastOp rc;
      mlir::Value cursor = viewLikeOp.getViewDest();
      llvm::SmallVector<mlir::Operation*> intermediates;
      intermediates.push_back(viewLikeOp.getOperation());
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

      // Compute start_address = base_addr + reinterpret_offset, where the
      // reinterpret offset may be a static constant OR a dynamic SSA value
      // (e.g. a per-compute-tile offset). getConstifiedMixedOffset() yields an
      // IntegerAttr for a static offset or the SSA Value for a dynamic one.
      mlir::Value start_address = cmv.getOffset();

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
      }

      // Get from_unit.
      auto ms = getMemorySpaceAttr(cursor.getType());
      if (!ms)
        return cmv.emitError(
            "construct_memory_view: no memory space found in chain");
      auto it = resolved_units.find(*ms);
      if (it == resolved_units.end())
        return cmv.emitError("no resolved unit for memory space");
      mlir::Value from_unit = it->second;

      // Emit get_logical_memory_view with plain result type (no memory space,
      // no strided layout). The builder is already positioned after the last
      // chain op (and after any offset arithmetic just emitted), so no
      // setInsertionPoint needed here.
      auto view_op = mlir::dataflow::GetLogicalMemoryViewOp::create(
          builder, cmv.getLoc(), plain_type, from_unit, start_address,
          mlir::AffineMapAttr::get(layout_map));

      // cursor is the tail of the chain, replace all its users with the new
      // view.
      cursor.replaceAllUsesWith(view_op.getData());

      // Erase the now-dead chain tail-to-head. msc, rc and any other
      // intermediates are all in the vector; reverse order handles
      // dependencies.
      for (auto* op : llvm::reverse(intermediates)) op->erase();
    }

    // All chains sourced from this cmv have been replaced; erase it now.
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
    auto it = resolved_units.find(*ms);
    if (it == resolved_units.end())
      return ucc.emitError("no resolved unit for memory space");
    mlir::Value from_unit = it->second;

    mlir::Value addr = ucc.getInputs()[0];
    auto plain_type =
        mlir::MemRefType::get(shape, result_type.getElementType());

    builder.setInsertionPoint(ucc);
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
    // cast so that propagateTypes only ever sees data_transfer/select_memref
    // as consumers of ucc_result. Also update select_memref result types whose
    // operand type changed.
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
/// propagate plain-memref types through select_memref and data_transfer.
mlir::LogicalResult propagateTypes(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::DenseMap<mlir::Value, mlir::Value>& replacements,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  for (auto& [old_val, new_val] : replacements) {
    llvm::SmallVector<mlir::Operation*> users(old_val.getUsers().begin(),
                                              old_val.getUsers().end());
    for (auto* user : users) {
      if (auto dt = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(user)) {
        dt->replaceUsesOfWith(old_val, new_val);
      } else if (auto sel = mlir::dyn_cast<mlir::ktdf::SelectMemrefOp>(user)) {
        sel->replaceUsesOfWith(old_val, new_val);
        // All operands of select_memref have the same element type; use the
        // new value's plain type for the result.
        sel.getResult().setType(
            mlir::cast<mlir::MemRefType>(new_val.getType()));
        for (auto* sel_user : sel.getResult().getUsers()) {
          if (!mlir::isa<mlir::ktdf::DataTransferOp>(sel_user))
            return sel.emitError(
                "select_memref result used by unexpected op; expected "
                "data_transfer");
        }
      } else {
        // Sub-scratchpad spaces (e.g. SFU_REG) are register-file buffers
        // consumed directly by compute ops (linalg.*, write_to_fifo, …).
        // Any consumer is valid; just swap the operand.
        auto ms = getMemorySpaceAttr(old_val.getType());
        if (ms && memory_tree.isBelowScratchPad(*ms)) {
          user->replaceUsesOfWith(old_val, new_val);
        } else {
          return pu.emitError(
              "unexpected consumer of memory view; expected "
              "data_transfer or select_memref");
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
    const SchedulerExtContext& ext_ctx) {
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

    // Phase 3b: Source A chains.
    if (mlir::failed(
            replaceSourceAChains(pu, resolved_units, replacements, builder)))
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
