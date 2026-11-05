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
/// The IndirectComputeGroupSplit pass runs immediately after
/// ComputeGroupExtraction and splits each extracted child-module function
/// containing a ktdp.construct_indirect_access_tile into two sibling
/// child-module functions:
///
///   - @<name>_idx_to_addr resolves raw integer indices to flat absolute
///     addresses, in units of the words the indirect transfer addresses the
///     data's memory space in, and stores them into a global address buffer.
///   - @<name> loads that buffer into the IAB and performs the actual
///     gather/scatter through ktdp_lowering.construct_indirect_access_tile.
///
/// The orchestrator is updated to call the first before the second.  The two
/// meet primarily at the address buffer, whose address is a compile-time
/// constant both of them materialize; @<name>_idx_to_addr may additionally
/// take whatever block arguments its address computation cannot do without
/// (an indirect-index memref or data-tensor base or stride that is not a
/// compile-time constant), forwarded from @<name>'s own call site.
///
/// Each candidate is handled in two phases.  analyzeCandidate() and the helpers
/// it calls are pure: they validate the input and derive everything the rewrite
/// will need, but mutate nothing.  splitCandidate() then drives the three
/// rewrite steps — buildIdxToAddrModule(), rewriteGatherScatterModule() and
/// updateOrchestrator() — none of which can fail.  That split is what lets an
/// input the pass cannot handle be diagnosed with the IR still exactly as it
/// was found, rather than left half-transformed.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLoweringDialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "indirect-compute-group-split"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTCOMPUTEGROUPSPLITPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"

namespace {

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Indirect Compute Group Split pass"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Reproducing a value's defining chain in another region
//===----------------------------------------------------------------------===//

/// How walkDependency() treats a block argument it reaches.
enum class BlockArgPolicy {
  /// The block argument is already in scope at the insertion point (e.g. the
  /// target function takes the very same arguments), so it needs neither a
  /// clone nor a mapping.  Never fails.
  kAssumeInScope,
  /// Every block argument reached must already be present in the IRMapping: a
  /// block argument of another region cannot be reconstructed in the target
  /// region, so one that is not mapped is reported back to the caller.
  kRequireMapped,
  /// Like kAssumeInScope — never fails — but additionally records every
  /// not-yet-seen block argument into the walk's collector, so that the
  /// caller learns the full set of block arguments a value's dependency
  /// chain reaches rather than just whether they are already available.
  kCollect,
};

/// Clones the defining op of @p val (and all its transitive dependencies) at
/// @p builder's insertion point, remapping SSA values through @p mapper.
/// A null @p builder means "dry run": walk the chain and report whether it
/// could be reproduced, but clone nothing.
///
/// Implemented iteratively with an explicit work stack to avoid recursive call
/// overhead and stack frame pressure in deep IR graphs — common in kernel
/// compilation where dependency chains can exceed typical stack budgets.
///
/// @p mapper doubles as the visited set — a value it already contains is left
/// alone — so no separate allocation is needed.  Values a dependency captures
/// from above through a nested region are materialized as well;
/// IsolatedFromAbove regions are not descended into.
///
/// Fails only under kRequireMapped, and only when a block argument outside
/// @p mapper is reached; @p unmapped_block_arg then receives that argument so
/// the caller can name it in a diagnostic.  A real (non-dry) run leaves behind
/// whatever work-stack iterations already cloned, so callers are expected to
/// have established up front — e.g. by seeding @p mapper from a prior
/// collectBlockArgDependencies() walk over the same value — that this cannot
/// happen.
///
/// @p collected is only consulted under kCollect, which never fails; a
/// not-yet-visited block argument is appended to it and then, like
/// kAssumeInScope, treated as already available so the traversal need not
/// revisit it.
static LogicalResult walkDependency(Value val, IRMapping& mapper,
                                    OpBuilder* builder, BlockArgPolicy policy,
                                    Value* unmapped_block_arg,
                                    SetVector<Value>* collected = nullptr) {
  // Explicit work stack: pair of (value, phase) to avoid recursive call
  // overhead. Phase 0: process dependencies; phase 1: clone operation.
  SmallVector<std::pair<Value, unsigned>> work_stack;
  work_stack.push_back({val, 0});

  while (!work_stack.empty()) {
    auto [current, phase] = work_stack.pop_back_val();

    if (phase == 0) {
      // Process value and its dependencies.
      if (mapper.contains(current)) continue;

      if (isa<BlockArgument>(current)) {
        // Nothing to clone: the argument is either already in scope or
        // pre-mapped.
        if (policy == BlockArgPolicy::kAssumeInScope) {
          mapper.map(current, current);
          continue;
        }
        if (policy == BlockArgPolicy::kCollect) {
          collected->insert(current);
          mapper.map(current, current);
          continue;
        }
        if (unmapped_block_arg) *unmapped_block_arg = current;
        return failure();
      }

      Operation* op = current.getDefiningOp();
      assert(op &&
             "a Value that is not a BlockArgument must have a defining op");

      // Push phase 1 (cloning) so dependencies are processed first (pre-order).
      work_stack.push_back({current, 1});

      // Collect external operands: values defined outside op that must be
      // cloned. Cache region and use identity check to avoid repeated
      // isAncestor() calls.
      Region* op_region = op->getParentRegion();
      SmallVector<Value> external_operands;

      op->walk<WalkOrder::PreOrder>([&](Operation* child) {
        for (Value operand : child->getOperands()) {
          Region* operand_region = operand.getParentRegion();
          if (operand_region != op_region &&
              !operand_region->isAncestor(op_region))
            continue;
          if (!mapper.contains(operand)) {
            external_operands.push_back(operand);
          }
        }
        if (child->hasTrait<OpTrait::IsIsolatedFromAbove>())
          return WalkResult::skip();
        return WalkResult::advance();
      });

      // Push operands in reverse order so work-stack processes them in forward
      // order (stack is LIFO).  This maintains pre-order dependency resolution.
      for (auto it = external_operands.rbegin(); it != external_operands.rend();
           ++it) {
        work_stack.push_back({*it, 0});
      }
    } else {
      // Phase 1: clone the operation after all dependencies are resolved.
      Operation* op = current.getDefiningOp();
      if (builder) {
        builder->clone(*op, mapper);
      } else {
        // Dry run: record results as available so traversal treats op as
        // handled.
        for (Value result : op->getResults()) mapper.map(result, result);
      }
    }
  }

  return success();
}

static LogicalResult materializeDependency(
    Value val, IRMapping& mapper, OpBuilder& builder, BlockArgPolicy policy,
    Value* unmapped_block_arg = nullptr) {
  return walkDependency(val, mapper, &builder, policy, unmapped_block_arg);
}

/// Records, into @p collected, every block argument that materializing @p
/// val's dependency chain into another region would need forwarded there —
/// i.e. every block argument the chain bottoms out in that is not already
/// covered by an earlier call sharing the same @p mapper.  Clones nothing and
/// never fails: this is how the pass discovers what a generated function's
/// argument list has to be, rather than validating an assumed one.
///
/// @p mapper is used as the visited set and is written to (each value the
/// walk resolves, block arguments included, is marked as mapped to itself),
/// so passing the same @p mapper to several calls — one per root value whose
/// chains should be merged — avoids collecting a value's dependencies twice.
static void collectBlockArgDependencies(Value val, IRMapping& mapper,
                                        SetVector<Value>& collected) {
  LogicalResult status =
      walkDependency(val, mapper, /*builder=*/nullptr, BlockArgPolicy::kCollect,
                     /*unmapped_block_arg=*/nullptr, &collected);
  assert(succeeded(status) && "kCollect should never fail");
  (void)status;
}

//===----------------------------------------------------------------------===//
// IR builders shared by the two halves of the split
//===----------------------------------------------------------------------===//

/// Builds the ktdp.construct_memory_view spanning the whole address buffer,
/// which lives in global memory at the offset, in entries, @p base_offset
/// holds.
///
/// Nothing is threaded between the two halves of the split but that address, so
/// each rebuilds this view for itself.  Both model it on @p idx_view: the
/// address buffer holds one resolved address per raw index, so it shares the
/// index view's shape, sizes, strides and coordinate set exactly, and differs
/// only in its element type — @p element_type rather than the raw index type.
static Value createAddressBufferView(OpBuilder& builder, Location loc,
                                     ktdp::ConstructMemoryViewOp idx_view,
                                     Value base_offset, Type element_type) {
  auto global_mem_space = ktdp::MemorySpaceAttr::get(
      builder.getContext(), ktdp::MemorySpaceKind::global, /*ct_id=*/-1);
  auto view_type = MemRefType::get(
      cast<MemRefType>(idx_view.getResult().getType()).getShape(), element_type,
      nullptr, global_mem_space);

  return ktdp::ConstructMemoryViewOp::create(
             builder, loc,
             /*resultType=*/view_type,
             /*offset=*/base_offset,
             /*sizes=*/ValueRange{},
             /*strides=*/ValueRange{},
             /*static_sizes=*/idx_view.getStaticSizes(),
             /*static_strides=*/idx_view.getStaticStrides(),
             /*memory_space=*/global_mem_space,
             /*coordinate_set=*/idx_view.getCoordinateSetAttr())
      .getResult();
}

/// Builds a ktdp.construct_access_tile covering every element of
/// @p memory_view.
///
/// Each of the four buffers this pass loads or stores is transferred whole, so
/// the tile offset is @p zero_index in all of @p shape's dimensions and both
/// the base map and the tile order are the identity.  @p zero_index is passed
/// in rather than materialized here so that a caller building several tiles
/// shares a single constant.
///
/// @p coordinate_set is that of the indirect-index memory view, over which
/// every buffer involved in the split is laid out.  The tile's element type is
/// `index`, as it is for every ktdp access tile.
static Value createFullAccessTile(OpBuilder& builder, Location loc,
                                  Value memory_view, ArrayRef<int64_t> shape,
                                  Value zero_index, IntegerSet coordinate_set) {
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), builder.getContext());
  SmallVector<Value> zero_indices(shape.size(), zero_index);

  return ktdp::ConstructAccessTilesOp::create(
             builder, loc,
             ktdp::AccessTileType::get(shape, builder.getIndexType()),
             /*base=*/memory_view,
             /*base_map=*/identity_map,
             /*indices=*/zero_indices,
             /*access_tile_set=*/coordinate_set,
             /*access_tile_order=*/identity_map)
      .getResult();
}

//===----------------------------------------------------------------------===//
// State shared by every candidate
//===----------------------------------------------------------------------===//

/// What the device description says about the indirect address buffer, resolved
/// once per pass run rather than once per candidate.
struct AddressBufferInfo {
  /// IAB entry type as declared by the arch spec, e.g. si32.  The index
  /// memref's element type must match it exactly.
  Type entry_type;
  /// Signless form of `entry_type`, e.g. i32: the type the generated address
  /// arithmetic and the address buffer's own elements use.
  IntegerType compute_type;
  /// Memory resource the address buffer is allocated from: the root
  /// ancestor of the node that carries the indirect_address_buffer feature.
  ResourceType global_resource;
  /// Memory space name of the IAB node itself derived from the resource that
  /// carries the indirect_address_buffer feature.  Used as the memref
  /// memory_space attribute for the IAB view.
  StringAttr iab_mem_space;
  /// First IAB entry available to the scheduler, i.e. the first one past any
  /// bytes the arch spec reserves.  The IAB view starts here.  Counted in
  /// entries, as a memory view's offset is.
  size_t iab_base;
  /// Load and store units co-located with the IAB.  They perform the indirect
  /// transfer, so their word size is the granularity the resolved addresses are
  /// expressed in.  Null when the device declares no such unit.
  ktdf_arch::Resource load_unit;
  ktdf_arch::Resource store_unit;
  /// The device's mem_space_mapping, from a ktdp memory space to the arch kind
  /// the load/store features key their word sizes by.  Empty if absent.
  DenseMap<Attribute, Attribute> mem_space_map;
};

/// The orchestrator-side state every candidate needs, likewise resolved once.
///
/// All three members stay null or empty when the top-level module has no
/// orchestrator at all, which is how a bare test input arrives.
struct OrchestratorInfo {
  ModuleOp module;
  /// Call site keyed by callee — each function can only be called once.
  DenseMap<StringAttr, func::CallOp> call_sites;
  /// Symbol table of `module`, kept alive so that the forward declaration added
  /// for each candidate does not rebuild it.
  std::optional<SymbolTable> symbol_table;

  /// Returns the call to @p callee, null if there are none.
  func::CallOp getCallSite(StringAttr callee) const {
    auto it = call_sites.find(callee);
    return it == call_sites.end() ? func::CallOp() : it->second;
  }
};

/// One ktdp.construct_indirect_access_tile to split, together with everything
/// analyzeCandidate() derived from it.  Carrying all of it means that
/// splitCandidate() re-derives and re-validates nothing.
struct SplitCandidate {
  /// The op being lowered, and the function and child module holding it.
  ktdp::ConstructIndirectAccessTilesOp indirect_op;
  func::FuncOp func;
  ModuleOp module;
  /// Memory view of the indirect-index memref (element type from arch spec,
  /// e.g. i32), and its (fully static) type.
  ktdp::ConstructMemoryViewOp idx_view;
  MemRefType idx_type;
  /// Position of the single indirect dimension in $per_dim_subscript_kinds.
  unsigned indir_dim;
  /// Base address of the data being accessed: the $base view's offset.
  Value base_addr;
  /// Stride along `indir_dim`: the $base view's SSA operand, or an IntegerAttr
  /// when the view encodes it statically, in which case splitCandidate()
  /// materializes the constant.  Leaving it folded is what keeps the analysis
  /// from mutating IR that a later check may still reject.
  OpFoldResult stride;
  /// Byte size of the $base view's element type, and the size in bytes of the
  /// word the indirect transfer addresses $base's memory space in.  Together
  /// they convert `base_addr` and `stride` from elements to words.
  int64_t elem_bytes;
  int64_t word_size;
  /// The orchestrator's single call to `func`, or null when it makes none.
  func::CallOp call_site;
  /// $ind_addr_buf_dim_positions for the replacement lowering op.
  DenseI32ArrayAttr ind_addr_buf_dim_positions;
  /// Block arguments of `func` that idx_view's, base_addr's and stride's
  /// combined dependency chains need forwarded into the generated
  /// idx_to_addr function, in the order that function's parameter list uses.
  SmallVector<Value> forwarded_args;
};

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// Returns true if @p child is the orchestrator (original-content) module.
///
/// After ComputeGroupExtraction the top-level module contains:
///   - An optional ktdf_arch.device op (not a ModuleOp)
///   - The orchestrator child module — created without a sym_name
///   - One or more extracted child modules — named "local_schedule_N"
///
/// We distinguish them by the presence of a sym_name: extracted modules always
/// have one (assigned by SymbolTable::renameToUnique); the orchestrator does
/// not carry a user-visible symbol.
static bool isOrchestratorModule(ModuleOp child) {
  return !child.getSymName().has_value();
}

/// Collects the indirect access tile of every function defined in every
/// extracted child module of @p top.  A child module may define more than one
/// function; each is considered independently.
///
/// Functions without one are skipped silently: the pass is a no-op on them.
static LogicalResult collectCandidateOps(
    ModuleOp top,
    SmallVectorImpl<ktdp::ConstructIndirectAccessTilesOp>& candidate_ops) {
  for (auto child : top.getOps<ModuleOp>()) {
    if (isOrchestratorModule(child)) continue;

    for (auto func : child.getOps<func::FuncOp>()) {
      // Forward declarations carry no body to search.
      if (func.isExternal()) continue;

      // At most one indirect op per function is supported, so the walk stops
      // as soon as a second is seen rather than collecting every op only to
      // reject the function.
      ktdp::ConstructIndirectAccessTilesOp indirect_op;
      bool multiple_indirect_ops = false;
      func.walk([&](ktdp::ConstructIndirectAccessTilesOp op) {
        if (indirect_op) {
          multiple_indirect_ops = true;
          return WalkResult::interrupt();
        }
        indirect_op = op;
        return WalkResult::advance();
      });

      if (multiple_indirect_ops)
        return func->emitError(PASS_NAME
                               ": more than one indirect tensor in function '")
               << func.getName() << "' is not supported";

      if (indirect_op) candidate_ops.push_back(indirect_op);
    }
  }
  return success();
}

/// Resolves the address buffer's element type and home memory from the device
/// description.
static FailureOr<AddressBufferInfo> resolveAddressBufferInfo(
    ModuleOp top, const arch_view::MemoryTree& memory_tree,
    ktdf_arch::DeviceManager& device_manager) {
  const auto* device = device_manager.getOrImportDevice();
  if (!device)
    return top->emitError(PASS_NAME
                          ": unable to import device specification; "
                          "cannot resolve IAB entry type");
  const auto& resource_kinds =
      device_manager.getOrCreateView<ktdf_arch::ResourceKinds>(*device);

  // Find the first node (in MemoryTree's iteration order) that carries the
  // indirect_address_buffer feature. There may be more than one such node;
  // for now we just take the first.
  ktdf_arch::feature::IndirectAddressBuffer iab_feature;
  std::optional<arch_view::MemoryTree::MemoryNode> iab_node;
  for (const auto& node : memory_tree) {
    iab_feature =
        resource_kinds.getFeature<ktdf_arch::feature::IndirectAddressBuffer>(
            node.memory_resource);
    if (iab_feature) {
      iab_node = node;
      break;
    }
  }

  if (!iab_feature)
    return top->emitError(
        PASS_NAME
        ": no IAB (indirect_address_buffer) resource found in device "
        "description; cannot lower indirect access tiles");

  Type entry_type = iab_feature.getEntryType();
  if (!entry_type)
    return top->emitError(PASS_NAME
                          ": IAB resource has no entry_type; cannot resolve "
                          "element type for address buffer");
  auto int_type = dyn_cast<IntegerType>(entry_type);
  if (!int_type)
    return top->emitError(PASS_NAME
                          ": IAB entry_type is not an integer type; only "
                          "integer element types are supported for the "
                          "address buffer");

  // Strip signedness: si32 → i32, the form arithmetic ops take.
  IntegerType compute_type =
      int_type.isSignless()
          ? int_type
          : IntegerType::get(top.getContext(), int_type.getWidth());

  LDBG(1) << "IAB entry type " << entry_type << ", address arithmetic in "
          << compute_type;

  AddressBufferInfo info;
  info.entry_type = entry_type;
  info.compute_type = compute_type;
  info.global_resource = memory_tree.getRootOf(*iab_node).memory_resource;
  info.iab_mem_space = cast<StringAttr>(iab_node->memory_resource);
  // The arch spec reserves bytes, but a view's offset counts entries; a
  // partially reserved entry is unusable, so round up.
  info.iab_base = llvm::divideCeil(iab_node->reserved_in_bytes.value_or(0),
                                   *tryGetSizeInBytes(compute_type));

  // The units sharing an enclosing group kind with the IAB are the ones that
  // read it to perform the indirect load or store.
  auto iab_parents = resource_kinds[iab_node->memory_resource].getParentKinds();
  auto find_colocated = [&](auto has_feature) -> ktdf_arch::Resource {
    for (const ktdf_arch::ResourceKinds::Kind& kind : resource_kinds) {
      if (!has_feature(kind.getExemplar())) continue;
      for (ktdf_arch::KindAttr parent : kind.getParentKinds())
        if (iab_parents.count(parent)) return kind.getExemplar();
    }
    return nullptr;
  };
  info.load_unit = find_colocated([](ktdf_arch::Resource unit) {
    return static_cast<bool>(unit.getFeature<ktdf_arch::feature::Load>());
  });
  info.store_unit = find_colocated([](ktdf_arch::Resource unit) {
    return static_cast<bool>(unit.getFeature<ktdf_arch::feature::Store>());
  });

  if (auto mapping =
          device->getAttrOfType<ktdf_arch::MapAttr>("mem_space_mapping"))
    info.mem_space_map.insert_range(mapping);

  return info;
}

/// Locates the orchestrator module, its function and every call that function
/// makes.  All candidates update the same function, so this is looked up once
/// instead of being rediscovered per candidate.  Validates that each function
/// is called at most once, as the indirect compute group split requires a
/// single address buffer per function.
static FailureOr<OrchestratorInfo> resolveOrchestrator(ModuleOp top) {
  OrchestratorInfo orchestrator;
  for (auto child : top.getOps<ModuleOp>()) {
    if (!isOrchestratorModule(child)) continue;
    orchestrator.module = child;
    break;
  }
  if (!orchestrator.module) {
    LDBG(1) << "no orchestrator module; call sites will not be updated";
    return orchestrator;
  }

  // Each function can only be called once, as the address buffer is filled by a
  // single @<name>_idx_to_addr call placed before that call site.
  LogicalResult walk_status = success();
  orchestrator.module.walk([&](func::CallOp call_op) {
    StringAttr callee = call_op.getCalleeAttr().getAttr();
    if (orchestrator.call_sites.contains(callee)) {
      call_op->emitError(PASS_NAME ": the orchestrator calls '")
          << callee.getValue()
          << "' more than once; only a single call site per indirect compute "
             "group is supported";
      walk_status = failure();
      return WalkResult::interrupt();
    }
    orchestrator.call_sites[callee] = call_op;
    return WalkResult::advance();
  });
  if (failed(walk_status)) return failure();

  orchestrator.symbol_table.emplace(orchestrator.module);
  return orchestrator;
}

/// Returns the position of the single variable @p subscript selects, or nullopt
/// if it selects anything else.
///
/// @p zero_captured maps every captured variable bound to a literal zero onto
/// the constant zero and every other variable onto itself.  Substituting it
/// reduces the whole question to one pattern match: AffineExpr construction
/// folds the substituted zeros away, so the result is a bare dimension exactly
/// when the subscript depends on one variable and nothing else.
/// `%desc_0[%c0 + %arg5]`, whose map result is `d0 + d1`, reduces to `d1`.
///
/// A non-zero offset, a coefficient over a live variable, a floordiv/mod and a
/// second live variable all survive the substitution as something other than a
/// dimension, and are rejected: each selects a different entry than the
/// position alone does, and $ind_addr_buf_dim_positions has no way to carry the
/// difference.
static std::optional<unsigned> resolveSubscriptPosition(
    AffineExpr subscript, ArrayRef<AffineExpr> zero_captured) {
  AffineExpr reduced = subscript.replaceDims(zero_captured);
  if (auto dim = dyn_cast<AffineDimExpr>(reduced)) return dim.getPosition();

  // The subscript reduced to zero, so every variable it mentions is a literal
  // zero and any of them names the entry it selects — a live variable could
  // only have vanished through a multiplication by zero, which AffineExpr folds
  // at construction.  A subscript that is a bare constant mentions no variable
  // at all, leaving no position to encode.
  auto constant = dyn_cast<AffineConstantExpr>(reduced);
  if (!constant || constant.getValue() != 0) return std::nullopt;
  for (unsigned pos = 0, e = zero_captured.size(); pos < e; ++pos)
    if (subscript.isFunctionOfDim(pos)) return pos;
  return std::nullopt;
}

/// Derives $ind_addr_buf_dim_positions for
/// ktdp_lowering.construct_indirect_access_tile from the indirect dimension's
/// entry in @p indirect_op's $per_dim_subscript_maps.  Reads attributes and
/// operands only.
static FailureOr<DenseI32ArrayAttr> deriveIndAddrBufDimPositions(
    ktdp::ConstructIndirectAccessTilesOp indirect_op, unsigned indir_dim) {
  MLIRContext* ctx = indirect_op.getContext();
  ArrayAttr per_dim_maps = indirect_op.getPerDimSubscriptMaps();
  auto indir_map = cast<AffineMapAttr>(per_dim_maps[indir_dim]).getValue();

  // A captured variable that is a literal zero adds nothing to a subscript, so
  // a sum containing it still selects a single element.  Build the substitution
  // resolveSubscriptPosition() folds through, once for the whole map: the
  // constant zero for those variables, the variable itself for the rest.
  // Sizing it to the captured variables leaves the intermediate variables that
  // follow them in the unified space untouched, replaceDims() passing over a
  // dimension it has no replacement for — those range over the variable space
  // and are never treated as zero.
  ValueRange captured_vars = indirect_op.getCapturedVariables();
  SmallVector<AffineExpr> zero_captured;
  zero_captured.reserve(captured_vars.size());
  for (unsigned pos = 0, e = captured_vars.size(); pos < e; ++pos) {
    std::optional<int64_t> value = getConstantIntValue(captured_vars[pos]);
    zero_captured.push_back(value && *value == 0 ? getAffineConstantExpr(0, ctx)
                                                 : getAffineDimExpr(pos, ctx));
  }

  // Each IAB subscript names one variable of the unified
  // (captured_variables..., intermediate_variables...) space — a position is
  // all $ind_addr_buf_dim_positions can encode, never arithmetic.  That is
  // enough because the address buffer holds one resolved address per element of
  // the indirect-index memref, laid out identically, so subscript r has only to
  // select the same element it selected of the indirect-index memref.
  SmallVector<int32_t> iab_dim_positions;
  iab_dim_positions.reserve(indir_map.getNumResults());
  for (unsigned r = 0, e = indir_map.getNumResults(); r < e; ++r) {
    std::optional<unsigned> position =
        resolveSubscriptPosition(indir_map.getResult(r), zero_captured);
    if (!position) {
      return indirect_op->emitError(PASS_NAME ": result ")
             << r << " of per_dim_subscript_maps[" << indir_dim << "] ("
             << per_dim_maps[indir_dim]
             << ") cannot be used as an address-buffer subscript: it must name "
                "exactly one variable, optionally added to captured variables "
                "that are literal zeros, because $ind_addr_buf_dim_positions "
                "encodes a variable position and no arithmetic";
    }
    iab_dim_positions.push_back(static_cast<int32_t>(*position));
  }

  return DenseI32ArrayAttr::get(ctx, iab_dim_positions);
}

/// Returns the size in bytes of the word the unit performing
/// @p indirect_op's transfer addresses @p memory_space in: the IAB-co-located
/// load unit's for a gather, the store unit's for a scatter.  A tile that is
/// both loaded and stored needs the two to agree, since the address buffer
/// holds a single address per index.
static FailureOr<int64_t> resolveWordSize(
    ktdp::ConstructIndirectAccessTilesOp indirect_op, Attribute memory_space,
    const AddressBufferInfo& layout) {
  bool is_load = false, is_store = false;
  for (Operation* user : indirect_op.getResult().getUsers()) {
    is_load |= isa<ktdp::LoadOp>(user);
    is_store |= isa<ktdp::StoreOp>(user);
  }
  if (!is_load && !is_store)
    return indirect_op->emitError(
        PASS_NAME
        ": indirect access tile is neither loaded nor stored; cannot "
        "determine the address granularity of the indirect transfer");

  if (auto mapped = layout.mem_space_map.lookup(memory_space))
    memory_space = mapped;

  std::optional<int64_t> word_size;
  auto add = [&](ktdf_arch::LoadStoreAttr feature,
                 StringRef what) -> LogicalResult {
    if (!feature)
      return indirect_op->emitError(PASS_NAME ": no ")
             << what
             << " unit co-located with the IAB; cannot determine the address "
                "granularity of the indirect transfer";
    auto size = static_cast<int64_t>(feature.getWordSize(memory_space));
    if (word_size && *word_size != size)
      return indirect_op->emitError(PASS_NAME
                                    ": the IAB's load and store units address ")
             << memory_space << " in different word sizes (" << *word_size
             << " and " << size << " bytes)";
    word_size = size;
    return success();
  };
  // Resource's accessors are not const-qualified, hence the copies.
  ktdf_arch::Resource load_unit = layout.load_unit;
  ktdf_arch::Resource store_unit = layout.store_unit;
  if (is_load &&
      failed(add(load_unit ? load_unit.getFeature<ktdf_arch::feature::Load>()
                           : ktdf_arch::feature::Load(),
                 "load")))
    return failure();
  if (is_store &&
      failed(add(store_unit ? store_unit.getFeature<ktdf_arch::feature::Store>()
                            : ktdf_arch::feature::Store(),
                 "store")))
    return failure();
  return *word_size;
}

/// Determines the argument list buildIdxToAddrModule() must give the
/// generated function: the distinct block arguments of @p idx_view's
/// enclosing function that @p idx_view's operands, @p base_addr and
/// @p stride_value's dependency chains reach, in that order (each chain's own
/// internal traversal order, first chain wins on repeats). Everything else in
/// those chains — constants, arith ops, ... — is cloned wholesale into the
/// generated function rather than forwarded, so it never appears here.
///
/// @p stride_value is null when the stride is a constant that has not been
/// materialized yet, in which case it contributes nothing.
static SmallVector<Value> collectForwardedArgs(
    ktdp::ConstructMemoryViewOp idx_view, Value base_addr, Value stride_value) {
  IRMapping visited;
  SetVector<Value> collected;
  collectBlockArgDependencies(base_addr, visited, collected);
  if (stride_value)
    collectBlockArgDependencies(stride_value, visited, collected);
  for (Value operand : idx_view->getOperands())
    collectBlockArgDependencies(operand, visited, collected);
  return collected.takeVector();
}

/// Validates @p indirect_op and derives everything splitCandidate() will need.
///
/// Mutates nothing, so a candidate the pass cannot handle is diagnosed with its
/// function, the address buffer and the orchestrator all still exactly as they
/// were.
static FailureOr<SplitCandidate> analyzeCandidate(
    ktdp::ConstructIndirectAccessTilesOp indirect_op,
    const AddressBufferInfo& layout, const OrchestratorInfo& orchestrator) {
  SplitCandidate candidate;
  candidate.indirect_op = indirect_op;
  candidate.func = indirect_op->getParentOfType<func::FuncOp>();
  assert(candidate.func && "candidate op must be enclosed in a func");
  candidate.module = candidate.func->getParentOfType<ModuleOp>();
  assert(candidate.module && "candidate func must be enclosed in a module");

  // The indirect-index memref: exactly one indirect dimension is supported.
  auto idx_memrefs = indirect_op.getIndirectMemrefs();
  if (idx_memrefs.empty())
    return indirect_op->emitError(
        PASS_NAME
        ": indirect access tile has no indirect memref; "
        "at least one indirect dimension is required");
  if (idx_memrefs.size() > 1)
    return indirect_op->emitError(PASS_NAME ": indirect access tile has ")
           << idx_memrefs.size()
           << " indirect memrefs but only exactly one indirect dimension is "
              "supported";

  candidate.idx_type = cast<MemRefType>(idx_memrefs[0].getType());
  if (candidate.idx_type.getElementType() != layout.entry_type)
    return candidate.func->emitError(PASS_NAME
                                     ": indirect memref element type (")
           << candidate.idx_type.getElementType()
           << ") does not match IAB entry type (" << layout.entry_type << ")";

  candidate.idx_view = dyn_cast_or_null<ktdp::ConstructMemoryViewOp>(
      idx_memrefs[0].getDefiningOp());
  if (!candidate.idx_view)
    return candidate.func->emitError(
        PASS_NAME
        ": indirect memref operand is not defined by a "
        "ktdp.construct_memory_view");

  // The address buffer mirrors the indirect-index memref and is allocated at
  // compile time, so the memory tracker could not accept a runtime size.
  const auto is_dynamic = [](int64_t size) {
    return size == ShapedType::kDynamic;
  };
  if (!candidate.idx_type.hasStaticShape() ||
      !candidate.idx_view.getSizes().empty() ||
      llvm::any_of(candidate.idx_view.getStaticSizes(), is_dynamic))
    return candidate.func->emitError(
        PASS_NAME
        ": indirect memref has dynamic sizes; only fully static "
        "shapes are supported on the indirect-index memory view");
  if (!candidate.idx_view.getStrides().empty() ||
      llvm::any_of(candidate.idx_view.getStaticStrides(), is_dynamic))
    return candidate.func->emitError(
        PASS_NAME
        ": indirect memref has dynamic strides; only fully "
        "static strides are supported on the indirect-index memory view");

  // The indirect dimension is the unique position whose subscript kind is true.
  ArrayAttr subscript_kinds = indirect_op.getPerDimSubscriptKinds();
  auto indir_dim = llvm::find_if(subscript_kinds, [](Attribute kind) {
    return cast<BoolAttr>(kind).getValue();
  });
  if (indir_dim == subscript_kinds.end())
    return candidate.func->emitError(
        PASS_NAME
        ": no indirect dimension found in "
        "per_dim_subscript_kinds of the indirect access "
        "tile");
  candidate.indir_dim = std::distance(subscript_kinds.begin(), indir_dim);

  // Base address and stride both come from the $base memory view.
  auto base_view = dyn_cast_or_null<ktdp::ConstructMemoryViewOp>(
      indirect_op.getBase().getDefiningOp());
  if (!base_view)
    return candidate.func->emitError(
        PASS_NAME
        ": $base operand of indirect access tile is not "
        "defined by a ktdp.construct_memory_view; cannot "
        "resolve base address and stride");

  // The base is an index-typed SSA value carried verbatim.  The stride follows
  // ConstructMemoryViewOp's mixed static/dynamic encoding, which getMixedValues
  // folds back into one OpFoldResult per dimension. The stride may be either a
  // compile-time constant (Attribute) or a runtime value (Value).
  candidate.base_addr = base_view.getOffset();
  candidate.stride =
      getMixedValues(base_view.getStaticStrides(), base_view.getStrides(),
                     indirect_op.getContext())[candidate.indir_dim];

  // spyreop.idx32toaddr produces addresses in the words the indirect transfer
  // addresses $base's memory space in, so base and stride are converted from
  // elements to words.  A compile-time constant that is not a whole number of
  // words is a compile error; a runtime value is assumed to be one.
  std::optional<unsigned> elem_bytes = tryGetSizeInBytes(
      cast<MemRefType>(base_view.getResult().getType()).getElementType());
  if (!elem_bytes)
    return candidate.func->emitError(
        PASS_NAME
        ": cannot determine the byte size of the $base view's element type");
  candidate.elem_bytes = *elem_bytes;
  FailureOr<int64_t> word_size =
      resolveWordSize(indirect_op, base_view.getMemorySpace(), layout);
  if (failed(word_size)) return failure();
  candidate.word_size = *word_size;
  for (auto [value, what] :
       {std::pair<OpFoldResult, StringRef>{candidate.base_addr, "base address"},
        {candidate.stride, "stride"}}) {
    std::optional<int64_t> elems = getConstantIntValue(value);
    if (elems && (*elems * candidate.elem_bytes) % candidate.word_size != 0)
      return candidate.func->emitError(PASS_NAME ": ")
             << what << " of the indirect access (" << *elems << " elements, "
             << *elems * candidate.elem_bytes
             << " bytes) is not a multiple of the " << candidate.word_size
             << "-byte word the indirect transfer addresses memory in";
  }

  candidate.call_site = orchestrator.getCallSite(candidate.func.getNameAttr());

  // The last check reads attributes and operand chains without cloning
  // anything.
  FailureOr<DenseI32ArrayAttr> ind_addr_buf_dim_positions =
      deriveIndAddrBufDimPositions(indirect_op, candidate.indir_dim);
  if (failed(ind_addr_buf_dim_positions)) return failure();
  candidate.ind_addr_buf_dim_positions = *ind_addr_buf_dim_positions;

  candidate.forwarded_args =
      collectForwardedArgs(candidate.idx_view, candidate.base_addr,
                           dyn_cast<Value>(candidate.stride));

  return candidate;
}

//===----------------------------------------------------------------------===//
// Rewrite step 1: building @<name>_idx_to_addr
//===----------------------------------------------------------------------===//

/// Converts @p elems, a count of @p elem_bytes-byte elements, into words of
/// @p word_size bytes at @p builder's insertion point.  The ratio is reduced
/// first, so f16 elements in 128-byte words take a single division by 64.  A
/// constant folds to a constant.
static Value convertElementsToWords(OpBuilder& builder, Location loc,
                                    Value elems, int64_t elem_bytes,
                                    int64_t word_size) {
  const int64_t gcd = std::gcd(elem_bytes, word_size);
  const int64_t mul = elem_bytes / gcd, div = word_size / gcd;
  if (mul == 1 && div == 1) return elems;

  if (std::optional<int64_t> value = getConstantIntValue(elems))
    return arith::ConstantIndexOp::create(builder, loc, *value * mul / div);

  Value words = elems;
  if (mul != 1)
    words = arith::MulIOp::create(
        builder, loc, words, arith::ConstantIndexOp::create(builder, loc, mul));
  if (div != 1)
    words = arith::DivSIOp::create(
        builder, loc, words, arith::ConstantIndexOp::create(builder, loc, div));
  return words;
}

/// Builds the @<name>_idx_to_addr child module and inserts it just before
/// @p child_module in their common top-level module.
///
/// The new module holds one function, which:
///   - is named "<child function>_idx_to_addr" (uniqued via SymbolTable)
///   - receives @p forwarded_args, one `index` parameter per entry
///   - reconstructs @p base_addr's, @p stride_value's and @p idx_view's
///     dependency chains from those parameters, cloning everything else
///     (constants, arith ops, ...) in wholesale
///   - loads the raw integer indices through the resulting clone of @p idx_view
///   - converts the base and stride from @p elem_bytes-byte elements into
///     @p word_size-byte words
///   - turns the indices into flat absolute word addresses with
///     spyreop.idx32toaddr
///   - stores those addresses into the address buffer at @p addr_buf_base,
///     an offset in entries into global memory
///
/// Returns the function's final symbol name, which the caller must hand to
/// updateOrchestrator() so the call site names the same symbol.
///
/// @p forwarded_args must be exactly what collectForwardedArgs() returned for
/// the same @p idx_view, @p base_addr and @p stride_value — that is what
/// leaves this step no failure path.  @p top_sym_table is the top-level
/// module's, so it stays valid for subsequent candidates.
static std::string buildIdxToAddrModule(ModuleOp child_module,
                                        SymbolTable& top_sym_table,
                                        ktdp::ConstructMemoryViewOp idx_view,
                                        Value base_addr, Value stride_value,
                                        ArrayRef<Value> forwarded_args,
                                        size_t addr_buf_base, Type compute_type,
                                        int64_t elem_bytes, int64_t word_size) {
  auto func = idx_view->getParentOfType<func::FuncOp>();
  assert(func && "idx_view must be enclosed in a FuncOp");
  auto top = child_module->getParentOfType<ModuleOp>();
  assert(top && "child_module must be enclosed in a top-level ModuleOp");

  MLIRContext* ctx = top.getContext();
  Location loc = func.getLoc();

  // Create the new module as the sibling immediately preceding child_module.
  //
  // Inserting through the symbol table (rather than through an OpBuilder)
  // splices the module into the right place, registers the new symbol, and
  // keeps the caller's table — shared by every candidate — valid.  A function
  // holds at most one indirect access tile, so "<name>_idx_to_addr" is normally
  // free and the name is used verbatim; insert() only appends a uniquing suffix
  // if something already claims it, and returns whatever name it settled on.
  auto new_module =
      ModuleOp::create(loc, func.getName().str() + "_idx_to_addr");
  StringRef new_func_name =
      top_sym_table.insert(new_module, Block::iterator(child_module))
          .getValue();
  LDBG(1) << "building @" << new_func_name << " for compute group @"
          << func.getName() << ", address buffer at " << addr_buf_base;

  // @<new_func_name>(<one index per forwarded arg>)
  // forwarded_args are block arguments of the gather/scatter function whose
  // dependency chains reach base_addr, stride_value or idx_view's operands.
  // All of them are index-typed: ConstructMemoryViewOp's offset/stride operands
  // are always index in MLIR's memref model.
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(new_module.getBody());

  Type index_type = builder.getIndexType();
  auto new_func = func::FuncOp::create(
      loc, new_func_name,
      builder.getFunctionType(
          SmallVector<Type>(forwarded_args.size(), index_type), {}));
  if (auto grid_attr = func->getAttr("grid"))
    new_func->setAttr("grid", grid_attr);
  new_module.push_back(new_func);

  Block* entry = new_func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // Seed the mapping so that each forwarded block argument reads this
  // function's corresponding own argument instead of being cloned — there
  // would be nothing to clone it from, this function's block being the only
  // one in scope.
  IRMapping mapper;
  for (auto [i, arg] : llvm::enumerate(forwarded_args))
    mapper.map(arg, new_func.getArgument(i));

  // Reproduce base_addr's, stride_value's and the index view's own dependency
  // chains — its offset and any dynamic sizes or strides.  Note that the
  // latter iterates over idx_view's operands rather than passing its result,
  // which would clone idx_view here and leave the explicit clone below
  // emitting a spurious duplicate.
  //
  // forwarded_args is exactly collectForwardedArgs()'s result for these same
  // three chains, so every block argument any of them can reach is already in
  // `mapper`; kRequireMapped cannot trip. Assert on it rather than unwinding a
  // half-built module.
  auto materializeChain = [&](Value root) {
    LogicalResult materialized = materializeDependency(
        root, mapper, builder, BlockArgPolicy::kRequireMapped);
    (void)materialized;
    assert(succeeded(materialized) &&
           "forwarded_args must cover every block argument these chains reach");
  };
  materializeChain(base_addr);
  materializeChain(stride_value);
  for (Value operand : idx_view->getOperands()) materializeChain(operand);

  // The element-unit base and stride stay alive until the clone of idx_view
  // below, which may share their constants through `mapper`; whichever of them
  // the word conversion leaves unused is dropped once that clone exists.
  Value elem_base = mapper.lookupOrDefault(base_addr);
  Value elem_stride = mapper.lookupOrDefault(stride_value);
  Value arg_base =
      convertElementsToWords(builder, loc, elem_base, elem_bytes, word_size);
  Value arg_stride =
      convertElementsToWords(builder, loc, elem_stride, elem_bytes, word_size);

  auto idx_view_clone =
      cast<ktdp::ConstructMemoryViewOp>(builder.clone(*idx_view, mapper));
  ArrayRef<int64_t> idx_shape =
      cast<MemRefType>(idx_view_clone.getResult().getType()).getShape();
  unsigned idx_rank = idx_shape.size();
  IntegerSet coordinate_set = idx_view_clone.getCoordinateSetAttr().getValue();

  for (Value elems : {elem_base, elem_stride}) {
    Operation* def = elems.getDefiningOp();
    if (def && isa<arith::ConstantOp>(def) && def->use_empty()) def->erase();
    if (elem_base == elem_stride) break;
  }

  Value addr_buf_base_const = arith::ConstantIndexOp::create(
      builder, loc, static_cast<int64_t>(addr_buf_base));
  Value addr_buf_view = createAddressBufferView(
      builder, loc, idx_view_clone, addr_buf_base_const, compute_type);

  // Load the raw indices out of the indirect-index memref.  ktdp.load only
  // verifies shapes, so asking for compute_type elements out of a view whose
  // element type still carries signedness (si32) is valid and saves a
  // conversion.
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value idx_tile = createFullAccessTile(
      builder, loc, idx_view_clone.getResult(), idx_shape, c0, coordinate_set);
  Value idx_tensor =
      ktdp::LoadOp::create(builder, loc, idx_tile, compute_type).getResult();

  // addr = base + idx * stride, in words, element-wise over the whole index
  // tensor.
  auto addr_tensor_type = RankedTensorType::get(idx_shape, compute_type);
  Value init = tensor::EmptyOp::create(builder, loc, addr_tensor_type,
                                       /*dynamicSizes=*/ValueRange{});

  AffineMap identity_map = AffineMap::getMultiDimIdentityMap(idx_rank, ctx);
  SmallVector<Attribute> iter_types(
      idx_rank,
      linalg::IteratorTypeAttr::get(ctx, utils::IteratorType::parallel));

  auto generic_op = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{addr_tensor_type},
      /*inputs=*/ValueRange{idx_tensor},
      /*outputs=*/ValueRange{init},
      builder.getAffineMapArrayAttr({identity_map, identity_map}),
      builder.getArrayAttr(iter_types),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});

  // Build the body with a dedicated OpBuilder so that createBlock does not move
  // the outer insertion point out of the function's entry block.
  {
    OpBuilder body_builder(ctx);
    Block* body = body_builder.createBlock(&generic_op.getRegion());
    Value idx_arg = body->addArgument(compute_type, loc);
    body->addArgument(compute_type, loc);  // Unused output iteration argument.
    body_builder.setInsertionPointToEnd(body);

    // spyreop.idx32toaddr takes its base and stride in the address type, not as
    // `index`.
    Value base_cast =
        arith::IndexCastUIOp::create(body_builder, loc, compute_type, arg_base);
    Value stride_cast = arith::IndexCastUIOp::create(body_builder, loc,
                                                     compute_type, arg_stride);
    Value addr = spyreop::Idx32ToAddr::create(body_builder, loc, compute_type,
                                              idx_arg, base_cast, stride_cast);
    linalg::YieldOp::create(body_builder, loc, ValueRange{addr});
  }

  // Store the resolved addresses into the address buffer.  ktdp.store likewise
  // checks only that the tensor and the access tile agree in shape, so a
  // tensor<...xi32> going into an index-typed tile is valid.
  Value addr_buf_tile = createFullAccessTile(builder, loc, addr_buf_view,
                                             idx_shape, c0, coordinate_set);
  ktdp::StoreOp::create(builder, loc, generic_op.getResult(0), addr_buf_tile);

  func::ReturnOp::create(builder, loc, ValueRange{});

  return std::string(new_func_name);
}

//===----------------------------------------------------------------------===//
// Rewrite step 2: rewriting @<name> to address through the IAB
//===----------------------------------------------------------------------===//

/// Rewrites the gather/scatter child-module function around @p indirect_op in
/// place.  The function keeps its signature and its compute, but stops
/// subscripting the indirect-index memref and instead dereferences the IAB,
/// which it fills from the address buffer @<name>_idx_to_addr wrote:
///   - build an IAB view over the address buffer, and a global view of the same
///   - copy the address buffer into the IAB (ktdp.load then ktdp.store)
///   - rebuild the $base view with a zero offset, the original having been
///     folded into every address by @<name>_idx_to_addr
///   - replace @p indirect_op with ktdp_lowering.construct_indirect_access_tile
///
/// @p addr_buf_base is the address buffer's offset in global memory, and
/// @p iab_base the IAB's first available entry; both count entries, but they
/// live in different memories, so they are independent.
///
/// @p ind_addr_buf_dim_positions comes from deriveIndAddrBufDimPositions();
/// deriving it up front is what leaves this step no failure path.
static void rewriteGatherScatterModule(
    ktdp::ConstructIndirectAccessTilesOp indirect_op,
    ktdp::ConstructMemoryViewOp idx_view, size_t addr_buf_base,
    unsigned indir_dim, Type compute_type, StringAttr iab_mem_space,
    size_t iab_base, DenseI32ArrayAttr ind_addr_buf_dim_positions) {
  auto func = indirect_op->getParentOfType<func::FuncOp>();
  assert(func && "indirect_op must be enclosed in a FuncOp");
  MLIRContext* ctx = func.getContext();
  Location loc = func.getLoc();
  LDBG(1) << "rewriting @" << func.getName()
          << " to address through the IAB, indirect dimension " << indir_dim;

  // The address buffer and the IAB both mirror the indirect-index memref's
  // static shape, strides and coordinate set; only their element types differ.
  ArrayRef<int64_t> idx_shape =
      cast<MemRefType>(idx_view.getResult().getType()).getShape();
  IntegerSet coordinate_set = idx_view.getCoordinateSetAttr().getValue();

  auto base_view =
      cast<ktdp::ConstructMemoryViewOp>(indirect_op.getBase().getDefiningOp());

  // Insert everything immediately before the indirect op, so that it is all in
  // scope for the replacement.
  OpBuilder builder(indirect_op);
  Value addr_buf_base_const = arith::ConstantIndexOp::create(
      builder, loc, static_cast<int64_t>(addr_buf_base));
  Value iab_base_const = arith::ConstantIndexOp::create(
      builder, loc, static_cast<int64_t>(iab_base));

  // The IAB view over the address buffer.  This uses
  // ktdp_lowering.construct_memory_view rather than the ktdp one because "IAB"
  // is a plain StringAttr, which ktdp.construct_memory_view rejects at
  // verification time — it constrains $memory_space to a Ktdp_MemorySpaceAttr.
  // IAB entries use the arch-spec entry type (e.g. i32).
  Value iab_view = ktdp_lowering::ConstructMemoryViewOp::create(
      builder, loc,
      /*resultType=*/
      MemRefType::get(idx_shape, compute_type, nullptr, iab_mem_space),
      /*offset=*/iab_base_const,
      /*sizes=*/ValueRange{},
      /*strides=*/ValueRange{},
      /*static_sizes=*/idx_view.getStaticSizes(),
      /*static_strides=*/idx_view.getStaticStrides(),
      /*memory_space=*/iab_mem_space,
      /*coordinate_set=*/idx_view.getCoordinateSetAttr());

  // Copy the addresses @<name>_idx_to_addr resolved into the IAB, reading them
  // back out of global memory through a view of the very same buffer.
  Value addr_buf_view = createAddressBufferView(
      builder, loc, idx_view, addr_buf_base_const, compute_type);

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value addr_buf_tile = createFullAccessTile(builder, loc, addr_buf_view,
                                             idx_shape, c0, coordinate_set);
  Value addr_tensor =
      ktdp::LoadOp::create(builder, loc, addr_buf_tile, compute_type)
          .getResult();
  Value iab_tile = createFullAccessTile(builder, loc, iab_view, idx_shape, c0,
                                        coordinate_set);
  ktdp::StoreOp::create(builder, loc, addr_tensor, iab_tile);

  // Rebuild the $base view with a zero offset, reusing the constant above: the
  // original offset has already been folded into every address-buffer entry by
  // @<name>_idx_to_addr, and counting it twice would skew every access.
  Value zero_offset_base_view = ktdp::ConstructMemoryViewOp::create(
      builder, loc,
      /*resultType=*/cast<MemRefType>(base_view.getResult().getType()),
      /*offset=*/c0,
      /*sizes=*/base_view.getSizes(),
      /*strides=*/base_view.getStrides(),
      /*static_sizes=*/base_view.getStaticSizes(),
      /*static_strides=*/base_view.getStaticStrides(),
      /*memory_space=*/base_view.getMemorySpace(),
      /*coordinate_set=*/base_view.getCoordinateSetAttr());

  // The indirect dimension's original map subscripted into the indirect-index
  // memref (e.g. `(d0,d1,d2,d3) -> (d2, d3)`).  The lowering op resolves that
  // indirection through the IAB instead, so its direct offset into $base along
  // that dimension must be zero; every other dimension keeps its map verbatim.
  //
  // The maps address the unified (captured..., intermediate...) variable space
  // by position, which is independent of the tile-output ordering
  // variables_space_order induces, so the replacement map needs the same input
  // arity as its neighbours and nothing more.
  const unsigned num_intermediate_vars =
      indirect_op.getIntermediateVariables().size();
  const unsigned unified_dims =
      indirect_op.getCapturedVariables().size() + num_intermediate_vars;
  AffineMap zero_map = AffineMap::get(unified_dims, /*numSymbols=*/0,
                                      getAffineConstantExpr(0, ctx), ctx);

  ArrayAttr per_dim_maps = indirect_op.getPerDimSubscriptMaps();
  SmallVector<Attribute> lowering_per_dim_maps(per_dim_maps.begin(),
                                               per_dim_maps.end());
  lowering_per_dim_maps[indir_dim] = AffineMapAttr::get(zero_map);

  // The result type must match the original op's so that the existing consumers
  // (ktdp.load, ktdp.store) keep working untouched.  variables_space_order and
  // variables_space_set are forwarded unchanged: the tile-output ordering and
  // the iteration bounds are properties of the variable-space enumeration,
  // which factoring the indirection into the IAB does not disturb.
  auto lowering_op = ktdp_lowering::ConstructIndirectAccessTileOp::create(
      builder, loc,
      /*resultType=*/
      cast<ktdp::AccessTileType>(indirect_op.getResult().getType()),
      /*base=*/zero_offset_base_view,
      /*indAddrBufMemref=*/iab_view,
      /*indAddrBufDimPositions=*/ind_addr_buf_dim_positions,
      /*perDimSubscriptMaps=*/ArrayAttr::get(ctx, lowering_per_dim_maps),
      /*capturedVariables=*/indirect_op.getCapturedVariables(),
      /*numIntermediateVariables=*/num_intermediate_vars,
      /*variablesSpaceOrder=*/indirect_op.getVariablesSpaceOrder(),
      /*variablesSpaceSet=*/indirect_op.getVariablesSpaceSet().getValue());

  indirect_op.getResult().replaceAllUsesWith(lowering_op.getResult());
  indirect_op.erase();
}

//===----------------------------------------------------------------------===//
// Rewrite step 3: updating the orchestrator
//===----------------------------------------------------------------------===//

/// Makes the orchestrator call @<idx_to_addr_func_name>(<forwarded_args...>)
/// immediately before @p target_call, adding a private forward declaration of
/// that function to the orchestrator module if it does not already have one.
/// @p target_call itself is left unchanged; its signature has not moved, and
/// the two halves coordinate primarily through the address buffer's
/// compile-time address, plus whatever @p forwarded_args carries across.
///
/// Every entry of @p forwarded_args is, by construction, a block argument of
/// @p child_func — collectForwardedArgs() only ever collects those — so its
/// value at the orchestrator's call site is simply the call operand bound to
/// that argument position; nothing needs to be cloned or reconstructed here.
///
/// @param orch_sym_table Symbol table of the orchestrator child module, so the
///                       declaration is inserted without invalidating it for
///                       the next candidate.
/// @param target_call    The single func.call to the gather/scatter function
///                       this call must precede.  Its operands define what the
///                       child's arguments are bound to, so the orchestrator's
///                       own signature may differ freely from the child's in
///                       arity and order.
/// @param child_func     The gather/scatter child-module function, whose block
///                       arguments are mapped onto @p target_call's operands.
///                       Supplied by the caller, which already holds it: it
///                       must not be recovered by matching the child *module*'s
///                       symbol against the gather/scatter *function*'s name,
///                       as those are separate namespaces that only happen to
///                       agree today.
static void updateOrchestrator(SymbolTable& orch_sym_table,
                               func::CallOp target_call,
                               func::FuncOp child_func,
                               llvm::StringRef idx_to_addr_func_name,
                               ArrayRef<Value> forwarded_args) {
  assert(target_call && child_func &&
         "call site and child function are required");
  LDBG(1) << "inserting call to @" << idx_to_addr_func_name << " before @"
          << child_func.getName() << "'s call site";

  // Argument i of the child receives operand i of the call, whatever the
  // orchestrator's own signature happens to be — so each forwarded block
  // argument's orchestrator-scope value is that same-numbered call operand.
  auto call_args = target_call.getArgOperands();
  SmallVector<Value> new_operands;
  new_operands.reserve(forwarded_args.size());
  for (Value arg : forwarded_args)
    new_operands.push_back(call_args[cast<BlockArgument>(arg).getArgNumber()]);

  // Everything new goes immediately before the existing call.
  OpBuilder builder(target_call);
  func::CallOp::create(builder, target_call.getLoc(),
                       /*callee=*/idx_to_addr_func_name,
                       /*resultTypes=*/TypeRange{}, new_operands);

  // Declare the callee, unless an earlier candidate already did.
  if (orch_sym_table.lookup(idx_to_addr_func_name)) return;

  Type index_type = builder.getIndexType();
  auto decl = func::FuncOp::create(
      target_call.getLoc(), idx_to_addr_func_name,
      builder.getFunctionType(
          SmallVector<Type>(forwarded_args.size(), index_type), {}));
  decl.setPrivate();
  // Appends to the orchestrator module's body and registers the new symbol, so
  // the table stays usable for the next candidate.  The lookup above rules out
  // a collision, so no renaming happens here.
  orch_sym_table.insert(decl);
}

//===----------------------------------------------------------------------===//
// Rewrite driver
//===----------------------------------------------------------------------===//

/// Erases @p op and then any of its operands' defining ops that this made
/// trivially dead.  Used to drop the scaffolding the rewrite leaves behind: the
/// indirect-index memory view the gather/scatter function no longer reads, and
/// the stride constant that only the orchestrator ends up using.
static void eraseNowDeadOp(Operation* op) {
  SmallVector<Operation*> worklist{op};
  while (!worklist.empty()) {
    Operation* dead = worklist.pop_back_val();
    if (!isOpTriviallyDead(dead)) continue;
    for (Value operand : dead->getOperands())
      if (Operation* def = operand.getDefiningOp()) worklist.push_back(def);
    dead->erase();
  }
}

/// Performs the split @p candidate describes.
///
/// Allocating the address buffer is the only step that can fail, and it happens
/// before the first mutation; analyzeCandidate() has already rejected
/// everything else, so nothing past that point needs an escape route.
static LogicalResult splitCandidate(const SplitCandidate& candidate,
                                    const AddressBufferInfo& addr_info,
                                    OrchestratorInfo& orchestrator,
                                    SymbolTable& top_sym_table,
                                    MemoryTrackerAnalysis& memory_tracker) {
  func::FuncOp func = candidate.func;

  // One address-buffer entry per index, aligned to an entry: an entry's byte
  // width doubles as its alignment requirement.  resolveAddressBufferInfo()
  // rejected any non-integer entry type, so the size is known here.
  const size_t entry_bytes = *tryGetSizeInBytes(addr_info.compute_type);
  const size_t size_bytes =
      static_cast<size_t>(candidate.idx_type.getNumElements()) * entry_bytes;
  auto addr_buf_base = memory_tracker.allocate(addr_info.global_resource,
                                               size_bytes, entry_bytes);
  if (!addr_buf_base) {
    std::string reason;
    llvm::handleAllErrors(addr_buf_base.takeError(),
                          [&](const llvm::ErrorInfoBase& ei) {
                            if (!reason.empty()) reason += "; ";
                            reason += ei.message();
                          });
    return func->emitError(PASS_NAME
                           ": failed to allocate address buffer in global "
                           "memory: ")
           << reason;
  }
  LDBG(1) << "allocated " << size_bytes << " bytes for @" << func.getName()
          << "'s address buffer at byte " << *addr_buf_base;

  // The tracker hands out byte addresses, but the address buffer's views take
  // their offset in entries.  The allocation was aligned to an entry, so this
  // divides exactly.
  assert(*addr_buf_base % entry_bytes == 0 &&
         "address buffer must be entry-aligned");
  const size_t addr_buf_offset = *addr_buf_base / entry_bytes;

  // Extract or materialize the stride. candidate.stride may be either a Value
  // (already computed at runtime) or an Attribute (compile-time constant).
  // If it's a Value, use it directly. If it's an Attribute, materialize it as
  // a ConstantIndexOp. The result is placed just before the indirect op so that
  // it dominates all uses. Only the orchestrator ends up reading it — it clones
  // the value at its own call site — so the original is erased at the end of
  // this function.
  Value stride = dyn_cast<Value>(candidate.stride);
  Operation* materialized_stride = nullptr;
  if (!stride) {
    Operation* anchor = candidate.indirect_op;
    OpBuilder builder(anchor);
    stride = arith::ConstantIndexOp::create(
        builder, anchor->getLoc(), *getConstantIntValue(candidate.stride));
    materialized_stride = stride.getDefiningOp();
  }

  std::string idx_to_addr_func_name = buildIdxToAddrModule(
      candidate.module, top_sym_table, candidate.idx_view, candidate.base_addr,
      stride, candidate.forwarded_args, addr_buf_offset, addr_info.compute_type,
      candidate.elem_bytes, candidate.word_size);

  rewriteGatherScatterModule(
      candidate.indirect_op, candidate.idx_view, addr_buf_offset,
      candidate.indir_dim, addr_info.compute_type, addr_info.iab_mem_space,
      addr_info.iab_base, candidate.ind_addr_buf_dim_positions);

  if (candidate.call_site)
    updateOrchestrator(*orchestrator.symbol_table, candidate.call_site, func,
                       idx_to_addr_func_name, candidate.forwarded_args);
  else
    LDBG(1) << "no orchestrator call site for @" << func.getName()
            << "; skipping orchestrator update";

  // Drop the scaffolding this candidate no longer needs: the stride constant,
  // which lives on in the orchestrator's clone, and the indirect-index memory
  // view, which the gather/scatter function stops reading once its addresses
  // come from the IAB.
  if (materialized_stride) eraseNowDeadOp(materialized_stride);
  eraseNowDeadOp(candidate.idx_view);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct IndirectComputeGroupSplitPass
    : public impl::IndirectComputeGroupSplitPassBase<
          IndirectComputeGroupSplitPass> {
  void runOnOperation() final {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp top = getOperation();

    // Collect the candidates first.  Resolving the IAB from the device
    // description is not free, so a module holding no indirect access tile has
    // to reach the early return below without ever asking for it, keeping the
    // pass a true no-op there.
    SmallVector<ktdp::ConstructIndirectAccessTilesOp> candidate_ops;
    if (failed(collectCandidateOps(top, candidate_ops)))
      return signalPassFailure();
    if (candidate_ops.empty()) {
      LDBG(1) << "no indirect access tiles; nothing to split";
      return;
    }
    LDBG(1) << "splitting " << candidate_ops.size() << " compute group(s)";

    auto& memory_tracker = getAnalysis<MemoryTrackerAnalysis>();
    auto& device_manager = getAnalysis<ktdf_arch::DeviceManager>();
    FailureOr<AddressBufferInfo> addr_info = resolveAddressBufferInfo(
        top, memory_tracker.getMemoryTree(), device_manager);
    if (failed(addr_info)) return signalPassFailure();

    FailureOr<OrchestratorInfo> orchestrator_or_err = resolveOrchestrator(top);
    if (failed(orchestrator_or_err)) return signalPassFailure();
    OrchestratorInfo orchestrator = *orchestrator_or_err;

    // Symbol table of the top-level module, reused by every candidate to name
    // and insert its new @<name>_idx_to_addr child module.
    SymbolTable top_sym_table(top);

    for (ktdp::ConstructIndirectAccessTilesOp indirect_op : candidate_ops) {
      FailureOr<SplitCandidate> candidate =
          analyzeCandidate(indirect_op, *addr_info, orchestrator);
      if (failed(candidate)) return signalPassFailure();
      if (failed(splitCandidate(*candidate, *addr_info, orchestrator,
                                top_sym_table, memory_tracker)))
        return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createIndirectComputeGroupSplitPass() {
  return std::make_unique<IndirectComputeGroupSplitPass>();
}

}  // namespace scheduler
