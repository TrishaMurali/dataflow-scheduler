//===-- KTIRPipeline.cpp ----------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/SimplifyPipeline.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"

#define PASS_NAME "ktir-pipeline"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Pipeline pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRPIPELINEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Combines FIFO slot allocations of the same FIFO type.
class Allocator : public mlir::ktdf::PipelineBuilder::Allocator {
  // TODO: Add support for dynamic dimensions.
  //
  // canAllocate will have to ensure that the shape can be reified in the
  // PrivateOp. Since that will be canonicalized by the builder, it can do this
  // speculatively.
  //
  // allocate will need to use the reified shape computation and should combine
  // FIFO slot allocations based on type _and_ shape operands.

 public:
  auto allocate(mlir::ktdf::PipelineBuilder& builder, mlir::OpResult producer,
                mlir::ktdf::StageOp consumer) -> mlir::ktdf::FifoSlot override {
    mlir::IRRewriter rewriter(builder.getPrivateBuilder());

    const auto type = getFifoSlotType(producer, consumer);
    auto& alloc = by_type_[type];
    if (!alloc) {
      // Create a new allocation.
      alloc = mlir::ktdf::FifoAllocateOp::create(rewriter, producer.getLoc(),
                                                 {type}, {});
    } else {
      // Expand the existing allocation to have one more slot.
      const llvm::SmallVector<mlir::Type> slot_types(alloc->getNumResults() + 1,
                                                     type);
      auto old_alloc = std::exchange(
          alloc, mlir::ktdf::FifoAllocateOp::create(rewriter, producer.getLoc(),
                                                    slot_types, {}));
      rewriter.replaceOp(old_alloc, alloc->getResults().drop_front());
    }

    return mlir::cast<mlir::ktdf::FifoSlot>(alloc.getResults().front());
  }

 private:
  llvm::DenseMap<mlir::ktdf::FifoSlotType, mlir::ktdf::FifoAllocateOp> by_type_;
};

struct KTIRPipelinePass : public impl::KTIRPipelinePassBase<KTIRPipelinePass> {
  using KTIRPipelinePassBase<KTIRPipelinePass>::KTIRPipelinePassBase;

  void runOnOperation() override;

 private:
  static auto createPipeline(mlir::RewriterBase& rewriter, mlir::Location loc,
                             mlir::DominanceInfo& dominance)
      -> mlir::ktdf::PipelineOp;
};

template <class OpType>
[[nodiscard]] auto getUserOfType(mlir::Value value) -> OpType {
  for (auto* user : value.getUsers()) {
    if (auto typed = llvm::dyn_cast<OpType>(user); typed) {
      return typed;
    }
  }

  return nullptr;
}

template <class OpType>
[[nodiscard]] auto getSingleUserOfType(mlir::Value value) -> OpType {
  auto* const use = getSingleUse(value);
  return use ? mlir::dyn_cast<OpType>(use->getOwner()) : nullptr;
}

struct LowerViaMemory : mlir::OpRewritePattern<mlir::ktdf::ViaOp> {
  explicit LowerViaMemory(mlir::MLIRContext* context,
                          mlir::ktdf_arch::Mapping& mapping)
      : OpRewritePattern(context), mapping_(mapping) {}

  auto matchAndRewrite(mlir::ktdf::ViaOp via,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto stage = via->getParentOfType<mlir::ktdf::StageOp>();
    if (!stage) {
      return rewriter.notifyMatchFailure(via, "expected stage");
    }
    auto pipeline = llvm::cast<mlir::ktdf::PipelineOp>(stage->getParentOp());

    if (via.getHops().size() != 1) {
      return rewriter.notifyMatchFailure(via, "expected one hop");
    }
    const auto memory_space = mlir::dyn_cast<mlir::ktdf_arch::ResourceSpecAttr>(
        via.getHops().getValue().front());
    if (!memory_space) {
      return rewriter.notifyMatchFailure(via, "exepcted resource specifier");
    }
    auto memory = mapping_.lookup<mlir::ktdf_arch::MemoryOp>(memory_space);
    if (!memory) {
      return rewriter.notifyMatchFailure(via, "expected memory space");
    }

    auto read = via.getOperand().getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
    if (!read->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          via, "expected to be single consumer of read");
    }
    auto write = getSingleUserOfType<mlir::ktdf::WriteToFifoOp>(via);
    if (!write) {
      return rewriter.notifyMatchFailure(
          via, "expected to be single producer of write");
    }

    const auto type = via.getType();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(via, "expected static shape");
    }
    const auto alloc_type =
        mlir::MemRefType::get(type.getShape(), type.getElementType(),
                              mlir::MemRefLayoutAttrInterface{}, memory_space);

    mlir::ktdf::PrivateBuilder private_builder(pipeline, std::nullopt,
                                               rewriter.getListener());
    auto token = private_builder.createToken(via->getLoc());
    auto alloc = mlir::memref::AllocOp::create(private_builder, alloc_type);

    rewriter.setInsertionPoint(stage);
    mlir::ktdf::StageOp::create(
        rewriter, via.getLoc(), stage.getDependsIn(), {token},
        [&](mlir::OpBuilder& builder, mlir::Location loc) {
          auto store = mlir::ktdp_lowering::StoreOp::create(
              builder, loc, read, alloc, {}, type.getShape());
          store->setDiscardableAttrs(via->getRawDictionaryAttrs());
          rewriter.moveOpBefore(read, store);
        });

    rewriter.setInsertionPoint(via);
    auto load = mlir::ktdp_lowering::LoadOp::create(rewriter, via.getLoc(),
                                                    alloc, {}, type.getShape(),
                                                    type.getElementType());
    load->setDiscardableAttrs(via->getRawDictionaryAttrs());
    rewriter.replaceOp(via, load);
    rewriter.modifyOpInPlace(stage, [&]() { stage.setDependsIn({token}); });
    return llvm::success();
  }

 private:
  mlir::ktdf_arch::Mapping& mapping_;
};

struct EraseStageMapping : mlir::OpRewritePattern<mlir::ktdf::StageOp> {
  explicit EraseStageMapping(mlir::MLIRContext* context,
                             mlir::ktdf_arch::Mapping& mapping)
      : OpRewritePattern(context), mapping_(mapping) {}

  auto matchAndRewrite(mlir::ktdf::StageOp stage,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    const auto units_attr = stage.getApplicableUnitsAttr();
    if (!units_attr) {
      return rewriter.notifyMatchFailure(stage, "no units set");
    }

    auto changed = false;
    llvm::SmallVector<mlir::Attribute> exec_units;
    for (auto unit : units_attr) {
      const auto resource_spec =
          llvm::dyn_cast<mlir::ktdf_arch::ResourceSpecAttr>(unit);
      if (!resource_spec ||
          !mapping_.lookup<mlir::ktdf_arch::ExecutionUnitOp>(resource_spec)) {
        changed = true;
        continue;
      }

      exec_units.push_back(unit);
    }
    if (!changed && !exec_units.empty()) {
      return llvm::failure();
    }

    rewriter.modifyOpInPlace(stage, [&]() {
      if (exec_units.empty()) {
        stage.removeApplicableUnitsAttr();
      } else {
        stage.setApplicableUnitsAttr(rewriter.getArrayAttr(exec_units));
      }
    });
    return llvm::success();
  }

 private:
  mlir::ktdf_arch::Mapping& mapping_;
};

/// Finds the innermost perfectly nested `scf.for` from @p loop .
[[nodiscard]] auto findInnermost(mlir::scf::ForOp loop) -> mlir::scf::ForOp {
  while (loop) {
    const auto body = loop.getBody()->without_terminator();
    if (body.empty() || std::next(body.begin()) != body.end()) {
      break;
    }

    auto inner = llvm::dyn_cast<mlir::scf::ForOp>(&*body.begin());
    if (!inner) {
      break;
    }

    loop = inner;
  }

  return loop;
}

/// Finds the view @p memref is derived from by looking through casts and
/// subviews, so that different views of the same buffer compare equal.
[[nodiscard]] auto getRootView(mlir::Value memref) -> mlir::Value {
  while (auto* const op = memref.getDefiningOp()) {
    if (auto cast = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op); cast) {
      memref = cast.getSource();
    } else if (auto cast = llvm::dyn_cast<mlir::memref::MemorySpaceCastOp>(op);
               cast) {
      memref = cast.getSource();
    } else if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(op);
               subview) {
      memref = subview.getSource();
    } else {
      break;
    }
  }

  return memref;
}

/// Gets the indirect address buffer of a `ktdp_lowering.ind_(load|store)`.
[[nodiscard]] auto getIndAddrBuf(mlir::Operation* op) -> mlir::Value {
  if (auto load = llvm::dyn_cast<mlir::ktdp_lowering::IndLoadOp>(op); load) {
    return load.getIndAddrBuf();
  }
  if (auto store = llvm::dyn_cast<mlir::ktdp_lowering::IndStoreOp>(op); store) {
    return store.getIndAddrBuf();
  }

  return nullptr;
}

/// Gets the memref written by a `ktdf.data_transfer` or `ktdp_lowering.store`.
[[nodiscard]] auto getWrittenMemRef(mlir::Operation* op) -> mlir::Value {
  if (auto transfer = llvm::dyn_cast<mlir::ktdf::DataTransferOp>(op);
      transfer) {
    return transfer.getDestination();
  }
  if (auto store = llvm::dyn_cast<mlir::ktdp_lowering::StoreOp>(op); store) {
    return store.getDest();
  }

  return nullptr;
}

}  // namespace

void KTIRPipelinePass::runOnOperation() {
  if (disable_this_pass) {
    return;
  }

  mlir::func::FuncOp func = getOperation();
  mlir::IRRewriter rewriter(func);
  mlir::DominanceInfo dominance;

  // Obtain the default device and start a mapping.
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());

  // Create the pipelines.
  const auto pipelines = llvm::map_to_vector(
      func.getOps<mlir::scf::ForOp>(),
      [&](mlir::scf::ForOp outermost) -> mlir::Operation* {
        rewriter.setInsertionPointToEnd(findInnermost(outermost).getBody());
        return createPipeline(rewriter, outermost->getLoc(), dominance);
      });

  // Normalize the pipelines.
  {
    mlir::RewritePatternSet patterns(&getContext());
    populateConvertToDataTransferPatterns(patterns);
    mlir::ktdf::populateSimplifyPipelinePatterns(patterns);
    patterns.add<LowerViaMemory, EraseStageMapping>(patterns.getContext(),
                                                    mapping);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // Check that the pipelines are legal.
  {
    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<mlir::ktdp_lowering::LoadOp>();
    target.addIllegalOp<mlir::ktdp_lowering::StoreOp>();
    target.addIllegalOp<mlir::ktdp_lowering::IndLoadOp>();
    target.addIllegalOp<mlir::ktdp_lowering::IndStoreOp>();
    target.addIllegalOp<mlir::ktdf::ViaOp>();

    if (failed(mlir::applyPartialConversion(pipelines, target, {}))) {
      signalPassFailure();
      return;
    }
  }
}

auto KTIRPipelinePass::createPipeline(mlir::RewriterBase& rewriter,
                                      mlir::Location loc,
                                      mlir::DominanceInfo& dominance)
    -> mlir::ktdf::PipelineOp {
  // Create the pipeline at the end of the insertion block.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(
      rewriter.getInsertionBlock(),
      rewriter.getInsertionBlock()->without_terminator().end());
  Allocator allocator;
  mlir::ktdf::PipelineBuilder builder(rewriter, loc, &allocator);
  llvm::DenseMap<mlir::Attribute, mlir::ktdf::StageOp> store_stages;

  const auto place_store =
      [&](mlir::ktdf::PipelineBuilder& builder, mlir::Operation* op,
          mlir::Value dest) -> mlir::ktdf::PipelineBuilder::Placement {
    LDBG() << "inserting store " << *op;
    const auto memory_space = getMemorySpace(dest);
    if (!memory_space) {
      LDBG() << "  (FAILED) unable to determine memory space" << *op;
      return nullptr;
    }

    const auto [it, invalid] = store_stages.try_emplace(memory_space);
    if (invalid) {
      auto stage_builder = builder.getStageBuilder();
      it->second = mlir::ktdf::StageOp::create(stage_builder, {}, {});
      it->second.setApplicableUnitsAttr(
          stage_builder.getArrayAttr({memory_space}));
    }

    return it->second;
  };

  const auto place =
      [&](mlir::ktdf::PipelineBuilder& builder,
          mlir::Operation* op) -> mlir::ktdf::PipelineBuilder::Placement {
    // Stores go into the special store_stages that aren't considerd otherwise.
    if (auto store = llvm::dyn_cast<mlir::ktdp_lowering::StoreOp>(op); store) {
      return place_store(builder, op, store.getDest());
    }
    if (auto store = llvm::dyn_cast<mlir::ktdp_lowering::IndStoreOp>(op);
        store) {
      return place_store(builder, op, store.getBase());
    }

    // Loads go into existing or newly created, reusable stages.
    if (auto load = llvm::dyn_cast<mlir::ktdp_lowering::LoadOp>(op); load) {
      return builder.tryPlacement(getMemorySpace(load.getSource()));
    }
    if (auto load = llvm::dyn_cast<mlir::ktdp_lowering::IndLoadOp>(op); load) {
      return builder.tryPlacement(getMemorySpace(load.getBase()));
    }

    // Vias are split into hops and go into newly created stages not considered
    // for any other placements.
    if (auto via = llvm::dyn_cast<mlir::ktdf::ViaOp>(op); via) {
      if (via.getHops().size() > 1) {
        rewriter.setInsertionPoint(via);
        auto rest = mlir::ktdf::ViaOp::create(
            rewriter, via.getLoc(), via.getOperand(),
            rewriter.getArrayAttr(via.getHops().getValue().drop_back(1)));
        rewriter.modifyOpInPlace(via, [&]() {
          via.setOperand(rest);
          via.setHopsAttr(
              rewriter.getArrayAttr(via.getHops().getValue().back()));
        });
      }

      auto stage = builder.createStage();
      stage.setApplicableUnitsAttr(via.getHopsAttr());
      return {stage, true};
    }

    // Mapped operations go into existing or newly created, reusable stages.
    if (auto mapping = mlir::ktdf_arch::Mappable::getMapsTo(op); mapping) {
      return builder.tryPlacement(mapping);
    }

    // Everything else inside the insertion block may follow its users.
    return rewriter.getInsertionBlock()->getParentOp()->isAncestor(op)
               ? mlir::ktdf::PipelineBuilder::Placement::natural(op)
               : nullptr;
  };

  // Collect the indirect transfers and the indirect address buffers they use.
  auto* const block = rewriter.getInsertionBlock();
  llvm::SmallVector<mlir::Operation*> ind_ops;
  llvm::DenseSet<mlir::Value> ind_addr_bufs;
  block->walk([&](mlir::Operation* op) {
    if (auto ind_addr_buf = getIndAddrBuf(op); ind_addr_buf) {
      ind_ops.push_back(op);
      ind_addr_bufs.insert(getRootView(ind_addr_buf));
    }
  });

  // Insert all the stores and their dependencies. Stores that fill an indirect
  // address buffer are left to be co-located with their indirect transfer.
  block->walk([&](mlir::Operation* op) {
    if (auto store = llvm::dyn_cast<mlir::ktdp_lowering::StoreOp>(op);
        store && !ind_addr_bufs.contains(getRootView(store.getDest()))) {
      builder.insert({op}, place, dominance);
    } else if (llvm::isa<mlir::ktdp_lowering::IndStoreOp>(op)) {
      builder.insert({op}, place, dominance);
    }
  });

  // Move the fills of each indirect address buffer into the stage of the
  // indirect transfer that uses it, ahead of that transfer. The buffer's state
  // is local to the unit that executes the stage, so it can't cross a stage
  // boundary. A fill is moved as the op in the insertion block that contains
  // the writer (e.g. its `scf.if` guard), together with its producers.
  for (auto* const ind_op : ind_ops) {
    auto stage = builder.getStage(ind_op);
    if (!stage) {
      LDBG() << "(WARN) indirect transfer not in pipeline " << *ind_op;
      continue;
    }

    const auto ind_addr_buf = getRootView(getIndAddrBuf(ind_op));
    llvm::SmallVector<mlir::Operation*> fills;
    block->walk([&](mlir::Operation* op) {
      const auto dest = getWrittenMemRef(op);
      if (!dest || getRootView(dest) != ind_addr_buf || builder.getStage(op)) {
        return;
      }
      if (auto* const fill = block->findAncestorOpInBlock(*op);
          fill && !llvm::is_contained(fills, fill)) {
        fills.push_back(fill);
      }
    });

    const auto place_fill =
        [&](mlir::ktdf::PipelineBuilder& builder,
            mlir::Operation* op) -> mlir::ktdf::PipelineBuilder::Placement {
      return llvm::is_contained(fills, op)
                 ? mlir::ktdf::PipelineBuilder::Placement(stage)
                 : place(builder, op);
    };
    // Ops are inserted at the beginning of the stage, so insert the fills in
    // reverse to keep their order.
    for (auto* const fill : llvm::reverse(fills)) {
      LDBG() << "inserting fill " << *fill;
      builder.insert({fill}, place_fill, dominance);
    }
  }

  auto result = builder.build();
  LDBG() << "created " << result;
  return result;
}
