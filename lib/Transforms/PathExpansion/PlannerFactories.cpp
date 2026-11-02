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
// Planner Factory Implementations
// - PrivateResourceFactory: Manages private resource specs
// - TransferInfoFactory: Manages transfer materialization info
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/PathExpansion/PlannerFactories.h"

#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace scheduler;

namespace scheduler {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Move \p object into \p owned and return a pointer to it. The factories use
/// this to keep every object they create alive for as long as they live.
template <typename T>
static T* takeOwnership(llvm::SmallVectorImpl<std::unique_ptr<T>>& owned,
                        std::unique_ptr<T> object) {
  T* ptr = object.get();
  owned.push_back(std::move(object));
  return ptr;
}

/// Helper to create zero indices for an intermediate buffer
/// Used when a FIFO (which has no indices) is replaced by a memref buffer
static void createIndicesForIntermediateBuffer(
    const PrivateResourceSpec* buffer, mlir::Operation* contextOp,
    mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& indices) {
  // Find the parent function to set insertion point
  auto funcOp = contextOp->getParentOfType<mlir::func::FuncOp>();
  assert(funcOp && "Expected operation to be within a function");

  // Set insertion point at the beginning of the function's entry block
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  mlir::Value zero_index =
      mlir::arith::ConstantIndexOp::create(builder, contextOp->getLoc(), 0);
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    indices.push_back(zero_index);
  }
}

/// Append to \p sizes the size of the whole of \p spec: the shape of a buffer,
/// or the capacity of slot \p slot_index of a FIFO. Does nothing if \p spec
/// is null or stands for an existing value.
static void deriveSizesFromResourceSpec(
    const PrivateResourceSpec* spec, size_t slot_index,
    mlir::MLIRContext* context,
    llvm::SmallVectorImpl<mlir::OpFoldResult>& sizes) {
  if (!spec) return;

  mlir::Builder builder(context);
  if (spec->kind == PrivateResourceSpec::Kind::kFifo) {
    // For FIFO, use the specific slot's capacity
    assert(slot_index < spec->elements_per_slot.size() &&
           "Slot index out of bounds");
    int64_t slot_capacity = spec->elements_per_slot[slot_index];
    sizes.push_back(builder.getI64IntegerAttr(slot_capacity));
  } else if (spec->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
    // For memory buffer, use the shape
    for (int64_t dim : spec->shape) {
      sizes.push_back(builder.getI64IntegerAttr(dim));
    }
  }
}

//===----------------------------------------------------------------------===//
// PrivateResourceFactory Implementation
//===----------------------------------------------------------------------===//

PrivateResourceSpec* PrivateResourceFactory::createMemoryBuffer(
    ResourceType memory_resource, llvm::ArrayRef<int64_t> shape,
    mlir::Type element_type) {
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kMemoryBuffer;
  spec->memory_resource = memory_resource;
  spec->shape.assign(shape.begin(), shape.end());
  spec->element_type = element_type;

  return takeOwnership(specs_, std::move(spec));
}

PrivateResourceSpec* PrivateResourceFactory::createFifo(
    mlir::Attribute fifo_src, mlir::Attribute fifo_dest,
    llvm::ArrayRef<int64_t> elements_per_slot, mlir::Type element_type) {
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kFifo;
  spec->fifo_src = fifo_src;
  spec->fifo_dest = fifo_dest;
  spec->elements_per_slot.assign(elements_per_slot.begin(),
                                 elements_per_slot.end());
  spec->element_type = element_type;

  return takeOwnership(specs_, std::move(spec));
}

PrivateResourceSpec* PrivateResourceFactory::createExistingValue(
    mlir::Value value) {
  auto memref_type = mlir::cast<mlir::MemRefType>(value.getType());
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kExistingValue;
  spec->existing_value = value;
  spec->memory_resource =
      llvm::cast<ResourceType>(memref_type.getMemorySpace());
  spec->shape.assign(memref_type.getShape().begin(),
                     memref_type.getShape().end());
  spec->element_type = memref_type.getElementType();

  return takeOwnership(specs_, std::move(spec));
}

void PrivateResourceFactory::registerAllocation(
    const PrivateResourceSpec* spec, llvm::ArrayRef<mlir::Value> ssa_values) {
  auto alloc = std::make_unique<PrivateResourceAllocation>(spec);
  alloc->ssa_values.assign(ssa_values.begin(), ssa_values.end());
  allocations_[spec] = std::move(alloc);
}

const PrivateResourceAllocation* PrivateResourceFactory::getAllocation(
    const PrivateResourceSpec* spec) const {
  auto it = allocations_.find(spec);
  return it != allocations_.end() ? it->second.get() : nullptr;
}

//===----------------------------------------------------------------------===//
// TransferInfoFactory Implementation
//===----------------------------------------------------------------------===//

/// Return the source operand of \p op if \p source_side, else its destination
/// operand, and how the op addresses it. \p op must be a DataTransferOp or an
/// IndDataTransferOp; for an indirect transfer this is the direct operand,
/// not the address operand.
static std::pair<mlir::Value, TransferSide> getTemplateOperand(
    mlir::Operation* op, bool source_side) {
  auto mapOf = [](mlir::AffineMapAttr attr) {
    return attr ? attr.getValue() : mlir::AffineMap();
  };
  mlir::Value value;
  TransferSide side;
  if (auto dt = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
    mlir::OperandRange indices =
        source_side ? dt.getSourceIndices() : dt.getDestIndices();
    value = source_side ? dt.getSource() : dt.getDestination();
    side.indices.assign(indices.begin(), indices.end());
    side.sizes =
        source_side ? dt.getMixedSourceSizes() : dt.getMixedDestSizes();
    side.map = mapOf(source_side ? dt.getSourceMapAttr() : dt.getDestMapAttr());
    return {value, std::move(side)};
  }
  auto ind = mlir::cast<mlir::ktdf::IndDataTransferOp>(op);
  mlir::OperandRange indices =
      source_side ? ind.getDirSrcIndices() : ind.getDirDstIndices();
  value = source_side ? ind.getDirSrc() : ind.getDirDst();
  side.indices.assign(indices.begin(), indices.end());
  side.sizes =
      source_side ? ind.getMixedDirSrcSizes() : ind.getMixedDirDstSizes();
  side.map =
      mapOf(source_side ? ind.getDirSrcMapAttr() : ind.getDirDstMapAttr());
  return {value, std::move(side)};
}

TransferSide TransferInfoFactory::describeTemplateSide(mlir::Operation* op,
                                                       bool source_side) {
  return getTemplateOperand(op, source_side).second;
}

TransferSide TransferInfoFactory::describePrivateSide(
    const PrivateResourceSpec* spec, size_t slot_index,
    mlir::Operation* context_op, mlir::OpBuilder& builder) {
  TransferSide side;
  side.resource = spec;
  side.slot_index = slot_index;
  deriveSizesFromResourceSpec(spec, slot_index, builder.getContext(),
                              side.sizes);
  if (spec->kind != PrivateResourceSpec::Kind::kMemoryBuffer) return side;
  createIndicesForIntermediateBuffer(spec, context_op, builder, side.indices);
  side.map = mlir::AffineMap::getMultiDimIdentityMap(spec->shape.size(),
                                                     builder.getContext());
  return side;
}

void TransferInfoFactory::retargetSide(TransferMaterializationInfo* transfer,
                                       bool source_side,
                                       const PrivateResourceSpec* spec,
                                       size_t slot_index,
                                       mlir::OpBuilder& builder) {
  mlir::Operation* op = transfer->template_op;
  mlir::Value operand = getTemplateOperand(op, source_side).first;
  TransferSide& side = transfer->side(source_side);
  // A FIFO operand replaced by another FIFO keeps its indices and sizes.
  if (spec->kind == PrivateResourceSpec::Kind::kFifo &&
      mlir::isa<mlir::ktdf::FifoSlotType>(operand.getType())) {
    side.resource = spec;
    side.slot_index = slot_index;
    return;
  }
  side = describePrivateSide(spec, slot_index, op, builder);
}

TransferMaterializationInfo* TransferInfoFactory::createFromTemplate(
    mlir::Operation* template_op,
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType source_resource, ResourceType dest_resource) {
  assert((mlir::isa<mlir::ktdf::DataTransferOp, mlir::ktdf::IndDataTransferOp>(
             template_op)) &&
         "template_op must be a DataTransferOp or IndDataTransferOp");

  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = template_op;
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;
  transfer->source = describeTemplateSide(template_op, /*source_side=*/true);
  transfer->dest = describeTemplateSide(template_op, /*source_side=*/false);

  return takeOwnership(transfers_, std::move(transfer));
}

TransferMaterializationInfo* TransferInfoFactory::createFromFifoOp(
    mlir::Operation* fifo_op,
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType source_resource, ResourceType dest_resource,
    const PrivateResourceSpec* fifo_spec, size_t slot_index, bool is_read) {
  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = fifo_op;
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;

  // The FIFO is the source of a read_from_fifo and the destination of a
  // write_to_fifo. FIFO ops have no indices, sizes or maps, so those stay
  // empty on both sides.
  TransferSide& fifo_side = transfer->side(/*source_side=*/is_read);
  fifo_side.resource = fifo_spec;
  fifo_side.slot_index = slot_index;

  return takeOwnership(transfers_, std::move(transfer));
}

TransferMaterializationInfo* TransferInfoFactory::createSynthetic(
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType source_resource, ResourceType dest_resource,
    const TransferSide& source, const TransferSide& dest,
    mlir::MLIRContext* context) {
  assert(source.resource && dest.resource &&
         "Synthetic transfer must have private resources on both sides");

  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;
  transfer->source = source;
  transfer->dest = dest;

  // Fill in what a side leaves unset with the whole buffer or FIFO slot.
  for (TransferSide* side : {&transfer->source, &transfer->dest}) {
    if (side->sizes.empty())
      deriveSizesFromResourceSpec(side->resource, side->slot_index, context,
                                  side->sizes);
    if (!side->map && side->resource->kind != PrivateResourceSpec::Kind::kFifo)
      side->map = mlir::AffineMap::getMultiDimIdentityMap(
          side->resource->shape.size(), context);
  }

  return takeOwnership(transfers_, std::move(transfer));
}

}  // namespace scheduler

// Made with Bob