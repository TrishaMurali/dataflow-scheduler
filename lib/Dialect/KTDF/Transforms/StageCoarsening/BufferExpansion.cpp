//===-- BufferExpansion.cpp -------------------------------------*- c++ -*-===//
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
// Buffer Expansion Implementation
//
// We need to expand buffers to make loop distribution legal. This
// implementation helps with identifying the buffers that need expansion and
// recording relevant information for the materializer to use. It identifies
// which buffers are candidate for expansion as follows:
//   1. Use the result of the stage grouping analysis and look for common
//   memrefs across groups of stages. Consider a memref as candidate for
//   expansion if it is written to in any of the stages under this pipeline.
//   2. Further check to see if the candidate is allocated in the private region
//   of the parent pipeline, otherwise ignore it.
//   3. Collect the loop variables on which the candidate memref has
//   dependency (see explanation below)...only collect loops that are in the
//   nest of candidates being distributed.
// Determine how to expand the buffers (ie what dimensions to add, what sizes,
// etc) and record relevant information
//   1. Expand the memref only in dimensions corresponding to the loops
//   collected (ie do not expand in dimensions that are loop invariant)... also
//   check if the memref already contains those dims assert/skip them.
//   2: Expand the memref in dims to the left of its current buffer and in the
//   same order as the candidate loop nest.
//
// Where the rules above come from? Consider this example:
//
// C[N] = A + B[M]
//
// S0:  for n...N
// S1:    t1 = load A
// S2:    t2 = load B[n]
// S3:    t3 = t1 + t2
// S4:    store t3, C[n]
//
// Suppose we want to distribute to {[S1,S2],S3,S4}. The questions are:
// 1. which of t1, t2, or t3 need expansion?
// 2. in what dimensions to expand so that we preserve loop invariance for reuse
// promotion opportunity in later passes.
//
// Focusing on S1, we notice that all the n iterations are using the same value
// of A (ie A's subscripts (can think of [0]) are invariant to loop-N). Each
// iteration of N reads the same A into t1, so t1 need not be expanded to enable
// loop distribution. Focusing on S2, the load of B[n] is dependent on loop-N,
// so t2 receives different data in each iteration of loop-N. To enable
// distribution t2 needs to be expanded in the loop-N dimension. Similarly, t3
// needs to be expanded in the N dimension:
//
// S1':  t1 = load A
// S2':  t2[n] = load B[n]
// S3':  t3[n] = t1 + t2[n]
// S4':  store t3[n], C[n]
//
// Without above rules, a naive transformation would have unnecessarily expanded
// t1 to t[n] as well, making it difficult to recover the invariance.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/BufferExpansion.h"

#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGrouping.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "buffer-expansion"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

// Given an AffineMap `original` of shape (d0..d_{m-1}) -> (e0, ..., e_{r-1}),
// produce a new map (d0..d_{m+n-1}) -> (d0, d1, ..., d_{n-1}, e0', ...,
// e_{r-1}') where each e_i' is e_i with its dim positions shifted by +n.
static AffineMap prependIdentityDims(AffineMap original, unsigned n,
                                     MLIRContext* ctx) {
  if (n == 0) return original;
  unsigned old_dims = original.getNumDims();
  unsigned new_dims = old_dims + n;

  llvm::SmallVector<AffineExpr> new_results;
  new_results.reserve(n + original.getNumResults());

  // Leading identity dims for the prepended indices.
  for (unsigned i = 0; i < n; ++i) {
    new_results.push_back(getAffineDimExpr(i, ctx));
  }

  // Original results, shifted up by n.
  for (AffineExpr e : original.getResults()) {
    new_results.push_back(e.shiftDims(/*numDims=*/old_dims, /*shift=*/n));
  }

  return AffineMap::get(new_dims, /*symbolCount=*/0, new_results, ctx);
}

// Compute the buffer dimension size for a loop.
// Returns an OpFoldResult containing either:
//   - An IntegerAttr if the size is statically known
//   - A Value if the size is dynamic
//
// Strategy:
// 1. First try to compute a constant trip count using ceildiv formula
// 2. If not constant, assert the loop is normalized (lb=0, step=1) and:
//    a. If upper bound is from ktdf.tiling.derive_size, extract the tile size
//       from the right-most [iv : size] pair. This would be the maximum size of
//       that tile dimension.
//    b. Otherwise, use the upper bound directly as the trip count
static OpFoldResult computeLoopSize(scf::ForOp loop, OpBuilder& builder) {
  Value lower = loop.getLowerBound();
  Value upper = loop.getUpperBound();
  Value step = loop.getStep();

  // Try to compute constant trip count if all bounds are constant
  std::optional<int64_t> lower_const, upper_const, step_const;
  if (auto cst = lower.getDefiningOp<arith::ConstantIndexOp>())
    lower_const = cst.value();
  if (auto cst = upper.getDefiningOp<arith::ConstantIndexOp>())
    upper_const = cst.value();
  if (auto cst = step.getDefiningOp<arith::ConstantIndexOp>())
    step_const = cst.value();

  if (lower_const && upper_const && step_const) {
    // Compute constant trip count using ceildiv formula
    int64_t trip_count =
        (*upper_const - *lower_const + *step_const - 1) / *step_const;
    IntegerAttr size_attr = builder.getI64IntegerAttr(trip_count);
    LDBG(1) << "      Dimension size (constant): " << trip_count;
    return OpFoldResult(size_attr);
  }

  // Dynamic case - assert that the loop is normalized (lb == 0 and step == 1)
  auto lower_const_op = lower.getDefiningOp<arith::ConstantIndexOp>();
  auto step_const_op = step.getDefiningOp<arith::ConstantIndexOp>();

  assert(lower_const_op && lower_const_op.value() == 0 &&
         "Expected loop lower bound to be 0 (normalized loop)");
  assert(step_const_op && step_const_op.value() == 1 &&
         "Expected loop step to be 1 (normalized loop)");

  // Determine the trip count based on the upper bound
  Value size_val;

  // Check if upper bound is the result of ktdf.tiling.derive_size
  if (auto derive_size_op = upper.getDefiningOp<ktdf::TilingDeriveSizeOp>()) {
    // Use the size from the right-most [iv : size] pair
    auto tile_sizes = derive_size_op.getTileSizes();
    assert(!tile_sizes.empty() &&
           "Expected at least one tile size in ktdf.tiling.derive_size");
    size_val = tile_sizes.back();
    LDBG(1) << "      Dimension size from ktdf.tiling.derive_size";
  } else {
    // Use the upper bound directly as the trip count
    size_val = upper;
    LDBG(1) << "      Dimension size from upper bound";
  }

  // Dynamic size - return the Value directly
  LDBG(1) << "      Dimension size (dynamic)";
  return OpFoldResult(size_val);
}

}  // namespace

//===----------------------------------------------------------------------===//
// StageCoarseningBufferCloner Implementation
//===----------------------------------------------------------------------===//

ktdf::StageCoarseningBufferCloner::StageCoarseningBufferCloner(
    llvm::ArrayRef<const BufferExpansionInfo*> expansion_infos)
    : expansion_infos_(expansion_infos.begin(), expansion_infos.end()) {
  // Build a map from original buffer to expansion info for quick lookup
  for (const BufferExpansionInfo* info : expansion_infos_) {
    assert(info);
    buffer_to_info_[info->buffer] = info;
  }
}

memref::AllocOp StageCoarseningBufferCloner::cloneAllocOp(
    memref::AllocOp alloc_op, OpBuilder& builder, IRMapping& value_map) {
  // Get the result of the allocation
  Value original_result = alloc_op.getResult();

  // Check if this buffer needs expansion
  // The buffer might be identified by its yielded value from ktdf.private,
  // so we need to check if this allocation is yielded and matches a candidate
  const BufferExpansionInfo* info = nullptr;
  auto it = buffer_to_info_.find(original_result);
  if (it != buffer_to_info_.end()) {
    info = it->second;
  } else {
    // Check if this allocation is yielded by a private operation
    // and the yielded value is a candidate for expansion
    for (Operation* user : original_result.getUsers()) {
      auto yield_op = dyn_cast<ktdf::PrivateYieldOp>(user);
      if (!yield_op) continue;
      // Find which operand index this is
      for (auto [idx, operand] : llvm::enumerate(yield_op.getOperands())) {
        if (operand != original_result) continue;
        // Get the corresponding result from the parent private operation
        auto private_op = cast<ktdf::PrivateOp>(yield_op->getParentOp());
        Value yielded_result = private_op.getResult(idx);

        // Check if this yielded result is a candidate
        auto yielded_it = buffer_to_info_.find(yielded_result);
        if (yielded_it != buffer_to_info_.end()) {
          info = yielded_it->second;
          break;
        }
      }
      if (info) break;
    }
  }

  if (!info) {
    // No expansion needed, use default behavior
    return nullptr;
  }

  auto original_type = alloc_op.getType();

  LDBG(1) << "  Expanding buffer allocation with " << info->sizes.size()
          << " new dimension(s)";

  // Build new shape: [expansion_dims..., original_dims...]
  llvm::SmallVector<int64_t> new_shape;
  llvm::SmallVector<Value> dynamic_sizes;

  // Add expansion dimensions (to the left)
  for (const OpFoldResult& size : info->sizes) {
    if (auto attr = dyn_cast<Attribute>(size)) {
      // Constant size
      auto int_attr = cast<IntegerAttr>(attr);
      new_shape.push_back(int_attr.getInt());
    } else {
      // Dynamic size
      new_shape.push_back(ShapedType::kDynamic);
      Value size_val = cast<Value>(size);
      // Map the size value if it's been cloned
      if (value_map.contains(size_val)) {
        size_val = value_map.lookup(size_val);
      }
      dynamic_sizes.push_back(size_val);
    }
  }

  // Add original dimensions
  for (int64_t dim : original_type.getShape()) {
    new_shape.push_back(dim);
    if (!ShapedType::isDynamic(dim)) continue;
    // Original dimension was dynamic - need to get its size
    // For memref.alloc, dynamic sizes are operands
    // Find the corresponding dynamic size operand
    unsigned dynamic_idx = 0;
    for (int64_t orig_dim : original_type.getShape()) {
      if (!ShapedType::isDynamic(orig_dim)) continue;
      if (dynamic_idx < alloc_op.getDynamicSizes().size()) {
        Value orig_size = alloc_op.getDynamicSizes()[dynamic_idx];
        if (value_map.contains(orig_size)) {
          orig_size = value_map.lookup(orig_size);
        }
        dynamic_sizes.push_back(orig_size);
      }
      dynamic_idx++;
    }
  }

  // Create new memref type with expanded dimensions
  MemRefType new_type = MemRefType::get(
      new_shape, original_type.getElementType(), MemRefLayoutAttrInterface(),
      original_type.getMemorySpace());

  // Create new allocation with expanded type
  auto new_alloc = memref::AllocOp::create(builder, alloc_op.getLoc(), new_type,
                                           dynamic_sizes);

  LDBG(1) << "    Original type: " << original_type;
  LDBG(1) << "    Expanded type: " << new_type;

  return new_alloc;
}

ktdf::DataTransferOp StageCoarseningBufferCloner::cloneDataTransfer(
    DataTransferOp transfer_op, OpBuilder& builder, IRMapping& value_map) {
  // Get source and destination buffers
  Value src = transfer_op.getSource();
  Value dst = transfer_op.getDestination();

  // Map to cloned values
  if (value_map.contains(src)) {
    src = value_map.lookup(src);
  }
  if (value_map.contains(dst)) {
    dst = value_map.lookup(dst);
  }

  // Check if either buffer was expanded
  const BufferExpansionInfo* src_info = nullptr;
  const BufferExpansionInfo* dst_info = nullptr;

  for (const BufferExpansionInfo* info : expansion_infos_) {
    Value mapped_buffer = info->buffer;
    if (value_map.contains(mapped_buffer)) {
      mapped_buffer = value_map.lookup(mapped_buffer);
    }
    if (mapped_buffer == src) {
      src_info = info;
    }
    if (mapped_buffer == dst) {
      dst_info = info;
    }
  }

  if (!src_info && !dst_info) {
    // No expansion needed
    return nullptr;
  }

  LDBG(1) << "  Adjusting data transfer indices for expanded buffer(s)";

  // Build new index lists with loop IV values prepended
  llvm::SmallVector<Value> new_src_indices;
  llvm::SmallVector<Value> new_dst_indices;

  // Lambda to compute normalized indices for expanded dimensions
  auto computeExpandedIndices = [&](const BufferExpansionInfo* info,
                                    llvm::SmallVector<Value>& indices) {
    if (!info) return;

    // Prepend loop IV values for expanded dimensions
    // Need to compute normalized index: (iv - lower_bound) / step
    for (size_t i = 0; i < info->loop_ivs.size(); ++i) {
      Value loop_iv = info->loop_ivs[i];
      Value lower_bound = info->loop_lower_bounds[i];
      Value step = info->loop_steps[i];

      // Map values if they've been cloned
      if (value_map.contains(loop_iv)) {
        loop_iv = value_map.lookup(loop_iv);
      }
      if (value_map.contains(lower_bound)) {
        lower_bound = value_map.lookup(lower_bound);
      }
      if (value_map.contains(step)) {
        step = value_map.lookup(step);
      }

      // Compute normalized index: (iv - lower_bound) / step
      Location loc = transfer_op.getLoc();
      Value offset =
          arith::SubIOp::create(builder, loc, loop_iv, lower_bound).getResult();
      Value normalized_idx =
          arith::DivSIOp::create(builder, loc, offset, step).getResult();

      indices.push_back(normalized_idx);
    }
  };

  // Compute expanded indices for source
  computeExpandedIndices(src_info, new_src_indices);

  // Add original source indices
  for (Value idx : transfer_op.getSourceIndices()) {
    Value mapped_idx = idx;
    if (value_map.contains(idx)) {
      mapped_idx = value_map.lookup(idx);
    }
    new_src_indices.push_back(mapped_idx);
  }

  // Compute expanded indices for destination
  computeExpandedIndices(dst_info, new_dst_indices);

  // Add original destination indices
  for (Value idx : transfer_op.getDestIndices()) {
    Value mapped_idx = idx;
    if (value_map.contains(idx)) {
      mapped_idx = value_map.lookup(idx);
    }
    new_dst_indices.push_back(mapped_idx);
  }

  // Create new data transfer with adjusted indices
  // Get the mixed sizes (OpFoldResult) from the original transfer
  llvm::SmallVector<OpFoldResult> src_sizes = transfer_op.getMixedSourceSizes();
  llvm::SmallVector<OpFoldResult> dst_sizes = transfer_op.getMixedDestSizes();

  // Extend the source/dest affine maps with identity dims for the
  // expansion-prepended indices, in lockstep with new_src_indices /
  // new_dst_indices growth above.
  unsigned src_prepend = src_info ? src_info->loop_ivs.size() : 0;
  unsigned dst_prepend = dst_info ? dst_info->loop_ivs.size() : 0;
  MLIRContext* ctx = transfer_op->getContext();

  AffineMap new_src_map;
  if (transfer_op.isSourceMemRef()) {
    AffineMap orig = transfer_op.getSourceMapAttr().getValue();
    new_src_map = prependIdentityDims(orig, src_prepend, ctx);
  }
  AffineMap new_dst_map;
  if (transfer_op.isDestMemRef()) {
    AffineMap orig = transfer_op.getDestMapAttr().getValue();
    new_dst_map = prependIdentityDims(orig, dst_prepend, ctx);
  }

  // Prepend a size of 1 for each expansion dimension, matching the prepended
  // indices above. Each expanded dimension transfers one tile slot at a time.
  OpFoldResult one = IntegerAttr::get(IndexType::get(ctx), 1);
  src_sizes.insert(src_sizes.begin(), src_prepend, one);
  dst_sizes.insert(dst_sizes.begin(), dst_prepend, one);

  auto new_transfer = DataTransferOp::create(
      builder, transfer_op.getLoc(), src, new_src_map, new_src_indices,
      src_sizes, dst, new_dst_map, new_dst_indices, dst_sizes,
      transfer_op.getTransferMode());

  // Copy what the structural builder arguments do not cover.
  new_transfer->setDiscardableAttrs(transfer_op->getRawDictionaryAttrs());

  return new_transfer;
}

//===----------------------------------------------------------------------===//
// Buffer Expansion Analysis - Helper Functions
//===----------------------------------------------------------------------===//

llvm::MapVector<Value, llvm::SmallSet<size_t, 4>>
mlir::ktdf::identifyCommonMemrefs(const StageGroupingAnalysis& grouping) {
  llvm::MapVector<Value, llvm::SmallSet<size_t, 4>> memref_to_groups;
  const llvm::SmallVector<ktdf::StageGroup>& groups = grouping.getGroups();

  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const StageGroup& group = groups[group_idx];

    for (ktdf::StageOp stage : group.getStages()) {
      // Walk the stage body to find all memref uses
      stage.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
          if (isa<MemRefType>(operand.getType())) {
            memref_to_groups[operand].insert(group_idx);
          }
        }
      });
    }
  }

  LDBG(1) << "    Found " << memref_to_groups.size()
          << " unique memrefs across all stages";

  return memref_to_groups;
}

llvm::SmallVector<Value> mlir::ktdf::filterCandidateBuffers(
    const llvm::MapVector<Value, llvm::SmallSet<size_t, 4>>& memref_to_groups,
    const StageGroupingAnalysis& grouping) {
  llvm::SmallVector<Value> candidate_buffers;
  const llvm::SmallVector<ktdf::StageGroup>& groups = grouping.getGroups();

  for (const auto& entry : memref_to_groups) {
    Value memref = entry.first;
    const llvm::SmallSet<size_t, 4>& group_set = entry.second;

    // Check if used across multiple groups
    if (group_set.size() <= 1) {
      continue;
    }

    // Check if written to in any stage
    bool is_written = false;
    for (size_t group_idx : group_set) {
      const StageGroup& group = groups[group_idx];
      for (ktdf::StageOp stage : group.getStages()) {
        if (ktdf::isTransferTarget(memref, stage.getOperation())) {
          is_written = true;
          break;
        }
      }
      if (is_written) break;
    }

    if (!is_written) {
      continue;
    }

    // Check if allocated in private region
    auto defining_op = memref.getDefiningOp();
    if (!defining_op) {
      continue;
    }

    // Check if it's coming from a ktdf.private operation
    if (!isa<ktdf::PrivateOp>(defining_op)) {
      continue;
    }

    // This is a valid candidate
    if (std::find(candidate_buffers.begin(), candidate_buffers.end(), memref) ==
        candidate_buffers.end())
      candidate_buffers.push_back(memref);

    LDBG(1) << "    Candidate buffer found: used in " << group_set.size()
            << " groups";
  }

  LDBG(1) << "    Identified " << candidate_buffers.size()
          << " candidate buffers for expansion";

  return candidate_buffers;
}

llvm::MapVector<Value, llvm::SmallVector<Value>>
mlir::ktdf::analyzeLoopDependencies(llvm::ArrayRef<Value> candidate_buffers,
                                    llvm::ArrayRef<scf::ForOp> loop_nest,
                                    PipelineOp pipeline) {
  llvm::MapVector<Value, llvm::SmallVector<Value>> buffer_to_loop_ivs;

  // Build set of loop IVs from candidate loop nest for quick lookup
  llvm::SmallVector<Value> candidate_loop_ivs;
  for (scf::ForOp loop : loop_nest) {
    candidate_loop_ivs.push_back(loop.getInductionVar());
  }

  for (Value buffer : candidate_buffers) {
    llvm::SmallVector<Value> dependent_ivs;

    // Find all data transfer operations that use this buffer
    pipeline.walk([&](ktdf::DataTransferOp transfer) {
      // Check if this transfer uses the buffer
      if (transfer.getSource() != buffer &&
          transfer.getDestination() != buffer) {
        return;
      }

      // Analyze the indices used in the transfer
      // For source indices
      for (Value idx : transfer.getSourceIndices()) {
        // Check if this index depends on any candidate loop IV
        for (Value loop_iv : candidate_loop_ivs) {
          if (isValueDependentOn(idx, loop_iv)) {
            // Add to dependent IVs if not already present
            if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) {
              dependent_ivs.push_back(loop_iv);
            }
          }
        }
      }

      // For destination indices
      for (Value idx : transfer.getDestIndices()) {
        for (Value loop_iv : candidate_loop_ivs) {
          if (isValueDependentOn(idx, loop_iv)) {
            if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) {
              dependent_ivs.push_back(loop_iv);
            }
          }
        }
      }
    });

    if (!dependent_ivs.empty()) {
      buffer_to_loop_ivs[buffer] = dependent_ivs;
      LDBG(1) << "    Buffer depends on " << dependent_ivs.size()
              << " loop IV(s)";
    }
  }

  return buffer_to_loop_ivs;
}

void mlir::ktdf::calculateExpansionInfo(
    const llvm::MapVector<Value, llvm::SmallVector<Value>>& buffer_to_loop_ivs,
    llvm::ArrayRef<scf::ForOp> loop_nest,
    llvm::SmallVectorImpl<std::unique_ptr<BufferExpansionInfo>>&
        expansion_infos) {
  for (const auto& entry : buffer_to_loop_ivs) {
    Value buffer = entry.first;
    const llvm::SmallVector<Value>& dependent_ivs = entry.second;

    auto info = std::make_unique<BufferExpansionInfo>();
    info->buffer = buffer;

    // Order the dependent IVs according to the loop nest order
    for (scf::ForOp loop : loop_nest) {
      Value loop_iv = loop.getInductionVar();
      if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) continue;
      info->loop_ivs.push_back(loop_iv);

      // Calculate the size for this dimension
      // Size = (upper_bound - lower_bound + step - 1) / step
      Value lower = loop.getLowerBound();
      Value step = loop.getStep();

      // Store loop bounds and step for later index computation
      info->loop_lower_bounds.push_back(lower);
      info->loop_steps.push_back(step);

      // Compute the size for this dimension
      OpBuilder builder(loop);
      OpFoldResult size = computeLoopSize(loop, builder);
      info->sizes.push_back(size);
    }

    if (!info->loop_ivs.empty()) {
      LDBG(1) << "    Buffer expansion: " << info->loop_ivs.size()
              << " new dimension(s)";

      // Assert that all users of this buffer are data transfer operations.
      // If other operations use a buffer that is a candidate for expansion,
      // we would likely require custom cloners for them as well.
      for (Operation* user : buffer.getUsers()) {
        assert(isa<ktdf::DataTransferOp>(user) &&
               "Expected all users of buffers marked for expansion to be "
               "ktdf::DataTransferOp.");
      }

      expansion_infos.push_back(std::move(info));
    }
  }

  if (expansion_infos.empty()) {
    LDBG(1) << "    No buffers need expansion";
  } else {
    LDBG(1) << "    Total buffers to expand: " << expansion_infos.size();
  }
}

//===----------------------------------------------------------------------===//
// Buffer Expansion Analysis - Main Function
//===----------------------------------------------------------------------===//

void mlir::ktdf::PerformBufferExpansionAnalysis(
    llvm::ArrayRef<scf::ForOp> loop_nest, PipelineOp pipeline,
    const StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<std::unique_ptr<BufferExpansionInfo>>&
        expansion_infos) {
  LDBG(1) << "  Performing buffer expansion analysis...";

  // Step 1: Identify common memrefs across stage groups
  auto memref_to_groups = identifyCommonMemrefs(grouping);

  // Step 2: Filter for candidate buffers that need expansion
  auto candidate_buffers = filterCandidateBuffers(memref_to_groups, grouping);

  // Step 3: Analyze loop dependencies for each candidate buffer
  auto buffer_to_loop_ivs =
      analyzeLoopDependencies(candidate_buffers, loop_nest, pipeline);

  // Step 4: Calculate expansion dimensions and sizes
  calculateExpansionInfo(buffer_to_loop_ivs, loop_nest, expansion_infos);
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Internal recursive helper that carries a visited set to avoid
/// re-traversing shared sub-graphs (e.g. common index sub-expressions).
static bool isValueDependentOnImpl(Value value, Value target,
                                   llvm::DenseSet<Value>& visited) {
  if (value == target) return true;
  if (!visited.insert(value).second) return false;

  Operation* defining_op = value.getDefiningOp();
  if (!defining_op) return false;

  for (Value operand : defining_op->getOperands()) {
    if (isValueDependentOnImpl(operand, target, visited)) return true;
  }
  return false;
}

bool mlir::ktdf::isValueDependentOn(Value value, Value target) {
  llvm::DenseSet<Value> visited;
  return isValueDependentOnImpl(value, target, visited);
}

// Made with Bob
