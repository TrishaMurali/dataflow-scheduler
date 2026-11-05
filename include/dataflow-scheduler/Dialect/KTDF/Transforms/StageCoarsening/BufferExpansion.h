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
// Buffer Expansion Support for Stage Coarsening
//
// This file provides utilities for expanding buffers when sinking loops into
// pipeline stages. Buffer expansion is necessary when a buffer is shared
// across multiple stage groups and has dependency on loop-dependent accesses.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_BUFFEREXPANSION_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_BUFFEREXPANSION_H_

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/Materializer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::ktdf {

class StageGroupingAnalysis;

//===----------------------------------------------------------------------===//
// BufferExpansionInfo
//===----------------------------------------------------------------------===//

/// Stores information about how to expand a buffer
struct BufferExpansionInfo {
  Value buffer;
  SmallVector<Value> loop_ivs;           // Ordered by loop nest
  SmallVector<Value> loop_lower_bounds;  // Lower bounds for each loop
  SmallVector<Value> loop_steps;         // Step values for each loop
  SmallVector<OpFoldResult> sizes;  // Expansion sizes (constant or dynamic)
};

//===----------------------------------------------------------------------===//
// StageCoarseningBufferCloner
//===----------------------------------------------------------------------===//

/// Buffer cloner for stage coarsening that expands buffers with additional
/// dimensions based on loop dependencies
class StageCoarseningBufferCloner : public BufferCloner {
 public:
  /// Construct a buffer cloner with expansion information
  explicit StageCoarseningBufferCloner(
      ArrayRef<const BufferExpansionInfo*> expansion_infos);

  /// Custom clone allocation operations to expand memref type with additional
  /// dimensions
  memref::AllocOp cloneAllocOp(memref::AllocOp alloc_op, OpBuilder& builder,
                               IRMapping& value_map) override;

  /// Custom clone data transfer operations to adjust indices for expanded
  /// dimensions
  ktdf::DataTransferOp cloneDataTransfer(ktdf::DataTransferOp transfer_op,
                                         OpBuilder& builder,
                                         IRMapping& value_map) override;

  /// Custom clone indirect data transfer operations to adjust indices for
  /// expanded dimensions
  ktdf::IndDataTransferOp cloneIndDataTransfer(
      ktdf::IndDataTransferOp transfer_op, OpBuilder& builder,
      IRMapping& value_map) override;

 private:
  SmallVector<const BufferExpansionInfo*> expansion_infos_;
  DenseMap<Value, const BufferExpansionInfo*> buffer_to_info_;
};

//===----------------------------------------------------------------------===//
// Buffer Expansion Analysis
//===----------------------------------------------------------------------===//

/// Analyze buffers and determine which ones need expansion when sinking loops
/// into pipeline stages.
///
/// This function identifies buffers that:
/// - Are used across multiple stage groups (shared)
/// - Are written to in at least one stage
/// - Are defined in the private region of the parent pipeline
/// - Have loop-dependent accesses
///
/// For each candidate buffer, it computes the expansion dimensions and sizes
///
/// \param loop_nest The loop nest being sunk into the pipeline
/// \param pipeline The pipeline operation
/// \param grouping The stage grouping analysis
/// \param expansion_infos Output vector to store expansion information
void PerformBufferExpansionAnalysis(
    ArrayRef<scf::ForOp> loop_nest, ktdf::PipelineOp pipeline,
    const ktdf::StageGroupingAnalysis& grouping,
    SmallVectorImpl<std::unique_ptr<BufferExpansionInfo>>& expansion_infos);

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Step 1: Identify common memrefs across stage groups
/// Returns a map from memref value to the set of group indices that use it
llvm::MapVector<Value, llvm::SmallSet<size_t, 4>> identifyCommonMemrefs(
    const ktdf::StageGroupingAnalysis& grouping);

/// Step 2: Filter candidate buffers that need expansion
/// Returns buffers that are:
/// - Used across multiple groups
/// - Written to in at least one stage
/// - Defined in the private region of the parent pipeline
SmallVector<Value> filterCandidateBuffers(
    const llvm::MapVector<Value, llvm::SmallSet<size_t, 4>>& memref_to_groups,
    const ktdf::StageGroupingAnalysis& grouping);

/// Step 3: Analyze loop dependencies for candidate buffers
/// Returns a map from buffer to the list of loop IVs it depends on
llvm::MapVector<Value, llvm::SmallVector<Value>> analyzeLoopDependencies(
    ArrayRef<Value> candidate_buffers, ArrayRef<scf::ForOp> loop_nest,
    ktdf::PipelineOp pipeline);

/// Step 4: Calculate expansion dimensions and sizes for each buffer
/// Creates BufferExpansionInfo objects for buffers that need expansion
void calculateExpansionInfo(
    const llvm::MapVector<Value, SmallVector<Value>>& buffer_to_loop_ivs,
    ArrayRef<scf::ForOp> loop_nest,
    SmallVectorImpl<std::unique_ptr<BufferExpansionInfo>>& expansion_infos);

/// Check if a value depends on a target value through its defining operation
/// and operands (recursively)
bool isValueDependentOn(Value value, Value target);

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_BUFFEREXPANSION_H_

// Made with Bob
