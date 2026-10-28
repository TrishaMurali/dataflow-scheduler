//===-- SimplifyPipeline.h --------------------------------------*- c++ -*-===//
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
// Utilities for simplifying `ktdf.pipeline` operations, as used by the
// `simplify-pipeline` pass.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_SIMPLIFYPIPELINE_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_SIMPLIFYPIPELINE_H_

#include <mlir/IR/PatternMatch.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"

namespace mlir::ktdf {

/// Moves the stages of @p pipeline in front of their consumers.
///
/// The PrivateOp is moved to the top first. Stages are then moved upwards, an
/// SCC at a time, directly in front of their earliest consumer outside of it.
/// Stages that don't need to move keep their relative order.
///
/// @return Whether @p pipeline was modified.
auto orderStages(RewriterBase& rewriter, PipelineOp pipeline) -> bool;

/// Normalizes the dependency tokens of @p pipeline , preserving the edges.
///
/// - Tokens repeated in the in or out dependencies of a stage are deduplicated.
/// - Every stage with dependencies outputs exactly one token that no other
///   stage outputs, and its consumers await that token.
///
/// Tokens that are used by anything but the stages of @p pipeline are left
/// untouched.
///
/// @return Whether @p pipeline was modified.
auto normalizeTokens(RewriterBase& rewriter, PipelineOp pipeline) -> bool;

/// Reduces the dependency edges of @p pipeline , normalizing its tokens.
///
/// In addition to normalizeTokens(), erases every edge between two SCCs that is
/// implied by a path through another SCC. Edges within an SCC are kept.
///
/// Edges over tokens that are used by anything but the stages of @p pipeline
/// are preserved, and don't imply any other edge.
///
/// @return Whether @p pipeline was modified.
auto reduceDependencies(RewriterBase& rewriter, PipelineOp pipeline) -> bool;

/// Populates @p patterns with the patterns of the `simplify-pipeline` pass.
void populateSimplifyPipelinePatterns(
    RewritePatternSet& patterns,
    const SimplifyPipelinePassOptions& options = {});

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_SIMPLIFYPIPELINE_H_
