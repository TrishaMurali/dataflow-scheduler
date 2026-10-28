//===-- SimplifyLinalg.h -----------------------------------------*- c++ -*-==//
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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_SIMPLIFYLINALG_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_SIMPLIFYLINALG_H_

#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/OpDefinition.h>

namespace mlir {

class RewritePatternSet;

}  // namespace mlir

namespace scheduler {

/// Matches a `linalg.generic` op that is an invariant fill.
///
/// The fill value is defined outside of @p op , and is either yielded directly
/// or read from an input through a constant map.
///
/// @retval failure       @p op is not a fill, or fills with an element of a
///                       non-constant tensor.
/// @retval OpFoldResult  The fill value, which is an attribute if constant.
[[nodiscard]] auto matchFill(mlir::linalg::GenericOp op)
    -> llvm::FailureOr<mlir::OpFoldResult>;

/// Populates @p patterns with the patterns of the `simplify-linalg` pass.
void populateSimplifyLinalgPatterns(mlir::RewritePatternSet& patterns);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_SIMPLIFYLINALG_H_
