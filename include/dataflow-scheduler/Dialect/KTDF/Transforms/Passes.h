//===-- Passes.h ------------------------------------------------*- c++ -*-===//
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
// This file declares all ktdf dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_H_

#include <mlir/Pass/Pass.h>

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace mlir::ktdf {

auto createBroadcastPromotionPass() -> std::unique_ptr<Pass>;

auto createMapReductionPartialsPass() -> std::unique_ptr<Pass>;

auto createReductionDimChunkingPass() -> std::unique_ptr<Pass>;

auto createSplitReductionInnerOuterDimPass() -> std::unique_ptr<Pass>;

auto createReductionLoopExposurePass() -> std::unique_ptr<Pass>;

auto createStageCoarseningPass() -> std::unique_ptr<Pass>;

auto createSubsumeLinearizeIndexPass() -> std::unique_ptr<Pass>;

auto createTileSizeSelectionPass() -> std::unique_ptr<Pass>;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_H_
