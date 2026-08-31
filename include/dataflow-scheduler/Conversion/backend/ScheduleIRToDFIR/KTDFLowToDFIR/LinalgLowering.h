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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_LINALGLOWERING_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_LINALGLOWERING_H_

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/SymbolicStartAddress.h"
#include "mlir/IR/PatternMatch.h"

namespace scheduler {

/// Register LowerLinalgGenericPattern into the given pattern set.
void populateLinalgLoweringPatterns(mlir::RewritePatternSet& patterns,
                                    arch_view::ResourceKinds& resource_kinds,
                                    SymbolAllocator& symbols);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_LINALGLOWERING_H_
