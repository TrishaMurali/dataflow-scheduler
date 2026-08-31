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
/// Operation lowerings for KTDFLowToDFIR pass
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_OPERATIONLOWERINGS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_OPERATIONLOWERINGS_H_

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/SymbolicStartAddress.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// Lower remaining KTDF/KTDF_Lowering operations into DFIR. Should be called on
//  the function after program units have been created.
mlir::LogicalResult runOperationLowerings(
    mlir::func::FuncOp func,
    const scheduler::SchedulerExtContext& scheduler_ctx,
    const ResourceToUnits& components, arch_view::ResourceKinds& resource_kinds,
    SymbolAllocator& symbols);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_OPERATIONLOWERINGS_H_

// Made with Bob
