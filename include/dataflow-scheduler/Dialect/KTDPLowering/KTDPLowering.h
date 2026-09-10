//===-- KTDPLowering.h -----------------------------------------*- C++ -*-===//
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
// This file includes the entire ktdp_lowering dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERING_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERING_H_

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLoweringDialect.h"  // IWYU pragma: keep
// Full KTDP ops header needed so that ktdp::RegionTerminatorOp (used as the
// SingleBlockImplicitTerminator of our hidden region) is declared before the
// generated .h.inc is parsed by the C++ compiler.
#include "ktir/Dialect/KTDP/KTDP.h"

/// Auto-generated includes.
#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h.inc"

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERING_H_
