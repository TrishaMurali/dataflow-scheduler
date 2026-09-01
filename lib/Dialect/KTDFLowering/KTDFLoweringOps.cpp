//===-- KTDFLoweringOps.cpp -------------------------------------*- c++ -*-===//
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
// This file implements the ktdf_lowering dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
// clang-format on

#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::ktdf_lowering;

//===----------------------------------------------------------------------===//
// KTDFLoweringDialect
//===----------------------------------------------------------------------===//

void KTDFLoweringDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom Assembly Format: ExecuteOnOp
//===----------------------------------------------------------------------===//

ParseResult ExecuteOnOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse operands: (unit operands)
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  llvm::SmallVector<Type, 8> operand_types;

  // If there are operands, parse them
  if (parser.parseOperandList(operands) ||
      parser.resolveOperands(operands, parser.getBuilder().getIndexType(),
                             result.operands)) {
    return failure();
  }

  // Parse the body region
  Region* body_region = result.addRegion();
  if (parser.parseRegion(*body_region, {}, {})) {
    return failure();
  }

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void ExecuteOnOp::print(OpAsmPrinter& printer) {
  // Print operands
  printer << " ";
  printer.printOperands(getUnits());

  // Print body region
  printer << " ";
  printer.printRegion(getBody());

  // Print attributes
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// Custom Assembly Format: SignalOp
//===----------------------------------------------------------------------===//

ParseResult SignalOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse operands: (unit operands)
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;

  // Parse the operand list
  if (parser.parseOperandList(operands) ||
      parser.resolveOperands(operands, parser.getBuilder().getIndexType(),
                             result.operands)) {
    return failure();
  }

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void SignalOp::print(OpAsmPrinter& printer) {
  // Print operands
  printer << " ";
  printer.printOperands(getUnits());

  // Print attributes
  printer.printOptionalAttrDict((*this)->getAttrs());
}
