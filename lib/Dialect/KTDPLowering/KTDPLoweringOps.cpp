//===-- KTDPLoweringOps.cpp -------------------------------------*- C++ -*-===//
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
// This file implements the ktdp_lowering dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
// clang-format on

#include <llvm/ADT/SetVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

using namespace mlir;
using namespace mlir::ktdp_lowering;

//===----------------------------------------------------------------------===//
// KTDPLoweringDialect — op registration
//===----------------------------------------------------------------------===//

void KTDPLoweringDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp — builder
//===----------------------------------------------------------------------===//

// Build method contract
// ---------------------
// `perDimSubscriptMaps` must already be canonicalized over the unified
// dimension ordering (capturedVariables..., intermediateVariables...).
// `indAddrBufDimPositions[i]` is the index of the i-th IAB dimension in that
// same unified ordering.
// `numIntermediateVariables` must equal variablesSpaceSet.getNumDims().
void ConstructIndirectAccessTileOp::build(
    OpBuilder& builder, OperationState& result, ktdp::AccessTileType resultType,
    Value base, Value indAddrBufMemref,
    DenseI32ArrayAttr indAddrBufDimPositions, ArrayAttr perDimSubscriptMaps,
    ValueRange capturedVariables, unsigned numIntermediateVariables,
    AffineMap variablesSpaceOrder, IntegerSet variablesSpaceSet) {
  assert(
      variablesSpaceSet.getNumDims() == variablesSpaceOrder.getNumInputs() &&
      "variables_space_order input count must match variables_space_set dims");
  assert(variablesSpaceOrder.getNumInputs() ==
             variablesSpaceOrder.getNumResults() &&
         "variables_space_order must have equal input and output dimensions");
  assert(variablesSpaceSet.getNumDims() == numIntermediateVariables &&
         "numIntermediateVariables must equal variables_space_set dims");

  result.addOperands(base);
  result.addOperands(indAddrBufMemref);
  result.addOperands(capturedVariables);
  auto& props = result.getOrAddProperties<Properties>();
  props.ind_addr_buf_dim_positions = indAddrBufDimPositions;
  props.per_dim_subscript_maps = perDimSubscriptMaps;
  props.variables_space_order = AffineMapAttr::get(variablesSpaceOrder);
  props.variables_space_set = IntegerSetAttr::get(variablesSpaceSet);

  // Hidden region: one index-typed block arg per intermediate variable.
  // ensureTerminator is called before adding block arguments, matching the
  // pattern established by ConstructIndirectAccessTilesOp in the ktdp dialect.
  Region* region = result.addRegion();
  Block& body = region->emplaceBlock();
  ensureTerminator(*region, builder, builder.getUnknownLoc());
  for (unsigned i = 0; i < numIntermediateVariables; ++i)
    body.addArgument(builder.getIndexType(), builder.getUnknownLoc());

  result.types.push_back(resultType);
}

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp — custom assembly format
//
// Printed form:
//
//   ktdp_lowering.construct_indirect_access_tile
//       intermediate_variables(%iv0, %iv1, ...)
//       base_ptr = %iab[%var_at_pos0, %var_at_pos1, ...]
//       %base[(affine-expr), (affine-expr), ...]
//       { attr-dict }
//       : type($base), type($ind_addr_buf_memref) -> qualified(type($result))
//
// `intermediate_variables` names the hidden region's block arguments.
// IAB subscripts are printed as the SSA names from the unified variable list
// (captured..., intermediate...) at the positions stored in
// `ind_addr_buf_dim_positions`.
// Per-dimension base subscripts use `(affine-expr)` syntax, with maps over the
// unified dimension space.
//===----------------------------------------------------------------------===//

ParseResult ConstructIndirectAccessTileOp::parse(OpAsmParser& parser,
                                                 OperationState& result) {
  auto& builder = parser.getBuilder();
  MLIRContext* ctx = builder.getContext();

  // --- intermediate_variables(%iv0, %iv1, ...) ---
  // These will become block arguments of the hidden region; NOT op operands.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> ivNames;
  if (parser.parseKeyword("intermediate_variables") ||
      parser.parseOperandList(ivNames, AsmParser::Delimiter::Paren))
    return failure();

  // --- base_ptr = %iab_memref[%var0, %var1, ...] ---
  // The IAB subscripts are SSA names drawn from the unified variable space
  // (captured + intermediate).  We collect their names here and resolve their
  // positions after we know the full unified ordering.
  OpAsmParser::UnresolvedOperand iabMemref;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> iabSubscriptNames;
  if (parser.parseKeyword("base_ptr") || parser.parseEqual() ||
      parser.parseOperand(iabMemref) || parser.parseLSquare() ||
      parser.parseOperandList(iabSubscriptNames) || parser.parseRSquare())
    return failure();

  // --- %base[affine-expr, ...] ---
  OpAsmParser::UnresolvedOperand base;
  if (parser.parseOperand(base)) return failure();

  SmallVector<AffineMapAttr> rawMaps;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> rawMapOperands;
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        SmallVector<OpAsmParser::UnresolvedOperand> dimOps;
        SmallVector<OpAsmParser::UnresolvedOperand> symOps;
        AffineExpr expr;
        if (parser.parseAffineExprOfSSAIds(dimOps, symOps, expr))
          return failure();

        SmallVector<OpAsmParser::UnresolvedOperand> mapOps(dimOps);
        mapOps.append(symOps.begin(), symOps.end());
        rawMaps.push_back(AffineMapAttr::get(
            AffineMap::get(dimOps.size(), symOps.size(), expr, ctx)));
        rawMapOperands.push_back(mapOps);
        return success();
      }))
    return failure();
  result.attributes.clear();  // remove temporaries from parseAffineMapOfSSAIds

  // --- optional attr-dict ---
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // --- : type($base), type($iab) -> type($result) ---
  Type baseType, iabType, resultType;
  if (parser.parseColon() || parser.parseType(baseType) ||
      parser.parseComma() || parser.parseType(iabType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  // --- Determine the unified variable ordering (captured..., intermediate...)
  // --- Captured variables: SSA names referenced in per-dim maps or IAB
  // subscripts that are NOT in the intermediate-variables list.
  llvm::SmallSetVector<StringRef, 8> ivNameSV;
  for (auto& iv : ivNames) ivNameSV.insert(iv.name);

  SmallVector<StringRef> capturedNames;
  llvm::SmallDenseSet<StringRef> capturedSeen;
  auto maybeCapture = [&](StringRef name) {
    if (!ivNameSV.contains(name) && capturedSeen.insert(name).second)
      capturedNames.push_back(name);
  };
  for (auto& ops : rawMapOperands)
    for (auto& op : ops) maybeCapture(op.name);
  for (auto& sub : iabSubscriptNames) maybeCapture(sub.name);

  unsigned unifiedDims = capturedNames.size() + ivNameSV.size();

  // Build a position map from name → unified dim index.
  llvm::SmallDenseMap<StringRef, unsigned> posMap;
  for (unsigned c = 0; c < capturedNames.size(); ++c)
    posMap[capturedNames[c]] = c;
  for (unsigned v = 0; v < ivNameSV.size(); ++v)
    posMap[ivNameSV[v]] = capturedNames.size() + v;

  // --- Canonicalize per-dim subscript maps to the unified dimension space ---
  // Strategy: build a "remap" map from the unified domain to each map's local
  // domain, then compose — matching the approach in
  // canonicalizeAffineMapsToUnifiedOperands used by the ktdp variant.
  SmallVector<Attribute> canonicalMaps;
  for (size_t i = 0; i < rawMaps.size(); ++i) {
    AffineMap raw = rawMaps[i].getValue();
    // Build remap: (unified dims...) -> (local dim for each map operand)
    SmallVector<AffineExpr> remapResults;
    for (auto& opName : rawMapOperands[i]) {
      auto it = posMap.find(opName.name);
      if (it == posMap.end())
        return parser.emitError(parser.getNameLoc())
               << "subscript operand '" << opName.name
               << "' is neither a captured variable nor an intermediate "
                  "variable";
      remapResults.push_back(getAffineDimExpr(it->second, ctx));
    }
    AffineMap remap =
        AffineMap::get(unifiedDims, /*numSymbols=*/0, remapResults, ctx);
    // raw(localDims...) . remap(unifiedDims...) => canonical(unifiedDims...)
    canonicalMaps.push_back(AffineMapAttr::get(raw.compose(remap)));
  }
  auto& props = result.getOrAddProperties<Properties>();
  props.per_dim_subscript_maps = builder.getArrayAttr(canonicalMaps);

  // Build the hidden region with one block arg per intermediate variable.
  // Region is added after parsing is complete, matching the ktdp variant's
  // ordering. ensureTerminator is called before adding block arguments.
  Region* region = result.addRegion();
  Block& body = region->emplaceBlock();
  ensureTerminator(*region, builder, result.location);
  for (size_t i = 0; i < ivNames.size(); ++i)
    body.addArgument(builder.getIndexType(), builder.getUnknownLoc());

  // --- Resolve IAB subscript names to positions in the unified ordering ---
  SmallVector<int32_t> iabPositions;
  for (auto& sub : iabSubscriptNames) {
    auto it = posMap.find(sub.name);
    if (it == posMap.end())
      return parser.emitError(parser.getNameLoc())
             << "IAB subscript '" << sub.name
             << "' is neither a captured variable nor an intermediate variable";
    iabPositions.push_back(static_cast<int32_t>(it->second));
  }
  props.ind_addr_buf_dim_positions = builder.getDenseI32ArrayAttr(iabPositions);

  // --- Resolve operands ---
  if (parser.resolveOperand(base, baseType, result.operands) ||
      parser.resolveOperand(iabMemref, iabType, result.operands))
    return failure();

  // Collect and resolve captured variables in first-occurrence order.
  SmallVector<OpAsmParser::UnresolvedOperand> capturedResolvable;
  {
    llvm::SmallDenseSet<StringRef> resolvedSeen;
    auto collect = [&](llvm::ArrayRef<OpAsmParser::UnresolvedOperand> ops) {
      for (auto& op : ops)
        if (!ivNameSV.count(op.name) && resolvedSeen.insert(op.name).second)
          capturedResolvable.push_back(op);
    };
    for (auto& ops : rawMapOperands) collect(ops);
    collect(iabSubscriptNames);
  }
  if (parser.resolveOperands(capturedResolvable, builder.getIndexType(),
                             result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

void ConstructIndirectAccessTileOp::print(OpAsmPrinter& p) {
  // Build the unified value list once: (captured..., intermediate...)
  SmallVector<Value> allVars(getCapturedVariables().begin(),
                             getCapturedVariables().end());
  for (Value iv : getIntermediateVariables()) allVars.push_back(iv);

  // intermediate_variables(...)  — region block args.
  // Use p << args to match the ktdp variant's idiomatic block-arg printing.
  p << " intermediate_variables(";
  p << getRegion().getArguments();
  p << ")";

  // base_ptr = %iab[%var_at_pos0, ...]
  p << " base_ptr = " << getIndAddrBufMemref() << "[";
  llvm::interleaveComma(getIndAddrBufDimPositions(), p, [&](int32_t pos) {
    p << allVars[static_cast<unsigned>(pos)];
  });
  p << "]";

  // %base[affine-expr, ...]
  p << " " << getBase() << "[";
  auto maps = getPerDimSubscriptMaps();
  for (unsigned i = 0, e = maps.size(); i < e; ++i) {
    if (i > 0) p << ", ";
    p.printAffineMapOfSSAIds(llvm::cast<AffineMapAttr>(maps[i]), allVars);
  }
  p << "]";

  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getIndAddrBufDimPositionsAttrName(),
                                           getPerDimSubscriptMapsAttrName()});

  p << " : " << getBase().getType() << ", " << getIndAddrBufMemref().getType()
    << " -> ";
  p.printType(getResult().getType());
}

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructIndirectAccessTileOp::verify() {
  // The region must be empty (only the implicit terminator is allowed).
  Block& body = getRegion().front();
  if (!body.without_terminator().empty())
    return emitOpError("region must be empty (only the terminator is allowed)");

  auto iabType = mlir::cast<MemRefType>(getIndAddrBufMemref().getType());
  unsigned unifiedDims =
      getCapturedVariables().size() + getIntermediateVariables().size();

  // IAB dim-positions count must match the rank of the IAB memref.
  auto iabPositions = getIndAddrBufDimPositions();
  if (static_cast<int64_t>(iabPositions.size()) != iabType.getRank())
    return emitOpError() << "ind_addr_buf_dim_positions has "
                         << iabPositions.size()
                         << " entry/entries but ind_addr_buf_memref has rank "
                         << iabType.getRank() << "; they must be equal";

  // Each IAB position must be within [0, unifiedDims).
  for (auto [idx, pos] : llvm::enumerate(iabPositions)) {
    if (pos < 0 || static_cast<unsigned>(pos) >= unifiedDims)
      return emitOpError() << "ind_addr_buf_dim_positions[" << idx
                           << "] = " << pos << " is out of range [0, "
                           << unifiedDims << ")";
  }

  // per_dim_subscript_maps count must match the rank of $base.
  auto baseType = mlir::cast<MemRefType>(getBase().getType());
  auto numMaps = static_cast<int64_t>(getPerDimSubscriptMaps().size());
  if (numMaps != baseType.getRank())
    return emitOpError() << "per_dim_subscript_maps has " << numMaps
                         << " map(s) but base memref has rank "
                         << baseType.getRank() << "; they must be equal";

  // Each map must have exactly `unifiedDims` input dimensions.
  for (auto [idx, attr] : llvm::enumerate(getPerDimSubscriptMaps())) {
    auto mapAttr = mlir::dyn_cast<AffineMapAttr>(attr);
    if (!mapAttr)
      return emitOpError() << "per_dim_subscript_maps[" << idx
                           << "] is not an AffineMapAttr";
    if (mapAttr.getValue().getNumDims() != unifiedDims)
      return emitOpError()
             << "per_dim_subscript_maps[" << idx << "] has "
             << mapAttr.getValue().getNumDims()
             << " dimension(s) but the unified variable space has "
             << unifiedDims << " (captured=" << getCapturedVariables().size()
             << ", intermediate=" << getIntermediateVariables().size() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructMemoryViewOp::verify() {
  unsigned nDims = getStaticSizes().size();
  if (getStaticStrides().size() != nDims)
    return emitOpError(
        "static_sizes and static_strides must have equal length");
  unsigned dynSizes = llvm::count(getStaticSizes(), mlir::ShapedType::kDynamic);
  unsigned dynStrides =
      llvm::count(getStaticStrides(), mlir::ShapedType::kDynamic);
  if (getSizes().size() != dynSizes)
    return emitOpError(
        "number of dynamic size operands does not match "
        "kDynamic entries in static_sizes");
  if (getStrides().size() != dynStrides)
    return emitOpError(
        "number of dynamic stride operands does not match "
        "kDynamic entries in static_strides");
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(getResult().getType());
  if (!memrefType) return emitOpError("result must be a memref type");
  if (memrefType.getRank() != static_cast<int64_t>(nDims))
    return emitOpError(
        "result memref rank does not match sizes/strides length");

  // Check 1: memory_space attribute must match the memref's memory space.
  if (getMemorySpace() != memrefType.getMemorySpace())
    return emitOpError(
        "memory_space attribute does not match result memref memory space");

  // Check 2: strides must be consistent with the result memref's layout.
  // If the memref has a StridedLayoutAttr, compare directly. If it has no
  // layout (identity / row-major), verify that the provided static strides
  // are strictly decreasing left-to-right and that each stride equals the
  // product of all static sizes to its right (i.e. they encode a row-major
  // layout).  Dynamic strides/sizes are skipped in both paths.
  llvm::ArrayRef<int64_t> opStrides = getStaticStrides();
  if (auto stridedLayout = mlir::dyn_cast_or_null<mlir::StridedLayoutAttr>(
          memrefType.getLayout())) {
    llvm::ArrayRef<int64_t> layoutStrides = stridedLayout.getStrides();
    for (unsigned i = 0; i < nDims; ++i) {
      int64_t opS = opStrides[i];
      int64_t lyS = layoutStrides[i];
      if (opS == mlir::ShapedType::kDynamic ||
          lyS == mlir::ShapedType::kDynamic)
        continue;
      if (opS != lyS)
        return emitOpError()
               << "stride " << i << " (" << opS
               << ") does not match the result memref layout stride (" << lyS
               << ")";
    }
  } else if (!memrefType.getLayout() ||
             mlir::isa<mlir::AffineMapAttr>(memrefType.getLayout())) {
    // Identity (no layout) or affine-map layout — check that the static
    // strides encode a row-major (decreasing) layout consistent with the
    // static sizes.
    llvm::ArrayRef<int64_t> sizes = getStaticSizes();
    // Walk from the rightmost dimension outward.  expectedStride accumulates
    // the product of static sizes seen so far (starting from 1 for dim N-1).
    int64_t expectedStride = 1;
    for (int i = static_cast<int>(nDims) - 1; i >= 0; --i) {
      int64_t s = opStrides[i];
      if (s != mlir::ShapedType::kDynamic) {
        if (s != expectedStride)
          return emitOpError()
                 << "stride " << i << " (" << s
                 << ") is inconsistent with a row-major layout for the given "
                    "static sizes (expected "
                 << expectedStride << ")";
      }
      // Advance the expected stride by this dimension's size (skip if dynamic).
      if (i > 0) {
        int64_t sz = sizes[i];
        if (sz == mlir::ShapedType::kDynamic)
          expectedStride = mlir::ShapedType::kDynamic;  // can't predict further
        else if (expectedStride != mlir::ShapedType::kDynamic)
          expectedStride *= sz;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp — ViewLikeOpInterface
//===----------------------------------------------------------------------===//

mlir::Value ConstructMemoryViewOp::getViewSource() { return getOffset(); }
