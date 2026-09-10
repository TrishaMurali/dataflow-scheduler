//===-- KTDPLowering.td ------------------------------------*- tablegen -*-===//
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
// This file defines ktdp_lowering operations produced during the scheduling
// transformation pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERING_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERING_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLoweringDialect.td"

// Import the KTDP types so we can use Ktdp_AccessTileType as the result type.
// ktdp::RegionTerminatorOp is referenced by its C++ class name only (as a
// string argument to SingleBlockImplicitTerminator), so we do not need to
// include the full KTDP ops tablegen here.
include "ktir/Dialect/KTDP/KTDPTypes.td"

class KTDPLowering_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDPLowering_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp
//===----------------------------------------------------------------------===//

def KTDPLowering_ConstructIndirectAccessTileOp :
    KTDPLowering_Op<"construct_indirect_access_tile",
                    [SingleBlockImplicitTerminator<"ktdp::RegionTerminatorOp">]> {

  let summary = "indirect access tile with indirection factored into an IAB "
                "base address and direct element offsets";
  let description = [{
    The `ktdp_lowering.construct_indirect_access_tile` operation is the
    lowered form of `ktdp.construct_indirect_access_tile` in which the
    indirect address has been factored: the base address contribution is
    supplied by a pre-computed flat address stored in an entry of the indirect
    address buffer (IAB), selected by `$ind_addr_buf_subscripts` into
    `$ind_addr_buf_memref`.  The remaining dimensions of `$base` are described
    by `$per_dim_subscript_maps` — one affine map per dimension — in all other
    respects the semantics of this op follow those of
    `ktdp.construct_indirect_access_tile`.

    **Intermediate variables** are the loop-iteration values that enumerate
    points in the variable space defined by `$variables_space_set`.  They are
    modelled as block arguments of the op's hidden single-block region — exactly
    as in the `ktdp` variant — and are *not* passed as operands from the
    enclosing scope.  Access them via `getIntermediateVariables()`.

    **IAB subscripts vs. direct subscript maps:**

    `$ind_addr_buf_subscripts` are plain SSA `index` values — each one is
    either a block argument of `$region` (an intermediate variable) or a
    member of `$captured_variables` (an outer loop IV or constant).  No affine
    map is needed here because the subscript is used directly as an index into
    `$ind_addr_buf_memref`; the address buffer lookup itself supplies the
    selected address, so the subscript only needs to identify which entry to
    read.  Note that even at the KTDP level, an affine map is not applied to
    the value loaded from an indirect subscript either.

    `$per_dim_subscript_maps` are affine maps over the unified dimension space
    `(captured_variables..., intermediate_variables...)`.  They *can* contain
    arithmetic — e.g. `(%c0 + %arg7)` lowers to `affine_map<(d0,d1)->d0+d1>`
    applied to `(%c0, %arg7)` — because the hardware memory unit still applies
    strides to these subscripts at transfer time.  The canonical dimension
    ordering mirrors `ktdp.construct_indirect_access_tile`:
    captured variables occupy the leading dimensions, intermediate variables
    follow.

    **Semantics of `base_ptr = %iab_mv[%s0, %s1, ...]`:**  the flat address
    value stored at those subscripts in the IAB memref is added as an
    element-offset delta to the address computed from `$base`'s
    `construct_memory_view` base.  The IAB value already represents the
    complete contribution of the indirect dimensions; no stride or affine
    subscript map is applied to an indirect subscript or to the value loaded
    from it.  `$per_dim_subscript_maps` contribute only the coordinates for
    direct dimensions, whose strides are applied by the hardware memory unit:
    `flat = cmv_base + iab[s0,s1,...] + direct_map0(...)*S0 + ...`

    The `$variables_space_set` defines the tile shape.
    `ConstructThreeStagePipeline` reads it to determine the element count of
    the `ktdf.ind_data_transfer` it emits.

    The result type `!ktdp.access_tile<...>` is identical to the type
    produced by `ktdp.construct_indirect_access_tile`, so all existing
    consumers (`ktdp.load`, `ktdp.store`, `ConstructThreeStagePipeline`)
    work unchanged.

    Example (after per-entry legalization, 1-D IAB; `%i2` is a captured
    loop IV selecting the IAB entry; `%arg7` and `%arg8` are intermediate
    variables; `%c0` is a captured constant):
    ```mlir
    // per_dim_subscript_maps:
    //   dim 0 (indirect dim, zero'd out): affine_map<(d0,d1,d2) -> (0)>
    //   dim 1 (direct, offset by captured c0): affine_map<(d0,d1,d2) -> (d0 + d1)>
    //   dim 2 (direct, identity): affine_map<(d0,d1,d2) -> (d2)>
    // captured_variables: [%c0, %arg7]  intermediate_variables: [%arg8]
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg7, %arg8)
        base_ptr = %iab_mv[%i2]
        %desc_1[(%c0), (%c0 + %arg7), (%arg8)]
        {variables_space_set = #set, variables_space_order = #map}
        : memref<64x2x64xf16>, memref<32xindex, "IAB">
        -> !ktdp.access_tile<2x64xindex>
    ```

    Example (before capacity legalization, 2-D IAB; all four intermediate
    variables are block args):
    ```mlir
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_1[(%c0), (%c0 + %arg7), (%arg8)]
        {variables_space_set = #set2, variables_space_order = #map}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
        -> !ktdp.access_tile<2x32x2x64xindex>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$base,
    // IAB memory view (the memory space is architecture-defined;
    // the verifier checks that ind_addr_buf_dim_positions.size() == rank of this
    // memref).
    Arg<AnyMemRef, "indirect address buffer", [MemRead]>:$ind_addr_buf_memref,
    // One integer per IAB dimension giving its position in the unified
    // (captured_variables..., intermediate_variables...) ordering.  No operand
    // is needed because each IAB subscript is always a direct use of exactly
    // one variable from that space — never an arithmetic expression — so a
    // simple index into the unified list is sufficient.
    DenseI32ArrayAttr:$ind_addr_buf_dim_positions,
    // One AffineMapAttr per dimension of $base.  Each map is expressed over
    // the unified dimension space (captured_variables..., intermediate_variables...).
    // Maps CAN contain arithmetic (e.g. d0 + d1, d2 mod 64) because the
    // hardware memory unit still applies strides to these values at transfer time.
    ArrayAttr:$per_dim_subscript_maps,
    // SSA values from the enclosing scope referenced by $per_dim_subscript_maps
    // that are NOT intermediate variables.
    // These occupy the leading dimensions of every subscript map.
    Variadic<Index>:$captured_variables,
    AffineMapAttr:$variables_space_order,
    Builtin_IntegerSetAttr:$variables_space_set
  );

  // Hidden single-block region whose block arguments are the intermediate
  // variables (the loop-iteration indices that enumerate the variable space).
  // The block is terminated by ktdp.region_terminator.
  let regions = (region SizedRegion<1>:$region);

  let results = (outs Ktdp_AccessTileType:$result);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::ktdp::AccessTileType":$resultType,
      "::mlir::Value":$base,
      "::mlir::Value":$indAddrBufMemref,
      "::mlir::DenseI32ArrayAttr":$indAddrBufDimPositions,
      "::mlir::ArrayAttr":$perDimSubscriptMaps,
      "::mlir::ValueRange":$capturedVariables,
      "unsigned":$numIntermediateVariables,
      "::mlir::AffineMap":$variablesSpaceOrder,
      "::mlir::IntegerSet":$variablesSpaceSet)>
  ];

  let extraClassDeclaration = [{
    /// Returns the intermediate variables (block arguments of the hidden
    /// region), which enumerate the variable space defined by
    /// $variables_space_set.  Mirrors
    /// ConstructIndirectAccessTilesOp::getIntermediateVariables() in the
    /// ktdp dialect.
    ::mlir::ValueRange getIntermediateVariables() {
      return getRegion().getArguments();
    }
  }];
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp
//===----------------------------------------------------------------------===//

def KTDPLowering_ConstructMemoryViewOp :
    KTDPLowering_Op<"construct_memory_view", [
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<ViewLikeOpInterface>,
      MemRefsNormalizable,
      Pure
    ]> {

  let summary = "Lowering-phase variant of ktdp.construct_memory_view that "
                "accepts any memory-space attribute";
  let description = [{
    The `ktdp_lowering.construct_memory_view` operation is the lowering-phase
    counterpart of `ktdp.construct_memory_view`.  It has identical semantics and
    assembly syntax but relaxes the `memory_space` attribute constraint: whereas
    `ktdp.construct_memory_view` requires a `#ktdp.memory_space<...>` typed
    attribute, this variant accepts any attribute (e.g. a plain `StringAttr`
    such as `"IAB"`, or other architecture-defined memory spaces that
    are not expressible as a data-parallel `MemorySpaceAttr`).

    This op is introduced by the `IndirectComputeGroupSplit` pass wherever an
    ind_addr_buf memory view is constructed and is carried forward unchanged
    through subsequent lowering steps.  All other fields — `offset`, `sizes`,
    `strides`, `static_sizes`, `static_strides`, `coordinate_set`, and the
    result memref type — are identical to those of `ktdp.construct_memory_view`.

    Example:
    ```mlir
    #set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set, memory_space = "IAB"}
        : memref<32xindex, "IAB">
    ```
  }];

  let arguments = (ins
    Index:$offset,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    AnyAttr:$memory_space,
    Builtin_IntegerSetAttr:$coordinate_set
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $offset `` `,`  `sizes` `` `:`
    custom<DynamicIndexList>($sizes, $static_sizes)
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERING_TD
