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
include "ktir/Dialect/KTDP/KTDPTypes.td"

class KTDPLowering_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDPLowering_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp
//===----------------------------------------------------------------------===//

def KTDPLowering_ConstructIndirectAccessTileOp :
    KTDPLowering_Op<"construct_indirect_access_tile",
                    [AttrSizedOperandSegments]> {

  let summary = "indirect access tile with indirection factored into an IAB "
                "base address and direct element offsets";
  let description = [{
    The `ktdp_lowering.construct_indirect_access_tile` operation is the
    lowered form of `ktdp.construct_indirect_access_tile` in which the
    indirect address has been factored: the base address contribution is
    supplied by an entry in the indirect address buffer (IAB), selected by
    `$ind_addr_buf_subscripts` into `$ind_addr_buf_memref`, and the remaining
    dimensions are expressed as direct element offsets into `$base` via
    `$direct_subscripts`.  In all other respects the semantics of this op are
    inherited from `ktdp.construct_indirect_access_tile`.

    **Semantics of `base_ptr = %iab_mv[%s0, %s1, ...]`:**  the `index` value
    stored at those subscripts in the IAB memref is added as an element-offset
    delta to the flat address computed from `$base`'s `construct_memory_view`
    base together with the explicit `$direct_subscripts`.  Concretely, if the
    `ind()` dimension of `$base` had stride `S`, the IAB entry stores
    `original_index * S`, and the explicit subscript for that dimension is
    `%c0` (contributing `0 * S = 0`).  The two terms together reproduce the
    original indirect address without double-counting.

    Any of the `$ind_addr_buf_subscripts` may be values drawn from the
    `$intermediate_variables` list, which are the same block arguments
    that parameterise `$direct_subscripts` and `$variables_space_set`.

    The `$intermediate_variables` and `$variables_space_set` define the tile
    shape.  `ConstructThreeStagePipeline` reads `$variables_space_set` to
    determine the element count of the `ktdf.ind_data_transfer` it emits.

    The result type `!ktdp.access_tile<...>` is identical to the type
    produced by `ktdp.construct_indirect_access_tile`, so all existing
    consumers (`ktdp.load`, `ktdp.store`, `ConstructThreeStagePipeline`)
    work unchanged.

    Example (after ind_addr_buf capacity legalization, 1-D IAB;
    `%arg6` is an intermediate variable used as the IAB subscript):
    ```mlir
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg6]
        %X[%c0, %arg7, %arg8]
        {variables_space_set = #set, variables_space_order = #map}
        : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<32x2x64xindex>
    ```

    Example (before capacity legalization, 2-D IAB;
    `%arg5` and `%arg6` are intermediate variables):
    ```mlir
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %X[%c0, %arg7, %arg8]
        {variables_space_set = #set2, variables_space_order = #map}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32x2x64xindex>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$base,
    // IAB memory view (the memory space is architecture-defined;
    // the verifier only checks that ind_addr_buf_subscripts count
    // equals the rank of this memref).
    Arg<AnyMemRef, "indirect address buffer", [MemRead]>:$ind_addr_buf_memref,
    // Per-dimension subscripts into ind_addr_buf_memref; count must equal
    // the rank of ind_addr_buf_memref (enforced by the verifier).
    // Any of these may be values drawn from $intermediate_variables.
    Variadic<Index>:$ind_addr_buf_subscripts,
    // Explicit direct subscripts into $base (one per base dimension).
    Variadic<Index>:$direct_subscripts,
    // Intermediate variables that parameterise the tile shape via
    // variables_space_set; may also appear in ind_addr_buf_subscripts
    // and direct_subscripts.  These operands play the same role as
    // the hidden-region block arguments used by the ktdp variant: they 
    // represent the loop-like iteration variables that enumerate points 
    // in the variable space defined by variables_space_set.
    Variadic<Index>:$intermediate_variables,
    AffineMapAttr:$variables_space_order,
    Builtin_IntegerSetAttr:$variables_space_set
  );

  let results = (outs Ktdp_AccessTileType:$result);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `intermediate_variables` `(` $intermediate_variables `)`
    `base_ptr` `=` $ind_addr_buf_memref `[` $ind_addr_buf_subscripts `]`
    $base `[` $direct_subscripts `]`
    attr-dict `:` type($base) `,` type($ind_addr_buf_memref) `->` qualified(type($result))
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
