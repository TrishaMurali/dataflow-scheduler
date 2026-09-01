//===-- Passes.td ------------------------------------------*- tablegen -*-===//
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
// This file declares all the ktdf dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def BroadcastPromotionPass : Pass<"broadcast-promotion", "mlir::ModuleOp"> {
  let summary = "Hoist invariant ktdf.data_transfer ops as sibling pipelines";
  let description = [{
    This pass identifies ktdf.data_transfer ops inside ktdf.stage bodies
    whose source and destination indices/sizes are invariant w.r.t. at
    least one enclosing scf.for induction variable, and hoists them
    outward as a sibling ktdf.pipeline placed before the invariant loop.

    Only broadcast-style hoists (single-transfer, hoist-up) are handled.
    Reduction hoists (paired pre-load/post-store) and FIFO-destination
    hoists (requiring co-hoisting of ktdf.fifo.allocate and consumer-side
    register-promotion rewrites) are handled by separate future passes.

    The pass runs a fixed-point loop: one hoist per iteration, re-scan
    after each hoist, exit when no candidate remains.
  }];
  let constructor = "::mlir::ktdf::createBroadcastPromotionPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect"];
}

def StageCoarseningPass : Pass<"stage-coarsening", "mlir::ModuleOp"> {
  let summary = "Perform stage coarsening transformation";
  let description = [{
    This pass implements a 4-stage stage coarsening algorithm to create
    hierarchical pipeline structures by sinking loops into stages.

    Following identification of stage groups (using loop distribution legality rules):
    - Stage 1a: Distribute and sink loops into pipeline
    - Stage 1b: Further sink into stages with single parent loop (expand buffers)
    - Stage 2: Discover and correct illegal structures iteratively
    - Stage 3: Correct stage dependencies that cross pipeline boundaries

    The transformation creates nested pipelines where:
    - Outer pipelines operate at tile granularity
    - Inner pipelines operate at hardware execution granularity

    This enables better hardware mapping and reduced synchronization overhead.
  }];
  let constructor = "::mlir::ktdf::createStageCoarseningPass()";
  let options = [
    Option<"loopNestDepth", "loop-nest-depth", "unsigned",
           /*default=*/"2",
           "Number of perfectly nested loops to consider for sinking">
  ];
  let dependentDialects = ["mlir::ktdf::KTDFDialect"];
}

def SplitReductionInnerOuterDimPass : Pass<"split-reduction-inner-outer-dim", "mlir::ModuleOp"> {
  let summary = "Split the inner and outer dimensions of a reduction";
  let description = [{
    Walks the module for linalg.generic ops with at least one reduction
    iterator and splits each reduction dimension into an inner dimension of
    stick size (derived from the hardware SIMD feature) and an outer
    dimension of size reduction_dim / stick size.
  }];
  let constructor = "::mlir::ktdf::createSplitReductionInnerOuterDimPass()";
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def ReductionLoopExposurePass : Pass<"reduction-loop-exposure", "mlir::ModuleOp"> {
  let summary = "Expose the reduction loop of a linalg.generic as an explicit scf.for with a loop-carry accumulator";
  let description = [{
    Walks the module for any ktdf.stage that contains a linalg.generic with a
    reduction iterator (the Compute stage).  For each one found it rewrites the
    parent ktdf.pipeline so that the outer reduction becomes one nested scf.for loop per reduction dimension, with the accumulator tensor
    threaded as iter_args.

    The innermost reduction dimension — identified as the reduction loop dim
    that maps to the rightmost input tensor dimension — is excluded from loop
    exposure and left as-is for later handling by a device PDL pattern.

    The pass may be composed with ReductionDimChunking.  When the input has
    already been chunked, the Load stage body contains a scf.if that feeds
    the previous chunk's partial result into fifo_partial on non-first
    iterations; the pass preserves that scf.if outside the new reduction loop.

    Before (R_i = size of outer reduction dimension i; S = innermost reduction
    size; C = non-reduction elements):

      ktdf.private: [fifo_in(R_0*...*R_n*C*S elements), fifo_out(C*S elements), ...]

      Load stage:    data_transfer  src -> fifo_in   (full R_0*...*R_n*C*S slice)
      Compute stage: read_from_fifo fifo_in -> tensor<..xR_0x..xR_nxCxSxf16>
                     linalg.generic (reduction over R_i and S) -> tensor<..xCxf16>
                     write_to_fifo fifo_out
      Store stage:   data_transfer  fifo_out -> dst

    After (only the outer reduction loop R is exposed; S stays as a reduction):

      ktdf.private: fifo_in element count shrunk by R = product(R_i).
                    (fifo_in: R*C*S -> C*S elements; other FIFOs unchanged.)

      Load stage:    scf.if %is_first \{} else \{ partial transfer }  (preserved outside loop)
                     scf.for %r0 = 0 to R_0 {
                       scf.for %r1 = 0 to R_1 { ...
                         data_transfer src[..,%r0,.,%r1,..] -> fifo_in  (size C*S)
                       }
                     }
      Compute stage: %seed = scf.if %is_first -> tensor<..xCxf16> {
                       tensor.empty
                     } else {
                       read_from_fifo fifo_partial
                     }
                     scf.for %r0 = 0 to R_0 iter_args(%a = %seed)
                         {loop_type = reduction_loop}
                       scf.for %r1 = 0 to R_1 iter_args(%a = %a) { ...
                         %slice = read_from_fifo fifo_in -> tensor<..x1x..x1xCxSxf16>
                         %a     = linalg.generic (reduction over S) ins(%slice) outs(%a)
                         if (all ivs == last): write_to_fifo %a, fifo_out
                         scf.yield %a
                       }
                     scf.yield %r0_result
      Store stage:   scf.for %r0 = 0 to R_0 {
                       scf.for %r1 = 0 to R_1 { ...
                         if (all ivs == last): data_transfer fifo_out -> dst
                       }
                     }

    Any data_transfer in any stage whose source or destination is the
    shrunken fifo_in has its static source sizes divided by R (where
    R = product(R_i)) automatically.
    The source AffineMap is extended with fresh dimensions for the reduction
    loop IVs so the correct per-iteration slice is addressed.

    Stage identification is structural, not string-based:
    - Load stage:    the sibling whose depends_out token matches the Compute
                     stage's depends_in token.
    - Compute stage: contains a linalg.generic with a reduction iterator.
    - Store stage:   the sibling whose depends_in token matches the Compute
                     stage's depends_out token (SSA identity check).
    - All other stages: wrapped verbatim in scf.for loops, nothing else
                        modified.

    Preconditions (checked; pass skips cleanly if not met):
    - A ktdf.stage somewhere in the module contains a linalg.generic with
      at least one statically-sized outer reduction iterator.
    - That stage is a direct child of a ktdf.pipeline with a ktdf.private.
    - The input FIFO (fifo_in) element count is divisible by R (where
      R = product(R_i)).
    - The Compute stage has exactly one write_to_fifo.
  }];
  let constructor = "::mlir::ktdf::createReductionLoopExposurePass()";
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}




def SubsumeLinearizeIndexPass : Pass<"subsume-linearize-index", "mlir::ModuleOp"> {
  let summary = "Fold ktdf.tiling.linearize_index into ktdf.data_transfer affine maps";
  let description = [{
    For each ktdf.data_transfer, any index operand that is produced by a
    ktdf.tiling.linearize_index with all-constant strides is replaced by an
    affine.apply encoding the weighted-sum expression. The affine.apply is
    then composed into the transfer's source_map/dest_map via
    fullyComposeAffineMapAndOperands. Dead tiling.linearize_index ops are erased.
  }];
  let constructor = "::mlir::ktdf::createSubsumeLinearizeIndexPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect", 
      "mlir::affine::AffineDialect",
      "mlir::arith::ArithDialect"];
}

def TileSizeSelectionPass : Pass<"tile-size-selection", "mlir::ModuleOp"> {
  let summary = "Resolve ktdf.tiling.reserve_size placeholders to concrete constants";
  let description = [{
    This pass analyzes each ktdf.tiling.reserve_size op to discover which scf.for
    loops it is used to tile and uses the loop trip count and other information 
    (eg. in future L1 usage) to replace the placeholder with an actual tile size.

    Unresolved placeholders are tracked and reported with clear debug messages so
    future post-processing can apply a fallback policy if desired.
  }];
  let constructor = "::mlir::ktdf::createTileSizeSelectionPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect", 
      "mlir::scf::SCFDialect",
      "mlir::arith::ArithDialect"];
}

def MapReductionPartialsPass : Pass<"map-reduction-partials", "mlir::ModuleOp"> {
  let summary = "Lower reduction linalg.generic to buffer-semantics form with a pre-allocated accumulator";
  let description = [{
    Walks the module for linalg.generic ops that have at least one reduction
    iterator and whose first input is produced by a ktdf.read_from_fifo.

    The expected input shape (produced by ReductionLoopExposure) is:
    (R is the size of the reduction dimension)

      %empty = tensor.empty() : tensor<...xf16>
      %result = scf.for %r = 0 to R iter_args(%carry = %empty)
                    {loop_type = reduction_loop}
        %slice = ktdf.read_from_fifo %fifo -> tensor<...xf16>
        %new_carry = linalg.generic ins(%slice) outs(%carry) -> tensor<...xf16>
        scf.if (%r == R-1): ktdf.write_to_fifo %new_carry, %fifo_out
        scf.yield %new_carry

    For each matching generic the pass:
      1. Allocates a memref of the output shape (compute memory space) at the
         top of the enclosing ktdf.stage body.
      2. Zero-fills it with arith.constant 0.0 + linalg.fill.
      3. Emits a new ktdf.read_from_fifo for the same fifo slot with a memref
         result type (matching the compute memory space).
      4. Replaces the generic with a buffer-semantics clone: same indexing maps
         and iterator types, memref ins from the new read, memref outs from the
         alloc, no result.
      5. Patches any ktdf.write_to_fifo that consumed the loop result to use
         the alloc instead, then erases the original generic and its tensor
         read_from_fifo input.
      6. Strips the accumulator iter_arg and its scf.yield operand from the
         enclosing scf.for (rebuilding the loop without that slot), and erases
         the now-dead tensor.empty initializer.
  }];
  let constructor = "::mlir::ktdf::createMapReductionPartialsPass()";
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ReductionDimChunkingPass : Pass<"reduction-dim-chunking", "mlir::ModuleOp"> {
  let summary = "Split the reduction dimension of inner linalg.generic ops into sequential chunks";
  let description = [{
    This pass targets the inner Load→Compute→Store pipeline that contains a
    linalg.generic with a reduction iterator.  It replaces that single
    pipeline with nested scf.for loops — one per reduction dimension that
    has more than one chunk — each innermost iteration containing one
    ktdf.pipeline (Load / Compute / Store stages).
    First-vs-rest accumulation behaviour is selected at runtime via
    `%condition = AND(iv_j == 0 for all active chunk loop IVs)`:

      Load stage   : transfers the current input chunk slice to fifo_in.
                     When !condition, also transfers the partial accumulator
                     (the existing local memory output buffer) to fifo_partial so the
                     Compute stage can read it back.
      Compute stage: when condition, initialises the output tensor with
                     tensor.empty; otherwise reads the partial result from
                     fifo_partial.  Runs the linalg.generic and writes its
                     result to fifo_out unconditionally.
      Store stage  : unconditionally reads fifo_out (the result written by
                     the Compute stage) and writes it back to the local memory output
                     buffer.

    The existing local memory output buffer (discovered via the Store stage's
    data_transfer destination) is reused as the partial-accumulation
    buffer — no new memref is allocated.

    Preconditions (checked, pass aborts cleanly if not met):
    - The module contains a ktdf.pipeline with three sibling stages matching
      the StageCoarsening output shape: a Load stage, a Compute stage, and
      a Store stage.
    - The Compute stage contains a scf.for (batch loop) whose body is a
      single ktdf.pipeline with three stages: Load, Compute, Store.
    - The inner Compute stage body contains exactly one linalg.generic with
      at least one `reduction` iterator type.

    Option `num-chunks` sets the static chunk count per reduction dim (one
    value per dim, in iterator-type order).  Values may differ across dims;
    the pass emits one nested scf.for per dim (dims with chunk count 1
    produce no loop).  When omitted (the default), the pass infers the
    chunk count automatically from the input tensor size and the
    `chunk-size-threshold` option: it picks the smallest integer N ≥ 1 such
    that (total_input_bytes / N) ≤ chunk-size-threshold AND N evenly divides
    every reduction-dimension size.

    Option `chunk-size-threshold` (default 1 MiB = 1048576 bytes) is the
    per-chunk byte budget used when `num-chunks` is not provided.  It is
    ignored when `num-chunks` is explicitly set.
  }];
  let constructor = "::mlir::ktdf::createReductionDimChunkingPass()";
  let options = [
    ListOption<"numChunks", "num-chunks", "unsigned",
               "Number of sequential chunks per reduction dimension (one value "
               "per reduction dim, in iterator-type order).  Values may differ "
               "across dims; the pass emits one nested scf.for per dim (dims "
               "with chunk count 1 produce no loop).  When empty (the default) "
               "the pass infers the chunk count from chunk-size-threshold.">,
    Option<"chunkSizeThreshold", "chunk-size-threshold", "int64_t",
           "1048576",
           "Per-chunk size budget in bytes used for automatic numChunks "
           "inference when num-chunks is not specified (default: 1 MiB).">
  ];
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD
