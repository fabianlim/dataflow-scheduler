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

def ReductionLoopExposurePass : Pass<"reduction-loop-exposure", "mlir::ModuleOp"> {
  let summary = "Expose the reduction loop of a linalg.generic as an explicit scf.for with a loop-carry accumulator";
  let description = [{
    Walks the module for any ktdf.stage that contains a linalg.generic with a
    reduction iterator (the compute stage).  For each one found it rewrites the
    parent ktdf.pipeline so that the reduction becomes an explicit scf.for loop:

    Before (StageCoarsening output):

      ktdf.private: [fifo_in(R*C elements), fifo_out(C elements), ...]

      load stage:    data_transfer  LX -> fifo_in   (full R*C slice)
      compute stage: read_from_fifo -> tensor<1xRxCxf16>
                     linalg.generic (reduction) -> tensor<1xCxf16>
                     write_to_fifo
      store stage:   data_transfer  fifo_out -> LX

    After:

      ktdf.private: all FifoSlotType results whose element count is a
                    multiple of R are shrunk by dividing by R.
                    (e.g. fifo_in: R*C -> C elements)

      load stage:    scf.for %r = 0 to R           -- all original ops inside
                       data_transfer LX -> fifo_in  (static dest size / R)
      compute stage: %acc = memref.alloc : memref<1xCxf16, "SFU_REG">
                     linalg.fill ins(0) outs(%acc)
                     scf.for %r = 0 to R           {loop_type = reduction_loop}
                       slice = read_from_fifo -> memref<1x1xCxf16>
                       linalg.generic ins(slice) outs(%acc)   -- in place
                       write_to_fifo %acc
      store stage:   scf.for %r = 0 to R           -- all original ops inside
                       data_transfer fifo_out -> LX

    The accumulator is allocated here, before address assignment, so that it
    is given a physical offset in the compute unit's register file.  Because it
    is a buffer, the linalg.generic has buffer semantics and no result, so the
    loop needs no iter_args.

    Any data_transfer in any stage whose source or destination is a shrunken
    FIFO has its static sizes divided by R automatically.

    Stage identification is structural, not string-based:
    - Load stage:    the sibling whose depends_out token matches the compute
                     stage's depends_in token; must have exactly one
                     FIFO-dest data_transfer (fails with error if > 1).
    - Compute stage: contains a linalg.generic with a reduction iterator.
    - Store stage:   the sibling whose depends_in token matches the compute
                     stage's depends_out token (SSA identity check).
    - All other stages: wrapped verbatim in scf.for, nothing else modified.

    Preconditions (checked; pass skips cleanly if not met):
    - A ktdf.stage somewhere in the module contains a linalg.generic with
      at least one statically-sized reduction iterator.
    - That stage is a direct child of a ktdf.pipeline with a ktdf.private.
    - At least one FifoSlotType result in ktdf.private has element count
      divisible by R (used to identify and shrink participating FIFOs).
    - The load stage has exactly one FIFO-dest data_transfer; the compute
      stage has exactly one write_to_fifo.
  }];
  let constructor = "::mlir::ktdf::createReductionLoopExposurePass()";
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
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

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD
