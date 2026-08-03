//===-- ReductionLoopExposure.cpp -------------------------------*- c++ -*-===//
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
// ReductionLoopExposure: expose the reduction dimension of a linalg.generic
// inside a ktdf.pipeline as an explicit scf.for loop accumulating into a
// compute-unit-local accumulator buffer.
//
// Algorithm:
//   1. Walk the module for any ktdf.stage containing a linalg.generic with
//      a reduction iterator — this is the compute stage.
//   2. Derive R (reduction size) from the input tensor's reduction dimension.
//   3. Get the parent ktdf.pipeline of the compute stage.
//   4. Find the load stage (upstream of compute via depends_in/depends_out
//      token chain).  Assert the load stage has exactly one FIFO-dest
//      data_transfer — its destination is fifo_in.
//   5. Shrink ALL FifoSlotType results in ktdf.private whose element count
//      is divisible by R (divide by R).  Patch every fifo.allocate inside
//      the private body to match.
//   6. Patch every data_transfer inside the pipeline whose source or
//      destination is a shrunken FIFO: divide its static sizes by R.
//   7. For every stage in the pipeline, wrap its entire body in
//      scf.for %r = 0 to R step 1.
//   8. For the compute stage specifically:
//      - A memref.alloc in the compute unit's register-file memory space plus a
//        zeroing linalg.fill are emitted before the loop (accumulator init).
//        Allocating here, ahead of address assignment, is what gets the buffer
//        a physical offset.
//      - The loop carries no iter_args: the accumulator is a buffer, so the
//        linalg.generic writes it in place and yields no SSA value.
//      - write_to_fifo is emitted unconditionally every iteration.
//      - The loop is tagged {loop_type = reduction_loop}.
//   9. Find the store stage (downstream of compute via depends_out/depends_in
//      token chain).  Its data_transfer runs unconditionally every iteration
//      (LXSU overwrites the same address; last write wins = final result).
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "reduction-loop-exposure"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONLOOPEXPOSUREPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Reduction Loop Exposure pass"),
    llvm::cl::init(false));

namespace {

// Memory space of the reduction accumulator: the register file private to the
// compute unit that runs the reduction.  Must name a memory declared in the
// device spec — an unrecognised space asserts in address assignment.
constexpr llvm::StringLiteral kAccumulatorMemorySpace = "SFU_REG";

// ---------------------------------------------------------------------------
// Return the first linalg.generic with a reduction iterator found anywhere
// inside `stage`, or null if none exists.
// ---------------------------------------------------------------------------
static linalg::GenericOp findReductionGenericOp(ktdf::StageOp stage) {
  linalg::GenericOp found;
  stage.getBody()->walk([&](linalg::GenericOp generic) {
    for (auto it : generic.getIteratorTypesArray()) {
      if (it == utils::IteratorType::reduction) {
        found = generic;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// ---------------------------------------------------------------------------
// Return true if `token` appears in the depends_in list of `stage`.
// ---------------------------------------------------------------------------
static bool stageConsumesToken(ktdf::StageOp stage, Value token) {
  for (Value dep : stage.getDependsIn())
    if (dep == token) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------
struct ReductionLoopExposurePass
    : public ktdf::impl::ReductionLoopExposurePassBase<
          ReductionLoopExposurePass> {
  using ReductionLoopExposurePassBase<
      ReductionLoopExposurePass>::ReductionLoopExposurePassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();
    if (failed(transformModule(module))) signalPassFailure();
  }

 private:
  // -------------------------------------------------------------------------
  // Collect every ktdf.stage that directly holds a reduction linalg.generic
  // (i.e. the generic is at some depth inside the stage body), then rewrite
  // the parent pipeline of each such stage.  We collect first to avoid
  // walking ops that we are about to rewrite.
  // -------------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    SmallVector<ktdf::StageOp> compute_stages;
    module.walk([&](linalg::GenericOp generic) {
      // Check if this generic has a reduction iterator.
      bool has_reduction = false;
      for (auto it : generic.getIteratorTypesArray())
        if (it == utils::IteratorType::reduction) {
          has_reduction = true;
          break;
        }
      if (!has_reduction) return WalkResult::advance();

      // Walk up to find the immediately enclosing ktdf.stage.
      Operation* parent = generic->getParentOp();
      while (parent && !isa<ktdf::StageOp>(parent))
        parent = parent->getParentOp();
      if (!parent) return WalkResult::advance();

      auto stage = cast<ktdf::StageOp>(parent);
      // Avoid duplicates (multiple generics in same stage).
      if (llvm::find(compute_stages, stage) == compute_stages.end())
        compute_stages.push_back(stage);
      return WalkResult::advance();
    });

    for (auto compute_stage : compute_stages)
      if (failed(rewritePipeline(compute_stage))) return failure();

    return success();
  }

  // -------------------------------------------------------------------------
  // Rewrite the ktdf.pipeline that contains `compute_stage`.
  // -------------------------------------------------------------------------
  LogicalResult rewritePipeline(ktdf::StageOp compute_stage) {
    linalg::GenericOp generic_op = findReductionGenericOp(compute_stage);
    if (!generic_op) return success();  // already removed?

    // --- Determine reduction size R from the input tensor's reduction dim ---
    int64_t reduction_dim = -1;
    auto iter_types = generic_op.getIteratorTypesArray();
    for (int64_t i = 0; i < static_cast<int64_t>(iter_types.size()); ++i)
      if (iter_types[i] == utils::IteratorType::reduction) {
        reduction_dim = i;
        break;
      }
    if (reduction_dim < 0) {
      compute_stage.emitError(PASS_NAME ": reduction dim not found");
      return failure();
    }

    auto input_type =
        cast<RankedTensorType>(generic_op.getInputs().front().getType());
    int64_t reduction_size = input_type.getDimSize(reduction_dim);
    if (reduction_size == ShapedType::kDynamic) {
      LDBG(1) << PASS_NAME ": dynamic reduction size not supported — skipping";
      return success();
    }

    LDBG(1) << PASS_NAME ": reduction_dim=" << reduction_dim
            << " reduction_size=" << reduction_size;

    // --- Get the parent pipeline ---
    auto inner_pipeline = compute_stage->getParentOfType<ktdf::PipelineOp>();
    if (!inner_pipeline) {
      compute_stage.emitError(PASS_NAME
                              ": compute stage has no parent pipeline");
      return failure();
    }

    MLIRContext* ctx = inner_pipeline.getContext();
    IRRewriter rewriter(ctx);
    Location loc = inner_pipeline.getLoc();

    // -----------------------------------------------------------------------
    // 1. Shrink ALL FIFO slots in ktdf.private whose element count is a
    //    multiple of reduction_size.  Every FIFO in the inner pipeline that
    //    carries R*C elements per iteration must become C elements so that
    //    each loop iteration transfers exactly one slice.
    //
    //    We also locate fifo_in (written by the load stage) and fifo_out
    //    (read by the compute stage's write_to_fifo) by tracing def-use:
    //      fifo_in  — destination of the FIFO-dest data_transfer in the stage
    //                 whose depends_out token matches compute_stage.depends_in
    //      fifo_out — fifo_slot operand of the write_to_fifo in compute_stage
    // -----------------------------------------------------------------------
    ktdf::PrivateOp priv_op = inner_pipeline.getPrivateOp();
    if (!priv_op) {
      inner_pipeline.emitError(PASS_NAME ": no ktdf.private in pipeline");
      return failure();
    }

    // Find the load stage: the sibling whose depends_out token appears in
    // the compute stage's depends_in.
    ktdf::StageOp load_stage;
    for (Value tok : compute_stage.getDependsIn()) {
      for (auto sibling : inner_pipeline.getStages()) {
        if (sibling == compute_stage) continue;
        for (Value out_tok : sibling.getDependsOut()) {
          if (out_tok == tok) {
            load_stage = sibling;
            break;
          }
        }
        if (load_stage) break;
      }
      if (load_stage) break;
    }
    if (!load_stage) {
      inner_pipeline.emitError(
          PASS_NAME ": cannot find load stage upstream of compute stage");
      return failure();
    }

    // From the load stage, find the data_transfer whose destination is a FIFO.
    // Assert that there is exactly one such transfer.
    SmallVector<ktdf::DataTransferOp> fifo_dest_transfers;
    load_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
      if (dt.isDestFifo()) fifo_dest_transfers.push_back(dt);
    });
    if (fifo_dest_transfers.empty()) {
      inner_pipeline.emitError(PASS_NAME
                               ": no FIFO-dest data_transfer in load stage");
      return failure();
    }
    if (fifo_dest_transfers.size() > 1) {
      inner_pipeline.emitError(
          PASS_NAME
          ": multiple FIFO-dest data_transfers in load stage — not supported");
      return failure();
    }
    Value orig_fifo_in = fifo_dest_transfers.front().getDestination();

    // Validate that fifo_in is a ktdf.private result.
    auto orig_fifo_in_result = dyn_cast<OpResult>(orig_fifo_in);
    if (!orig_fifo_in_result || orig_fifo_in_result.getOwner() != priv_op) {
      inner_pipeline.emitError(PASS_NAME
                               ": fifo_in is not a ktdf.private result");
      return failure();
    }
    unsigned fifo_in_idx = orig_fifo_in_result.getResultNumber();

    // Build a map: original FifoSlotType → shrunken FifoSlotType for every
    // private result whose element count is divisible by reduction_size.
    // This covers fifo_in and any other intermediate FIFOs in the pipeline.
    llvm::DenseMap<unsigned, ktdf::FifoSlotType> shrunken_fifo_map;
    SmallVector<Type> new_result_types(priv_op.getResultTypes());
    for (unsigned i = 0; i < priv_op.getNumResults(); ++i) {
      auto fifo_type =
          dyn_cast<ktdf::FifoSlotType>(priv_op.getResult(i).getType());
      if (!fifo_type) continue;
      int64_t n = fifo_type.getNumElements();
      if (n % reduction_size != 0) continue;
      auto shrunken = ktdf::FifoSlotType::get(
          ctx, fifo_type.getSrc(), fifo_type.getDest(), n / reduction_size,
          fifo_type.getElementType());
      shrunken_fifo_map[i] = shrunken;
      new_result_types[i] = shrunken;
    }
    if (shrunken_fifo_map.empty()) {
      inner_pipeline.emitError(PASS_NAME
                               ": no FIFO slots divisible by reduction_size");
      return failure();
    }

    // Replace priv_op with a new one carrying all shrunken FIFO types.
    rewriter.setInsertionPoint(priv_op);
    auto new_priv = ktdf::PrivateOp::create(rewriter, loc, new_result_types);
    rewriter.mergeBlocks(&priv_op.getRegion().front(),
                         &new_priv.getRegion().front(), {});

    // Patch every fifo.allocate inside the new private body.
    new_priv.getRegion().front().walk([&](ktdf::FifoAllocateOp alloc) {
      for (auto& [idx, shrunken] : shrunken_fifo_map) {
        auto orig_type =
            dyn_cast<ktdf::FifoSlotType>(priv_op.getResult(idx).getType());
        if (orig_type && alloc.getResult(0).getType() == orig_type)
          alloc.getResult(0).setType(shrunken);
      }
    });

    for (auto [old_res, new_res] :
         llvm::zip(priv_op.getResults(), new_priv.getResults()))
      old_res.replaceAllUsesWith(new_res);
    rewriter.eraseOp(priv_op);

    Value fifo_in = new_priv.getResult(fifo_in_idx);

    // fifo_out: the fifo_slot operand of the write_to_fifo in compute_stage
    // (already updated to new_priv after RAUW above).
    Value fifo_out;
    compute_stage.getBody()->walk([&](ktdf::WriteToFifoOp w) {
      fifo_out = w.getFifoSlot();
      return WalkResult::interrupt();
    });
    if (!fifo_out) {
      inner_pipeline.emitError(PASS_NAME ": no write_to_fifo in compute stage");
      return failure();
    }

    // Accumulator buffer type: the generic's output shape, in the compute
    // unit's register-file memory space.
    auto output_tensor_type =
        cast<RankedTensorType>(generic_op.getOutputs().front().getType());
    auto acc_memref_type = MemRefType::get(
        output_tensor_type.getShape(), output_tensor_type.getElementType(),
        /*layout=*/MemRefLayoutAttrInterface{},
        StringAttr::get(ctx, kAccumulatorMemorySpace));

    // Per-iteration input slice: same shape as the generic input but size-1 in
    // the reduction dim (e.g. 1x256x64 → 1x1x64).  The slice is produced by
    // read_from_fifo, so it carries no memory space.
    auto slice_tensor_type =
        cast<RankedTensorType>(generic_op.getInputs().front().getType());
    SmallVector<int64_t> slice_shape(slice_tensor_type.getShape().begin(),
                                     slice_tensor_type.getShape().end());
    slice_shape[static_cast<size_t>(reduction_dim)] = 1;
    auto one_row_memref_type =
        MemRefType::get(slice_shape, slice_tensor_type.getElementType());

    // -----------------------------------------------------------------------
    // 1b. Patch data_transfer static sizes for any transfer that uses a
    //     shrunken FIFO.  This applies to all stages (plain, compute, store).
    //     For each data_transfer whose source or destination is now a shrunken
    //     FIFO, divide the corresponding static sizes by reduction_size.
    // -----------------------------------------------------------------------
    inner_pipeline->walk([&](ktdf::DataTransferOp dt) {
      // Check if the destination FIFO was shrunken.
      if (dt.isDestFifo()) {
        auto fifo_res = dyn_cast<OpResult>(dt.getDestination());
        if (fifo_res && fifo_res.getOwner() == new_priv &&
            shrunken_fifo_map.count(fifo_res.getResultNumber())) {
          if (auto sizes = dt.getStaticDestSizes()) {
            SmallVector<int64_t> new_sizes;
            for (int64_t s : *sizes) new_sizes.push_back(s / reduction_size);
            dt.setStaticDestSizesAttr(DenseI64ArrayAttr::get(ctx, new_sizes));
          }
          // Also patch source_sizes when source is a memref (not a FIFO).
          // The reduction dimension is baked into the memref source sizes and
          // must be divided now that the transfer runs once per iteration.
          if (!dt.isSourceFifo()) {
            if (auto sizes = dt.getStaticSourceSizes()) {
              SmallVector<int64_t> new_sizes(sizes->begin(), sizes->end());
              for (int64_t i = 0; i < static_cast<int64_t>(new_sizes.size());
                   ++i) {
                if (new_sizes[i] == reduction_size) {
                  new_sizes[i] = 1;
                  break;
                }
              }
              dt.setStaticSourceSizesAttr(
                  DenseI64ArrayAttr::get(ctx, new_sizes));
            }
          }
        }
      }
      // Check if the source FIFO was shrunken.
      if (dt.isSourceFifo()) {
        auto fifo_res = dyn_cast<OpResult>(dt.getSource());
        if (fifo_res && fifo_res.getOwner() == new_priv &&
            shrunken_fifo_map.count(fifo_res.getResultNumber())) {
          if (auto sizes = dt.getStaticSourceSizes()) {
            SmallVector<int64_t> new_sizes;
            for (int64_t s : *sizes) new_sizes.push_back(s / reduction_size);
            dt.setStaticSourceSizesAttr(DenseI64ArrayAttr::get(ctx, new_sizes));
          }
        }
      }
    });

    // -----------------------------------------------------------------------
    // 2. Build shared loop-bound constants before the inner pipeline.
    // -----------------------------------------------------------------------
    rewriter.setInsertionPoint(inner_pipeline);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c_red = arith::ConstantIndexOp::create(rewriter, loc, reduction_size);

    // -----------------------------------------------------------------------
    // 3. Find the store stage: the sibling stage whose depends_in includes a
    //    token that the compute stage lists in depends_out.  Its data_transfer
    //    runs unconditionally every iteration (last write wins).
    // -----------------------------------------------------------------------
    ktdf::StageOp conditional_store_stage;
    for (Value tok : compute_stage.getDependsOut()) {
      for (auto sibling : inner_pipeline.getStages()) {
        if (sibling == compute_stage) continue;
        if (stageConsumesToken(sibling, tok)) {
          conditional_store_stage = sibling;
          break;
        }
      }
      if (conditional_store_stage) break;
    }

    // -----------------------------------------------------------------------
    // 4. Rewrite every stage: wrap its body in scf.for %r = 0 to R step 1.
    //    Extra surgery for the compute stage and the conditional store stage.
    // -----------------------------------------------------------------------
    for (auto stage : inner_pipeline.getStages()) {
      if (stage == compute_stage) {
        rewriteComputeStage(rewriter, loc, ctx, stage, generic_op,
                            one_row_memref_type, acc_memref_type, fifo_in,
                            fifo_out, c0, c1, c_red);
      } else if (stage == conditional_store_stage) {
        rewriteConditionalStoreStage(rewriter, loc, stage, c0, c1, c_red);
      } else if (stage == load_stage) {
        rewriteLoadStage(rewriter, loc, stage, c0, c1, c_red, reduction_size);
      } else {
        rewritePlainStage(rewriter, loc, stage, c0, c1, c_red);
      }
    }

    return success();
  }

  // -------------------------------------------------------------------------
  // Wrap the entire body of a plain (non-compute, non-conditional) stage in
  // scf.for %r = 0 to R.  All existing ops move into the loop body as-is.
  // -------------------------------------------------------------------------
  void rewritePlainStage(IRRewriter& rewriter, Location loc,
                         ktdf::StageOp stage, Value c0, Value c1, Value c_red) {
    Block* body = stage.getBody();

    // Collect all existing ops (snapshot before insertion).
    SmallVector<Operation*> ops;
    for (auto& op : *body) ops.push_back(&op);

    rewriter.setInsertionPointToStart(body);
    auto red_for = scf::ForOp::create(rewriter, loc, c0, c_red, c1);

    // Move all original ops into the loop body, before the terminator.
    Operation* loop_term = red_for.getBody()->getTerminator();
    for (auto* op : ops) op->moveBefore(loop_term);
  }

  // -------------------------------------------------------------------------
  // Rewrite the load stage: same as plain (wrap in scf.for), but also
  // patch the source_map of any memref-to-FIFO data_transfer to use the
  // loop IV at the reduction-dimension position.
  //
  // The source subscript is encoded as an AffineMap (source_map) over the
  // source_indices SSA operands.  E.g. source_map = (d0) -> (d0, 0, 0, 0)
  // with one SSA operand.  After the fix: (d0, d1) -> (d0, 0, d1, 0) with
  // the loop IV appended to source_indices.
  // -------------------------------------------------------------------------
  void rewriteLoadStage(IRRewriter& rewriter, Location loc,
                        ktdf::StageOp stage, Value c0, Value c1, Value c_red,
                        int64_t reduction_size) {
    Block* body = stage.getBody();

    // Collect all existing ops (snapshot before insertion).
    SmallVector<Operation*> ops;
    for (auto& op : *body) ops.push_back(&op);

    rewriter.setInsertionPointToStart(body);
    auto red_for = scf::ForOp::create(rewriter, loc, c0, c_red, c1);

    // Move all original ops into the loop body, before the terminator.
    Operation* loop_term = red_for.getBody()->getTerminator();
    for (auto* op : ops) op->moveBefore(loop_term);

    // Patch source_map: for each memref-to-FIFO data_transfer, add the loop
    // IV as a new dimension in the affine map at the reduction-dim result
    // position.
    Value iv = red_for.getInductionVar();
    MLIRContext* ctx = rewriter.getContext();
    red_for.getBody()->walk([&](ktdf::DataTransferOp dt) {
      if (!dt.isSourceMemRef() || !dt.isDestFifo()) return WalkResult::advance();

      // Identify the reduction dim in the source memref shape.
      auto src_memref_type = cast<MemRefType>(dt.getSource().getType());
      ArrayRef<int64_t> shape = src_memref_type.getShape();
      int64_t red_idx = -1;
      for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        if (shape[i] == reduction_size) {
          red_idx = i;
          break;
        }
      }
      if (red_idx < 0) return WalkResult::advance();

      // Get the current source_map. It has N input dims mapping to M results.
      // We add one more input dim (for the loop IV) and replace result[red_idx]
      // with that new dim.
      auto map_attr = dt.getSourceMapAttr();
      if (!map_attr) return WalkResult::advance();
      AffineMap old_map = map_attr.getValue();
      unsigned num_dims = old_map.getNumDims();
      unsigned new_dim_pos = num_dims;  // new dim at the end

      // Build new result expressions: replace the constant-0 at red_idx with
      // the new dimension.
      SmallVector<AffineExpr> new_results;
      for (unsigned i = 0; i < old_map.getNumResults(); ++i) {
        if (static_cast<int64_t>(i) == red_idx) {
          new_results.push_back(getAffineDimExpr(new_dim_pos, ctx));
        } else {
          new_results.push_back(old_map.getResult(i));
        }
      }
      AffineMap new_map = AffineMap::get(num_dims + 1, old_map.getNumSymbols(),
                                         new_results, ctx);
      dt.setSourceMapAttr(AffineMapAttr::get(new_map));

      // Append the loop IV to source_indices operands.
      dt.getSourceIndicesMutable().append(iv);

      return WalkResult::advance();
    });
  }

  // -------------------------------------------------------------------------
  // Rewrite the compute stage:
  //   %acc  = memref.alloc() : acc_memref_type                (before loop)
  //   %zero = arith.constant 0
  //   linalg.fill ins(%zero) outs(%acc)
  //   scf.for %r = 0 to R                     {loop_type = reduction_loop}
  //     %slice = read_from_fifo -> one_row_memref_type
  //     linalg.generic ins(%slice) outs(%acc)  (no results; in-place)
  //     write_to_fifo %acc, fifo_out           (unconditional every iter)
  //
  // The loop carries no iter_args: the accumulator lives in the buffer, so the
  // buffer-semantics linalg.generic has no SSA result to thread through.
  // -------------------------------------------------------------------------
  void rewriteComputeStage(IRRewriter& rewriter, Location loc, MLIRContext* ctx,
                           ktdf::StageOp stage, linalg::GenericOp generic_op,
                           MemRefType one_row_memref_type,
                           MemRefType acc_memref_type, Value fifo_in,
                           Value fifo_out, Value c0, Value c1, Value c_red) {
    Block* body = stage.getBody();

    // Collect existing ops to erase after the rewrite.
    SmallVector<Operation*> to_erase;
    for (auto& op : *body) to_erase.push_back(&op);

    rewriter.setInsertionPointToStart(body);

    // Accumulator buffer and its zero initialiser — emitted once, outside the
    // loop, so that address assignment gives the buffer a physical offset.
    Value acc = memref::AllocOp::create(rewriter, loc, acc_memref_type);
    Value zero = arith::ConstantOp::create(
        rewriter, loc,
        cast<TypedAttr>(
            rewriter.getZeroAttr(acc_memref_type.getElementType())));
    linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{acc});

    auto acc_for = scf::ForOp::create(rewriter, loc, c0, c_red, c1);
    rewriter.setInsertionPointToStart(acc_for.getBody());

    // read_from_fifo: one row at a time.
    auto slice =
        ktdf::ReadFromFifoOp::create(rewriter, loc, one_row_memref_type,
                                     fifo_in);

    // Rebuild the linalg.generic with buffer semantics: same maps, same
    // iterators, same body, but no result — it accumulates into %acc in place.
    auto new_generic = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{slice.getResult()}, /*outputs=*/ValueRange{acc},
        generic_op.getIndexingMapsArray(), generic_op.getIteratorTypesArray());
    IRMapping mapping;
    generic_op.getRegion().cloneInto(&new_generic.getRegion(), mapping);

    // Publish the running accumulator to fifo_out every iteration.  LXSU
    // overwrites the same address each iteration; last write wins.  FIFO
    // send/receive must be unconditional and balanced every iteration.
    ktdf::WriteToFifoOp::create(rewriter, loc, acc, fifo_out);

    acc_for->setAttr("loop_type", ktdf::LoopTypeAttr::get(
                                      ctx, ktdf::LoopType::ReductionLoop));

    // Erase original body ops (reverse order, only if unused).
    for (auto* op : llvm::reverse(to_erase))
      if (op->use_empty()) rewriter.eraseOp(op);
  }

  // -------------------------------------------------------------------------
  // Rewrite the store stage:
  //   scf.for %r = 0 to R
  //     <all original body ops including the data_transfer — unconditional>
  //
  // The data_transfer runs every iteration (LXSU overwrites the same LX
  // address; last write wins = final sum).  FIFO send/receive must be
  // unconditional and balanced every iteration to avoid hardware deadlock.
  // -------------------------------------------------------------------------
  void rewriteConditionalStoreStage(IRRewriter& rewriter, Location loc,
                                    ktdf::StageOp stage, Value c0, Value c1,
                                    Value c_red) {
    Block* body = stage.getBody();

    // Collect all existing ops.
    SmallVector<Operation*> all_ops;
    for (auto& op : *body) all_ops.push_back(&op);

    rewriter.setInsertionPointToStart(body);
    auto red_for = scf::ForOp::create(rewriter, loc, c0, c_red, c1);
    Operation* loop_term = red_for.getBody()->getTerminator();

    // Move all ops into the loop body unconditionally.
    for (auto* op : all_ops) op->moveBefore(loop_term);
  }
};

}  // namespace

auto mlir::ktdf::createReductionLoopExposurePass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionLoopExposurePass>();
}
