//===----------------------------------------------------------------------===//
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
// This pass constructs a three-stage pipeline.
//
// This pass performs the following steps in order to create a three-stage
// pipeline:
//   Step 1: Generalize linalg, arith, and math operations to linalg.generic
//   Step 2: Fuse consecutive linalg.generic operations
//   Step 3: Create loops from linalg operations (tiling)
//   Step 4: Create pipeline with three stages (load, compute, store)
//   Step 5: Replace access tiles with memref.reinterpret_cast
//   Step 6: Cleanup
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/LogicalResult.h>

#include "Ktdp/KtdpAttrs.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Links.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Utils/CustomLinalgTiling.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "construct-three-stage-pipeline"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_CONSTRUCTTHREESTAGEPIPELINEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {
const char VerboseDebug[] = DEBUG_TYPE "-verbose";

static llvm::cl::opt<bool> DisableThisPass(
    "construct-three-stage-pipeline-disable",
    llvm::cl::desc("Disable construction of three stage pipeline"),
    llvm::cl::init(false));

}  // unnamed namespace

namespace {

template <class T>
[[nodiscard]]
auto maxOrDefault(llvm::ArrayRef<T> items) -> T {
  if (items.empty()) {
    return {};
  }

  T result = items.front();
  for (auto item : items.drop_front()) {
    result = std::max(result, item);
  }

  return result;
}

}  // namespace

namespace {

struct ConstructThreeStagePipelinePass
    : public impl::ConstructThreeStagePipelinePassBase<
          ConstructThreeStagePipelinePass> {
  ConstructThreeStagePipelinePass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    ConstructThreeStagePipelinePassBase::getDependentDialects(registry);
    // The reduction accumulator rewrite emits memref.alloca and
    // bufferization.to_tensor / materialize_in_destination ops.
    registry.insert<mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect>();
  }

  void runOnOperation() final;

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  // Process a single function
  void runOnFunc(mlir::func::FuncOp func_op);

  // Reset state between function processing
  void resetState();

  // Generalize named linalg operations to linalg.generic
  mlir::LogicalResult generalizeLinalgArithMathOps(mlir::func::FuncOp func_op);

  // Fuse consecutive linalg.generic operations
  mlir::LogicalResult fuseLinalgOps(mlir::func::FuncOp func_op);

  // Determine tile sizes based on vector_length from linalg operation
  llvm::SmallVector<int64_t> determineTileSizes(
      mlir::linalg::LinalgOp linalgOp);

  // Create loops from linalg operations by tiling
  void createLoopsFromLinalg(llvm::ArrayRef<mlir::linalg::LinalgOp> linalg_ops);

  // Rewrite a matmul K-reduction loop's tensor iter_arg accumulator into a
  // compute-local (LRF) memref RMW with an scf.if first-tile seed. Returns true
  // if a reduction loop was found and rewritten. The ktdf.pipeline has no
  // results, so the accumulator cannot be a loop-carried tensor.
  bool rewriteReductionAccumulator(mlir::func::FuncOp func_op);

  // Create a 3-stage pipeline inside innermost_loop, with one stage for loads,
  // computes, stores.
  void createPipeline(mlir::scf::ForOp innermost_loop);

  // Create linalg compute operations in stage 2
  void createComputeOps(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::ktdf::PrivateOp private_op);

  // Create data transfer operations for loads or stores
  // is_load=true: transfer from access_tile to FIFO slot
  // is_load=false: transfer from FIFO slot to access_tile
  // private_result_offset: index into private_op results where FIFO slots for
  //   this transfer start.
  void createDataTransfers(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::ktdf::PrivateOp private_op,
                           llvm::ArrayRef<int64_t> tile_sizes, bool is_load,
                           size_t private_result_offset);

  // Create ktdf.private operation with FIFO slots and tokens
  // Returns the created private operation
  mlir::ktdf::PrivateOp createPrivateOp(mlir::OpBuilder& builder,
                                        mlir::Location loc);

  // Annotate loops with loop_type attributes based on linalg iterator types
  void annotateLoopsWithIteratorTypes(llvm::ArrayRef<mlir::Operation*> loops,
                                      mlir::linalg::GenericOp generic_op);

  // Delete op and recursively delete unused chain of operands
  void deleteOpAndUnusedChainOfOperands(mlir::Operation* op);

  // Replace construct_access_tile with memref.reinterpret_cast
  void replaceAccessTilesWithReinterpretCast(mlir::func::FuncOp func_op);

  // Map ktdp memory space attribute to device namespace using mem_space_mapping
  mlir::Attribute mapMemorySpace(mlir::Attribute ktdp_memory_space);

  // Get FIFO attributes for a load operation (memory -> compute unit)
  std::pair<mlir::Attribute, mlir::Attribute> getFifoAttributesForLoad(
      mlir::ktdp::LoadOp load_op);

  // Get FIFO attributes for a store operation (compute unit -> memory)
  std::pair<mlir::Attribute, mlir::Attribute> getFifoAttributesForStore(
      mlir::ktdp::StoreOp store_op);

  // Compute offset for reinterpret_cast from indices and strides
  mlir::Value computeReinterpretCastOffset(
      mlir::OpBuilder& builder, mlir::Location loc,
      llvm::SmallVector<mlir::Value>& indices,
      llvm::SmallVector<int64_t>& strides);

  // Get strides from memref type, computing default row-major strides if needed
  llvm::SmallVector<int64_t> getStridesFromMemRefType(
      mlir::MemRefType memref_type);

  // Create a pipeline for a post-reduction-loop store (Gap 5).
  // Called once per post-loop ktdp.store found outside the reduction loop.
  // The pipeline has a compute stage (write_to_fifo) and a store stage
  // (data_transfer → HBM), mirroring the store stage of the elementwise pipeline
  // but placed after the K-loop body.
  void createPostLoopStorePipeline(mlir::ktdp::StoreOp store_op,
                                   mlir::scf::ForOp enclosing_n_loop);

  // Clean up operations after pipeline creation
  void cleanupOperations();

  // Member variables
  const SchedulerExtContext& scheduler_ctx_;

  arch_view::ResourceKinds* resource_kinds_;

  // Collected ktdp.load and ktdp.store operations
  llvm::SmallVector<mlir::ktdp::LoadOp> load_ops_;
  llvm::SmallVector<mlir::ktdp::StoreOp> store_ops_;
  llvm::SmallVector<mlir::linalg::LinalgOp> compute_ops_;

  // Tiled loops from linalg tiling (outermost to innermost)
  llvm::SmallVector<mlir::Operation*> tiled_loops_;

  // K-reduction loop detected before the accumulator rewrite (null for
  // elementwise ops). Used by createLoopsFromLinalg to detect the matmul
  // without re-running the iter_arg detection after the rewrite changed it.
  mlir::scf::ForOp reduction_loop_;

  // Tile sizes determined from linalg operation
  llvm::SmallVector<int64_t> tile_sizes_;

  // Total number of elements (product of tile_sizes_)
  int64_t total_num_elements_ = 0;

  // Operations to delete after pipeline creation
  llvm::SmallVector<mlir::Operation*> ops_to_delete_;

  // LRF accumulator buffer (memref.alloca result), set by
  // rewriteReductionAccumulator when a reduction loop is found. Used by
  // createComputeOps to emit the in-place LRF read-modify-write (memref load
  // + linalg.generic + memref store) in the compute stage.
  // Null for elementwise ops.
  mlir::Value lrf_acc_buf_;

  // Builder for constants at function start
  std::optional<mlir::OpBuilder> const_builder_;
};

void ConstructThreeStagePipelinePass::resetState() {
  load_ops_.clear();
  store_ops_.clear();
  compute_ops_.clear();
  tiled_loops_.clear();
  tile_sizes_.clear();
  total_num_elements_ = 0;
  ops_to_delete_.clear();
  const_builder_.reset();
  reduction_loop_ = {};
  lrf_acc_buf_ = {};
}

mlir::LogicalResult
ConstructThreeStagePipelinePass::generalizeLinalgArithMathOps(
    mlir::func::FuncOp func_op) {
  LLVM_DEBUG(
      llvm::dbgs()
      << "Generalizing linalg, arith, and math operations to linalg.generic\n");

  llvm::SmallVector<mlir::linalg::LinalgOp> linalg_ops;

  func_op.walk([&](mlir::Operation* op) {
    // Collect linalg operations that are not already generic
    if (auto linalg_op = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
      if (!mlir::isa<mlir::linalg::GenericOp>(op)) {
        linalg_ops.push_back(linalg_op);
      }
    }
  });

  mlir::IRRewriter rewriter(&getContext());

  // Generalize named linalg operations
  for (mlir::linalg::LinalgOp linalg_op : linalg_ops) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Generalizing linalg op: " << linalg_op->getName() << "\n");
    rewriter.setInsertionPoint(linalg_op);

    llvm::FailureOr<mlir::linalg::GenericOp> generic_result =
        mlir::linalg::generalizeNamedOp(rewriter, linalg_op);

    if (mlir::failed(generic_result)) {
      linalg_op.emitError("Failed to generalize linalg operation");
      return mlir::failure();
    }
  }

  // Convert arith and math operations to linalg using elementwise patterns
  LLVM_DEBUG(llvm::dbgs() << "  Converting arith/math operations\n");

  mlir::RewritePatternSet patterns(&getContext());
  mlir::linalg::populateElementwiseToLinalgConversionPatterns(patterns);

  if (mlir::failed(mlir::applyPatternsGreedily(func_op, std::move(patterns)))) {
    func_op.emitError(
        "Failed to convert arith/math operations to linalg.generic");
    return mlir::failure();
  }

  LLVM_DEBUG(llvm::dbgs()
             << "  Successfully generalized compute ops to linalg.generic\n");
  return mlir::success();
}

mlir::LogicalResult ConstructThreeStagePipelinePass::fuseLinalgOps(
    mlir::func::FuncOp func_op) {
  LLVM_DEBUG(llvm::dbgs() << "Fusing consecutive linalg.generic operations\n");

  llvm::SmallVector<mlir::linalg::GenericOp> worklist;
  func_op.walk([&](mlir::linalg::GenericOp generic_op) {
    worklist.push_back(generic_op);
  });

  mlir::IRRewriter rewriter(&getContext());
  while (!worklist.empty()) {
    mlir::linalg::GenericOp consumer = worklist.pop_back_val();

    // Try to fuse producer into this consumer
    for (mlir::OpOperand* operand : consumer.getDpsInputOperands()) {
      mlir::Value input = operand->get();
      auto producer = input.getDefiningOp<mlir::linalg::GenericOp>();

      if (!producer) continue;

      rewriter.setInsertionPoint(consumer);
      // TODO: we may want do non-element-wise fusion as well such as matmul
      // followed by add.
      llvm::FailureOr<mlir::linalg::ElementwiseOpFusionResult> fusion_result =
          mlir::linalg::fuseElementwiseOps(rewriter, operand);

      if (mlir::succeeded(fusion_result)) {
        LLVM_DEBUG(llvm::dbgs() << "  Successfully fused operations\n");
        auto fused_op =
            mlir::cast<mlir::linalg::GenericOp>(fusion_result->fusedOp);

        // Replace uses of consumer with the fused operation (this also erases
        // consumer)
        rewriter.replaceOp(consumer, fused_op->getResults());

        // Erase the original producer operation
        rewriter.eraseOp(producer);

        // Remove producer from worklist if present
        llvm::erase(worklist, producer);

        // Add the fused operation to the worklist for further fusion in the
        // next iteration.
        worklist.push_back(fused_op);
        break;
      }
    }
  }

  return mlir::success();
}

llvm::SmallVector<int64_t> ConstructThreeStagePipelinePass::determineTileSizes(
    mlir::linalg::LinalgOp linalg_op) {
  assert(linalg_op->getNumResults() == 1 &&
         "linalg op expected to have exactly one tensor result");

  mlir::ShapedType shaped_type =
      llvm::dyn_cast<mlir::ShapedType>(linalg_op->getResult(0).getType());
  assert(shaped_type && shaped_type.hasRank());

  mlir::Type elem_type = shaped_type.getElementType();

  auto simd_feature =
      resource_kinds_->getFeature<mlir::ktdf_arch::feature::SIMD>(
          resource_kinds_->getComputeKind());
  const auto vector_length =
      std::max(simd_feature.getLanes(elem_type), int64_t(1));

  llvm::SmallVector<int64_t> tile_sizes;

  llvm::ArrayRef<int64_t> shape = shaped_type.getShape();
  int64_t rank = shape.size();

  // Start from rightmost dimension and multiply until we reach vector_length
  int64_t product = 1;
  int64_t covered_dims = 0;

  for (int64_t i = rank - 1; i >= 0; --i) {
    product *= shape[i];
    covered_dims++;
    if (product >= vector_length) {
      break;
    }
  }

  // Build tile sizes: 1 for uncovered dims, partial/full for covered dims
  tile_sizes.resize(rank);
  for (int64_t i = 0; i < rank - covered_dims; ++i) {
    tile_sizes[i] = 1;
  }

  // For covered dimensions, compute tile sizes that divide evenly
  int64_t remaining_product = vector_length;
  for (int64_t i = rank - 1; i >= rank - covered_dims; --i) {
    int64_t dim_size = shape[i];

    // Use GCD to find largest value that divides both dim_size and
    // remaining_product
    int64_t tile_size = std::gcd(dim_size, remaining_product);

    // Clamp to dimension size
    tile_size = std::min(tile_size, dim_size);

    tile_sizes[i] = tile_size;
    remaining_product /= tile_size;
    if (remaining_product <= 1) remaining_product = 1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Shape: [";
    for (int64_t i = 0; i < rank; ++i) {
      llvm::dbgs() << shape[i];
      if (i < rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n    Tile sizes: [";
    for (int64_t i = 0; i < rank; ++i) {
      llvm::dbgs() << tile_sizes[i];
      if (i < rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
  });

  return tile_sizes;
}

void ConstructThreeStagePipelinePass::annotateLoopsWithIteratorTypes(
    llvm::ArrayRef<mlir::Operation*> loops,
    mlir::linalg::GenericOp generic_op) {
  assert(generic_op);

  // Annotate each loop with its corresponding iterator type obtained from
  // generic_op.
  const auto& iterator_types = generic_op.getIteratorTypesArray();
  for (size_t i = 0; i < loops.size() && i < iterator_types.size(); ++i) {
    auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(loops[i]);
    assert(for_op);

    mlir::ktdf::LoopType loop_type;
    if (iterator_types[i] == mlir::utils::IteratorType::parallel) {
      loop_type = mlir::ktdf::LoopType::ParallelLoop;
    } else if (iterator_types[i] == mlir::utils::IteratorType::reduction) {
      loop_type = mlir::ktdf::LoopType::ReductionLoop;
    } else {
      for_op->emitError("Unsupported iterator type");
      signalPassFailure();
      return;
    }

    auto loop_type_attr =
        mlir::ktdf::LoopTypeAttr::get(&getContext(), loop_type);
    for_op->setAttr("loop_type", loop_type_attr);

    LLVM_DEBUG(llvm::dbgs() << "  Annotated loop " << i << " with loop_type: "
                            << (loop_type == mlir::ktdf::LoopType::ParallelLoop
                                    ? "parallel"
                                    : "reduction")
                            << "\n");
  }
}

bool ConstructThreeStagePipelinePass::rewriteReductionAccumulator(
    mlir::func::FuncOp func_op) {
  mlir::MLIRContext* ctx = &getContext();

  // Find the reduction loop: an scf.for with exactly one tensor iter_arg that
  // a linalg op accumulates into (its DPS init is the iter_arg).
  mlir::scf::ForOp k_loop;
  mlir::linalg::LinalgOp accum_lop;
  func_op.walk([&](mlir::scf::ForOp for_op) {
    if (for_op.getNumRegionIterArgs() != 1) return;
    mlir::Value iter_arg = for_op.getRegionIterArg(0);
    if (!mlir::isa<mlir::RankedTensorType>(iter_arg.getType())) return;
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(for_op.getBody()->getTerminator());
    auto lop =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
            yield.getOperand(0).getDefiningOp());
    if (!lop) return;
    for (mlir::Value init : lop.getDpsInits())
      if (init == iter_arg) {
        k_loop = for_op;
        accum_lop = lop;
      }
  });
  if (!k_loop) return false;

  mlir::Value iter_arg = k_loop.getRegionIterArg(0);
  auto acc_tensor_type = mlir::cast<mlir::RankedTensorType>(iter_arg.getType());
  mlir::Value zero_init = k_loop.getInitArgs()[0];  // zero, dominates the loop
  llvm::ArrayRef<int64_t> shape = acc_tensor_type.getShape();
  mlir::Type elem_type = acc_tensor_type.getElementType();
  mlir::Location loc = k_loop.getLoc();

  // Mark the K-loop so later passes know the accumulator is LRF-resident.
  k_loop->setAttr("ktdf.reduction_accumulator", mlir::UnitAttr::get(ctx));

  mlir::OpBuilder pre(k_loop);  // inserts before the reduction loop

  // Emit the accumulator as a Source-B logical-memory-view source: an
  // unrealized_conversion_cast(offset:index -> memref<NxT,"lrfreg">). Pass-20's
  // LogicalMemoryViewBuilder::replaceSourceBCasts turns exactly this form into a
  // dataflow.get_logical_memory_view (backed by get_local_unit "lrfreg" once the
  // lrfreg resolved-unit branch is added in buildResolvedUnits). A raw
  // memref.alloca is NOT recognized by that builder, so it must not be used.
  // Offset 0: a single per-N-stick accumulator in the local register file.
  mlir::Attribute lrf_space = mlir::StringAttr::get(ctx, "lrfreg");
  mlir::MemRefType lrf_buf_type = mlir::MemRefType::get(shape, elem_type,
                                                         /*layout=*/mlir::AffineMapAttr{},
                                                         lrf_space);
  mlir::Value lrf_off0 = mlir::arith::ConstantIndexOp::create(pre, loc, 0);
  auto lrf_cast = mlir::UnrealizedConversionCastOp::create(
      pre, loc, mlir::TypeRange{lrf_buf_type}, mlir::ValueRange{lrf_off0});
  lrf_acc_buf_ = lrf_cast.getResult(0);

  // Delta D-1: the constant zero-seed is REMOVED. The reduction accumulator is
  // initialised by the peeled first iteration in the DFIR-level peel step
  // (peelReductionComputeLoop in KTDFLowToDFIR.cpp). Storing a constant zero
  // to an SFP-local lrfreg at pass-02 fails on HW ("ConstantBitstreamOp
  // producers are only supported in L3"). Leave the buffer uninitialized here.

  // Replace uses of k_loop's SSA result (acc_final) with a post-loop memref
  // load of the alloca buffer.  After dropping the iter_arg the loop has no
  // results; the post-loop store pipeline (Gap 5) reads the alloca directly.
  mlir::IRRewriter rewriter(ctx);
  rewriter.setInsertionPointAfter(k_loop);
  // Load the accumulated result from the alloca as a tensor.
  auto post_tensor = mlir::bufferization::ToTensorOp::create(
      rewriter, loc, acc_tensor_type, lrf_acc_buf_,
      /*restrict=*/true, /*writable=*/false);
  k_loop->getResult(0).replaceAllUsesWith(post_tensor.getResult());

  // Build a replacement scf.for without the iter_arg.
  rewriter.setInsertionPoint(k_loop);
  auto new_loop = mlir::scf::ForOp::create(
      rewriter, loc, k_loop.getLowerBound(), k_loop.getUpperBound(),
      k_loop.getStep(), /*iterArgs=*/mlir::ValueRange{});

  mlir::Block* old_body = k_loop.getBody();
  mlir::Block* new_body = new_loop.getBody();
  // Replace IV and iter_arg block args.
  old_body->getArgument(0).replaceAllUsesWith(new_body->getArgument(0));
  // The iter_arg (%acc) — its only remaining in-loop use is as the linalg DPS
  // init, which createComputeOps will replace with an LRF load in stage 2.
  // Replace with a dummy tensor.empty so the IR is valid during the rewrite.
  if (!iter_arg.use_empty()) {
    rewriter.setInsertionPoint(&old_body->front());
    auto dummy = mlir::tensor::EmptyOp::create(rewriter, loc, shape, elem_type);
    iter_arg.replaceAllUsesWith(dummy.getResult());
  }

  // Fix the yield: original yield operand is the linalg result; change to
  // empty (no iter_args).
  auto yield_op =
      mlir::cast<mlir::scf::YieldOp>(old_body->getTerminator());
  rewriter.setInsertionPoint(yield_op);
  mlir::scf::YieldOp::create(rewriter, yield_op.getLoc(), mlir::ValueRange{});
  rewriter.eraseOp(yield_op);

  // Remove the default terminator that ForOp::create inserts into new_body
  // before splicing the old body's ops (which already contain our empty yield).
  if (!new_body->empty())
    new_body->back().erase();

  // Splice ops from old_body into new_body.
  new_body->getOperations().splice(new_body->begin(),
                                   old_body->getOperations());

  // Copy loop attributes (including ktdf.reduction_accumulator).
  for (auto attr : k_loop->getAttrs())
    new_loop->setAttr(attr.getName(), attr.getValue());

  rewriter.eraseOp(k_loop);
  reduction_loop_ = new_loop;
  return true;
}

void ConstructThreeStagePipelinePass::createLoopsFromLinalg(
    llvm::ArrayRef<mlir::linalg::LinalgOp> linalg_ops) {
  LLVM_DEBUG(
      llvm::dbgs() << "Creating SCF loops from linalg.generic operations\n");

  assert(linalg_ops.size() <= 1 &&
         "Currently only supporting one linalg.generic operation after fusion");

  mlir::IRRewriter rewriter(&getContext());
  for (mlir::linalg::LinalgOp linalg_op : linalg_ops) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Processing: " << linalg_op->getName() << "\n");

    rewriter.setInsertionPoint(linalg_op);

    // Reduction-accumulation (matmul) case: the compute op is inside the K-loop
    // we detected before the accumulator rewrite. Reuse the existing loop nest
    // as the pipeline loops instead of re-tiling.
    bool inside_reduction_nest =
        reduction_loop_ &&
        reduction_loop_->isAncestor(linalg_op);
    if (inside_reduction_nest) {
      // For the matmul, tile_sizes_ and total_num_elements_ govern the FIFO
      // slot size. The accumulator output is fp32 (the SIMD feature only
      // declares fp16 lanes, so determineTileSizes gives 1 for fp32 → wrong).
      // Use the fp16 SIMD vector length (one stick) as the transfer granularity
      // instead — the input loads (A/W) are fp16 sticks, and the pipeline
      // must transfer 64 elements (one stick) per time step.
      auto simd_feature =
          resource_kinds_->getFeature<mlir::ktdf_arch::feature::SIMD>(
              resource_kinds_->getComputeKind());
      auto f16_type = mlir::Float16Type::get(&getContext());
      const int64_t vector_length =
          std::max(simd_feature.getLanes(f16_type), int64_t(1));
      tile_sizes_ = {vector_length};
      total_num_elements_ = vector_length;

      // Collect the enclosing scf.for loops, outermost first.
      llvm::SmallVector<mlir::Operation*> enclosing;
      for (mlir::Operation* p = linalg_op->getParentOfType<mlir::scf::ForOp>();
           p; p = p->getParentOfType<mlir::scf::ForOp>()) {
        enclosing.push_back(p);
      }
      std::reverse(enclosing.begin(), enclosing.end());
      tiled_loops_.assign(enclosing.begin(), enclosing.end());
      continue;
    }

    // Determine tile sizes from output operand shape (needed for loop
    // creation)
    tile_sizes_ = determineTileSizes(linalg_op);

    if (tile_sizes_.empty()) {
      linalg_op.emitError("Could not determine tile sizes");
      signalPassFailure();
      return;
    }

    // Calculate total number of elements (product of tile_sizes_)
    total_num_elements_ = 1;
    for (int64_t dim : tile_sizes_) {
      total_num_elements_ *= dim;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Tile sizes: ";
      for (int64_t i : tile_sizes_) llvm::dbgs() << i << ", ";
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  Total num elements: " << total_num_elements_ << "\n";
    });

    // Tile the linalg.generic operation with the computed tile sizes
    // Use custom tiling that doesn't create iter_args.
    mlir::linalg::LinalgTilingOptions tiling_options;
    tiling_options.setTileSizes(tile_sizes_);
    tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);

    llvm::FailureOr<mlir::linalg::TiledLinalgOp> tiled_result =
        customTileLinalgOp(rewriter, linalg_op, tiling_options);

    if (mlir::failed(tiled_result)) {
      linalg_op.emitError("Failed to tile linalg operation");
      signalPassFailure();
      return;
    }

    // Replace uses of original results with tiled results
    assert(tiled_result->tensorResults.size() == linalg_op->getNumResults() &&
           "Tiled result count must match original result count");
    for (size_t i = 0; i < linalg_op->getNumResults(); ++i) {
      rewriter.replaceAllUsesWith(linalg_op->getResult(i),
                                  tiled_result->tensorResults[i]);
    }

    // Erase the original linalg operation. It has been replaced by a loop nest
    // containing the tiled linalg operation.
    rewriter.eraseOp(linalg_op);

    // Annotate loops with loop_type attributes based on iterator types
    if (!tiled_result->loops.empty()) {
      auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(
          tiled_result->op.getOperation());
      annotateLoopsWithIteratorTypes(tiled_result->loops, generic_op);
    }

    LLVM_DEBUG(llvm::dbgs() << "  Successfully tiled linalg operation\n");

    // Store the tiled loops for later pipeline creation
    if (!tiled_result->loops.empty()) {
      tiled_loops_.assign(tiled_result->loops.begin(),
                          tiled_result->loops.end());
    }
  }
}

mlir::ktdf::PrivateOp ConstructThreeStagePipelinePass::createPrivateOp(
    mlir::OpBuilder& builder, mlir::Location loc) {
  mlir::ktdf::TokenType token_type = mlir::ktdf::TokenType::get(&getContext());
  llvm::SmallVector<mlir::Type> private_result_types;

  // Group FIFO slot types by their FIFO type to preserve ordering
  llvm::MapVector<std::pair<mlir::Attribute, mlir::Attribute>,
                  llvm::SmallVector<mlir::ktdf::FifoSlotType>>
      fifo_type_groups;

  // Add FIFO slot types for load operations
  for (mlir::ktdp::LoadOp load_op : load_ops_) {
    auto tensor_type =
        mlir::dyn_cast<mlir::RankedTensorType>(load_op.getResult().getType());
    if (!tensor_type) {
      load_op.emitError("Expected RankedTensorType result from ktdp.load");
      signalPassFailure();
      return nullptr;
    }
    mlir::Type element_type = tensor_type.getElementType();

    // Get FIFO attributes for this specific load operation
    auto [load_src, load_dest] = getFifoAttributesForLoad(load_op);
    auto load_key = std::make_pair(load_src, load_dest);

    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), load_src, load_dest, total_num_elements_, element_type);
    fifo_type_groups[load_key].push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add FIFO slot types for store operations
  for (mlir::ktdp::StoreOp store_op : store_ops_) {
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(
        store_op.getDataTile().getType());
    if (!tensor_type) {
      store_op.emitError(
          "Expected RankedTensorType data operand in ktdp.store");
      signalPassFailure();
      return nullptr;
    }
    mlir::Type element_type = tensor_type.getElementType();

    // Get FIFO attributes for this specific store operation
    auto [store_src, store_dest] = getFifoAttributesForStore(store_op);
    auto store_key = std::make_pair(store_src, store_dest);

    auto fifo_slot_type =
        mlir::ktdf::FifoSlotType::get(&getContext(), store_src, store_dest,
                                      total_num_elements_, element_type);
    fifo_type_groups[store_key].push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add token types: 3 for load+compute+store pipeline, 2 when there is no
  // store stage (reduction K-loop: accumulator stays in LRF, no store FIFO).
  int num_token_types = store_ops_.empty() ? 2 : 3;
  for (int i = 0; i < num_token_types; ++i) {
    private_result_types.push_back(token_type);
  }

  // Create ktdf.private operation
  auto private_op =
      mlir::ktdf::PrivateOp::create(builder, loc, private_result_types);
  mlir::Region& private_region = private_op.getRegion();
  mlir::Block* private_body = &private_region.front();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(private_body);

  // Create ktdf.fifo.allocate operations grouped by FIFO type
  // This preserves the order of load/stores within each FIFO type
  llvm::SmallVector<mlir::Value> fifo_results;
  for (auto& [fifo_type, slot_types] : fifo_type_groups) {
    // Create a single fifo.allocate with multiple results for this FIFO type
    llvm::SmallVector<mlir::Type> result_types(slot_types.begin(),
                                               slot_types.end());
    auto fifo_alloc_op = mlir::ktdf::FifoAllocateOp::create(
        builder, loc, mlir::TypeRange{result_types}, mlir::ValueRange{});

    // Add all results from this allocation to fifo_results
    for (unsigned i = 0; i < fifo_alloc_op.getNumResults(); ++i) {
      fifo_results.push_back(fifo_alloc_op.getResult(i));
    }
  }

  // Create tokens: one per stage boundary. With a store stage we need 3 tokens
  // (load→compute, compute→store, store→done); without a store stage (matmul
  // accumulates into LRF, no pipeline output FIFO) we need only 2 tokens.
  int num_tokens = store_ops_.empty() ? 2 : 3;
  for (int i = 0; i < num_tokens; ++i) {
    fifo_results.push_back(
        mlir::ktdf::CreateTokenOp::create(builder, loc, token_type).getResult());
  }

  // Yield all results
  mlir::ktdf::PrivateYieldOp::create(builder, loc, fifo_results);

  return private_op;
}

void ConstructThreeStagePipelinePass::createPipeline(
    mlir::scf::ForOp innermost_loop) {
  LLVM_DEBUG(llvm::dbgs() << "Creating ktdf.pipeline with three stages\n");

  compute_ops_.clear();
  innermost_loop.getBody()->walk([&](mlir::linalg::LinalgOp linalg_op) {
    compute_ops_.push_back(linalg_op);
  });

  LLVM_DEBUG(llvm::dbgs() << "  Found " << compute_ops_.size()
                          << " linalg compute operations\n");

  // Save original scf.yield operands to ops_to_delete_ before changing the
  // yield. This ensures tensor.insert_slice and linalg.generic get cleaned up
  // later.
  auto yield_op =
      mlir::cast<mlir::scf::YieldOp>(innermost_loop.getBody()->getTerminator());
  for (mlir::Value operand : yield_op.getOperands()) {
    if (auto* def_op = operand.getDefiningOp()) {
      ops_to_delete_.push_back(def_op);
    }
  }
  // For the reduction case the yield has no operands (iter_arg was dropped),
  // so the linalg op would not be added above. Explicitly mark all compute
  // ops for deletion so their use-chains (ktdp.load → tensor.extract_slice →
  // linalg.generic) get cleaned up by cleanupOperations.
  for (mlir::linalg::LinalgOp lop : compute_ops_) {
    ops_to_delete_.push_back(lop.getOperation());
  }

  // Fix scf.yield to yield iter_args instead of linalg results. This allows
  // tensor.insert_slice to get cleaned up (no more uses) and the dead iter args
  // will be cleaned by a dead code elimination pass.
  mlir::OpBuilder yield_builder(yield_op);
  llvm::SmallVector<mlir::Value> new_yield_operands;
  for (mlir::BlockArgument iter_arg : innermost_loop.getRegionIterArgs()) {
    new_yield_operands.push_back(iter_arg);
  }
  mlir::scf::YieldOp::create(yield_builder, yield_op.getLoc(),
                             new_yield_operands);
  yield_op.erase();

  // Create ktdf.pipeline operation at the start of the loop body.
  mlir::OpBuilder builder(innermost_loop.getBodyRegion());

  auto compute =
      resource_kinds_->getResource(resource_kinds_->getComputeKind());
  auto incoming = mlir::ktdf_arch::getLink(
      mlir::ktdf_arch::LinkDirection::Incoming, compute);
  auto outgoing = mlir::ktdf_arch::getLink(
      mlir::ktdf_arch::LinkDirection::Outgoing, compute);
  assert(incoming && outgoing);

  // Query the interface type from memory to the compute unit
  if (!incoming.getFeature<mlir::ktdf_arch::feature::Queue>().isOrdered() ||
      !outgoing.getFeature<mlir::ktdf_arch::feature::Queue>().isOrdered()) {
    signalPassFailure();
    return;
  }

  mlir::ktdf::PipelineOp::create(
      builder, innermost_loop.getLoc(),
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        // Create ktdf.private operation with FIFO slots and tokens
        auto private_op = createPrivateOp(builder, loc);

        // Tokens are at the end: fifo_count + 0, fifo_count + 1, fifo_count + 2
        size_t fifo_count = load_ops_.size() + store_ops_.size();

        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{},
            /*depends_out=*/{private_op.getResult(fifo_count + 0U)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Add data transfer operations in stage1 for loads
              createDataTransfers(builder, loc, private_op, tile_sizes_,
                                  /*is_load=*/true,
                                  /*private_result_offset=*/0);
            });

        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{private_op.getResult(fifo_count + 0U)},
            /*depends_out=*/{private_op.getResult(fifo_count + 1U)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Create read_from_fifos, compute operations, write_to_fifos in
              // stage 2
              createComputeOps(builder, loc, private_op);
            })
            .setApplicableUnitsAttr(
                builder.getArrayAttr(resource_kinds_->getComputeKind()));

        // Only emit a store stage if there are store operations to pipeline
        // (the reduction's output-store is outside the reduction loop pipeline;
        // skipping the empty store stage prevents pass-12 from failing on a
        // stage with no applicable_units).
        if (!store_ops_.empty()) {
          mlir::ktdf::StageOp::create(
              builder, loc,
              /*depends_in=*/{private_op.getResult(fifo_count + 1U)},
              /*depends_out=*/{private_op.getResult(fifo_count + 2U)},
              [&](mlir::OpBuilder& builder, mlir::Location loc) {
                createDataTransfers(builder, loc, private_op, tile_sizes_,
                                    /*is_load=*/false,
                                    /*private_result_offset=*/load_ops_.size());
              });
        }
      });
}

void ConstructThreeStagePipelinePass::deleteOpAndUnusedChainOfOperands(
    mlir::Operation* op) {
  // Collect operands before erasing
  llvm::SmallVector<mlir::Value> operands(op->getOperands().begin(),
                                          op->getOperands().end());
  op->erase();

  // Recursively delete unused operand operations
  for (mlir::Value operand : operands) {
    if (auto* def_op = operand.getDefiningOp()) {
      if (def_op->use_empty()) deleteOpAndUnusedChainOfOperands(def_op);
    }
  }
}

void ConstructThreeStagePipelinePass::cleanupOperations() {
  LLVM_DEBUG(llvm::dbgs() << "Cleaning up operations\n");

  for (auto* op : ops_to_delete_) {
    if (!op || !op->use_empty()) continue;
    deleteOpAndUnusedChainOfOperands(op);
  }
}

// Find the linalg operand fed by `tensor_value` (directly or through a
// tensor.extract_slice) and return its matching indexing_map. Returns
// std::nullopt if no such operand is found.
static std::optional<mlir::AffineMap> findIndexingMapForLoadResult(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value) {
  // Helper to find indexing map for a value used by linalg_op
  auto findMapForValue = [&](mlir::Value v) -> std::optional<mlir::AffineMap> {
    for (mlir::OpOperand& operand : linalg_op->getOpOperands()) {
      if (operand.get() == v) return linalg_op.getMatchingIndexingMap(&operand);
    }
    return std::nullopt;
  };

  // Check direct use
  if (auto map = findMapForValue(tensor_value)) return map;

  // Check through extract_slice
  for (mlir::Operation* user : tensor_value.getUsers()) {
    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
      if (auto map = findMapForValue(extract.getResult())) return map;
    }
  }

  return std::nullopt;
}

// For a store, find the linalg DPS init operand whose value flows into the
// store's tensor input (directly or through a tensor.insert_slice) and return
// its matching indexing_map. Returns std::nullopt if not found.
static std::optional<mlir::AffineMap> findIndexingMapForStoreSource(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value) {
  mlir::Value v = tensor_value;
  if (auto insert = v.getDefiningOp<mlir::tensor::InsertSliceOp>())
    v = insert.getSource();
  for (int64_t i = 0, e = linalg_op->getNumResults(); i < e; ++i) {
    if (linalg_op->getResult(i) == v) {
      mlir::OpOperand* init = linalg_op.getDpsInitOperand(i);
      return linalg_op.getMatchingIndexingMap(init);
    }
  }
  return std::nullopt;
}

// Project tile_sizes through indexing_map to produce one entry per result
// dim of the map. Sizes are derived per result expr:
//   - AffineDimExpr  d_k  -> tile_sizes[k]   (projection of one iterator dim)
//   - AffineConstantExpr -> 1                (broadcast/squeeze: that source
//                                             dim is accessed at a fixed index)
// Other expressions (e.g. d0 + d1) are not currently supported and produce
// an error.
static mlir::LogicalResult projectSizesThroughIndexingMap(
    mlir::AffineMap indexing_map, llvm::ArrayRef<int64_t> tile_sizes,
    llvm::SmallVectorImpl<int64_t>& projected_sizes,
    mlir::function_ref<mlir::InFlightDiagnostic()> emit_error) {
  assert(indexing_map.getNumSymbols() == 0 &&
         "linalg indexing maps are expected to have no symbols");
  assert(indexing_map.getNumDims() == tile_sizes.size() &&
         "indexing map dims must match the iteration-space rank");

  for (mlir::AffineExpr result : indexing_map.getResults()) {
    if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(result)) {
      projected_sizes.push_back(tile_sizes[dim.getPosition()]);
      continue;
    }
    if (mlir::isa<mlir::AffineConstantExpr>(result)) {
      projected_sizes.push_back(1);
      continue;
    }
    std::string buf;
    llvm::raw_string_ostream os(buf);
    result.print(os);
    emit_error()
        << "operand indexing map has an unsupported result expression (" << buf
        << "); only single-dim projections and constant indices are "
           "supported";
    return mlir::failure();
  }
  return mlir::success();
}

// Build the source_map for a data_transfer from an access tile op.
//
// Model A (HBM, global memory): 1-D flat-offset map.
//   source_map has numDims == loop_ivs.size(), numResults == 1.
//   Result = sum_i(stride_i * iv_expr(i)).
//   Example: a[%n,  %m, 0] strides [16384,64,1] -> (d0,d1) -> (d0*16384+d1*64)
//
// Model B (LX/per-core, lrfreg): all-zeros N-D map (offset already baked
//   into the reinterpret_cast base; subscript is constant 0 per dim).
//   source_map has numDims == loop_ivs.size(), numResults == memref_rank.
//   Example: lx[0,0,0] -> (d0,d1) -> (0,0,0)
//
// `is_model_a` = true → model A (HBM); false → model B (LX/lrfreg).
static std::optional<mlir::AffineMap> buildSourceMapFromAccessTile(
    mlir::ktdp::ConstructAccessTilesOp access_tile_op,
    llvm::ArrayRef<mlir::Value> loop_ivs,
    mlir::MLIRContext* ctx,
    bool is_model_a) {
  mlir::AffineMap base_map = access_tile_op.getBaseMap();
  llvm::SmallVector<mlir::Value> raw_indices = access_tile_op.getIndices();
  unsigned memref_rank = base_map.getNumResults();

  // Build a quick lookup: loop IV value -> position in loop_ivs.
  llvm::DenseMap<mlir::Value, unsigned> iv_to_pos;
  for (unsigned k = 0; k < loop_ivs.size(); ++k)
    iv_to_pos[loop_ivs[k]] = k;

  if (!is_model_a) {
    // Model B: all-zeros map (offset baked into the reinterpret_cast offset).
    llvm::SmallVector<mlir::AffineExpr> zeros(
        memref_rank, mlir::getAffineConstantExpr(0, ctx));
    return mlir::AffineMap::get(loop_ivs.size(), 0, zeros, ctx);
  }

  // Model A: compute flat offset from strides and loop IVs.
  llvm::SmallVector<int64_t> strides(memref_rank, 1);
  {
    mlir::Value mem_view = access_tile_op.getBase();
    if (auto cmv = mlir::dyn_cast_or_null<mlir::ktdp::ConstructMemoryViewOp>(
            mem_view.getDefiningOp())) {
      auto sa = cmv->getAttrOfType<mlir::DenseI64ArrayAttr>("static_strides");
      if (sa && sa.size() == (int64_t)memref_rank)
        strides.assign(sa.asArrayRef().begin(), sa.asArrayRef().end());
    }
  }

  mlir::AffineExpr offset_expr = mlir::getAffineConstantExpr(0, ctx);
  for (unsigned i = 0; i < memref_rank; ++i) {
    mlir::AffineExpr result_expr = base_map.getResults()[i];
    mlir::AffineExpr iv_expr;

    if (auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(result_expr)) {
      mlir::Value idx = raw_indices[dim_expr.getPosition()];
      auto it = iv_to_pos.find(idx);
      if (it != iv_to_pos.end()) {
        iv_expr = mlir::getAffineDimExpr(it->second, ctx);
      } else {
        if (idx.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
          iv_expr = mlir::getAffineConstantExpr(0, ctx);
        } else {
          return std::nullopt;
        }
      }
    } else if (mlir::isa<mlir::AffineConstantExpr>(result_expr)) {
      iv_expr = mlir::getAffineConstantExpr(0, ctx);
    } else {
      return std::nullopt;
    }

    if (strides[i] != 0)
      offset_expr = offset_expr +
                    iv_expr * mlir::getAffineConstantExpr(strides[i], ctx);
  }

  return mlir::AffineMap::get(loop_ivs.size(), 0,
                              llvm::ArrayRef<mlir::AffineExpr>{offset_expr},
                              ctx);
}

void ConstructThreeStagePipelinePass::createDataTransfers(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf::PrivateOp private_op, llvm::ArrayRef<int64_t> tile_sizes,
    bool is_load, size_t private_result_offset) {
  // Collect loop induction variables from tiled loops
  llvm::SmallVector<mlir::Value> loop_ivs;
  for (auto* loop_op : tiled_loops_) {
    auto for_op = llvm::cast<mlir::scf::ForOp>(loop_op);
    loop_ivs.push_back(for_op.getInductionVar());
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Creating " << (is_load ? "load" : "store")
                 << " data transfers with " << loop_ivs.size() << " loop IVs\n";
  });

  // Get the appropriate operation list
  size_t op_count = is_load ? load_ops_.size() : store_ops_.size();

  // FIFO slot size is the product of tile_sizes (i.e. total_num_elements_)
  llvm::SmallVector<int64_t> fifo_sizes = {total_num_elements_};

  const auto compute_kind = resource_kinds_->getComputeKind();
  if (!compute_kind) {
    signalPassFailure();
    return;
  }
  auto compute = resource_kinds_->getResource(compute_kind);
  auto incoming = mlir::ktdf_arch::getLink(
      mlir::ktdf_arch::LinkDirection::Incoming, compute);
  auto outgoing = mlir::ktdf_arch::getLink(
      mlir::ktdf_arch::LinkDirection::Outgoing, compute);
  if (!incoming || !outgoing) {
    signalPassFailure();
    return;
  }
  const auto granularity_in =
      incoming.getProperty<mlir::ktdf_arch::TransferGranularityAttr>();
  const auto granularity_out =
      outgoing.getProperty<mlir::ktdf_arch::TransferGranularityAttr>();
  if (!granularity_in || !granularity_out) {
    signalPassFailure();
    return;
  }
  const auto max_in = maxOrDefault(granularity_in.asArrayRef());
  const auto max_out = maxOrDefault(granularity_out.asArrayRef());
  if (max_in != max_out) {
    signalPassFailure();
    return;
  }

  assert(total_num_elements_ <= max_in &&
         "FIFO slot size exceeds maximum allowed size for compute unit");

  // The post-fusion linalg op whose indexing_maps describe how each
  // load/store operand maps onto the iteration space.
  assert(compute_ops_.size() == 1 &&
         "expected exactly one linalg compute op after fusion");
  mlir::linalg::LinalgOp linalg_op = compute_ops_[0];

  // Create data_transfer for each operation
  for (size_t i = 0; i < op_count; ++i) {
    // Get the access_tile value (operand of load/store op).
    mlir::Value access_tile_value;
    mlir::Operation* err_anchor;
    if (is_load) {
      access_tile_value = load_ops_[i].getAccessTile();
      err_anchor = load_ops_[i].getOperation();
    } else {
      access_tile_value = store_ops_[i].getAccessTile();
      err_anchor = store_ops_[i].getOperation();
    }

    // Detect model-A (HBM) vs model-B (LX/lrfreg) for this access tile.
    bool is_model_a_tile = false;
    if (auto access_tile_op =
            access_tile_value
                .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
      mlir::Value mem_view_base = access_tile_op.getBase();
      if (auto cmv = mlir::dyn_cast_or_null<mlir::ktdp::ConstructMemoryViewOp>(
              mem_view_base.getDefiningOp())) {
        mlir::Attribute mapped_ms = mapMemorySpace(cmv.getMemorySpaceAttr());
        if (auto ms_str = mlir::dyn_cast<mlir::StringAttr>(mapped_ms)) {
          llvm::StringRef ms = ms_str.getValue();
          is_model_a_tile = (ms != "LX" && ms != "lrfreg");
        }
      }
    }

    // Build source_map from the access-tile's index structure.
    std::optional<mlir::AffineMap> source_map;
    if (auto access_tile_op =
            access_tile_value
                .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
      source_map = buildSourceMapFromAccessTile(access_tile_op, loop_ivs,
                                                &getContext(), is_model_a_tile);
    }

    // Fall back to linalg indexing map if access-tile approach fails.
    if (!source_map) {
      if (is_load) {
        source_map = findIndexingMapForLoadResult(linalg_op,
                                                  load_ops_[i].getResult());
      } else {
        source_map = findIndexingMapForStoreSource(linalg_op,
                                                   store_ops_[i].getDataTile());
      }
    }

    if (!source_map) {
      err_anchor->emitError(
          "could not build source_map for data_transfer; the access-tile "
          "index structure is not supported and no linalg indexing map "
          "fallback was found");
      signalPassFailure();
      return;
    }

    // Derive the access-tile sizes for the data_transfer's static sizes.
    // Model-A (HBM): 1-D stick → {lane}. Model-B (LX): keep N-D tile shape.
    llvm::SmallVector<int64_t> access_tile_sizes;
    bool got_sizes_from_tile = false;
    if (auto access_tile_op =
            access_tile_value
                .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
      auto at_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
          access_tile_op.getResult().getType());
      if (at_type) {
        if (is_model_a_tile && source_map->getNumResults() == 1) {
          // Model A: 1-D view → just the lane count (last dim of tile).
          access_tile_sizes = {at_type.getShape().back()};
        } else {
          // Model B or LX: keep the full tile shape.
          access_tile_sizes.assign(at_type.getShape().begin(),
                                   at_type.getShape().end());
        }
        got_sizes_from_tile = true;
      }
    }
    if (!got_sizes_from_tile) {
      // Fallback: project tile_sizes through source_map (works for identity maps).
      for (mlir::AffineExpr res_expr : source_map->getResults()) {
        if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(res_expr)) {
          unsigned pos = dim.getPosition();
          int64_t sz = (pos < tile_sizes.size()) ? tile_sizes[pos]
                                                  : tile_sizes.back();
          access_tile_sizes.push_back(sz);
        } else {
          access_tile_sizes.push_back(1);
        }
      }
    }

    // Get the FIFO slot from ktdf.private results.
    mlir::Value fifo_slot = private_op.getResult(private_result_offset + i);

    // The fifo side gets a null AffineMap; the access-tile (memref) side
    // gets the source_map built above. loop_ivs feeds the map's dim inputs.
    mlir::AffineMap null_map;
    if (is_load) {
      mlir::ktdf::DataTransferOp::create(builder, loc, access_tile_value,
                                         *source_map, loop_ivs,
                                         access_tile_sizes, fifo_slot, null_map,
                                         mlir::ValueRange{}, fifo_sizes);
    } else {
      mlir::ktdf::DataTransferOp::create(
          builder, loc, fifo_slot, null_map, mlir::ValueRange{}, fifo_sizes,
          access_tile_value, *source_map, loop_ivs, access_tile_sizes);
    }
  }
}

llvm::SmallVector<int64_t>
ConstructThreeStagePipelinePass::getStridesFromMemRefType(
    mlir::MemRefType memref_type) {
  llvm::SmallVector<int64_t> strides;
  if (auto strided_layout =
          mlir::dyn_cast<mlir::StridedLayoutAttr>(memref_type.getLayout())) {
    strides.assign(strided_layout.getStrides().begin(),
                   strided_layout.getStrides().end());
  } else {
    // Compute default row-major strides
    int64_t stride = 1;
    for (int i = memref_type.getRank() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      stride *= memref_type.getShape()[i];
    }
  }
  return strides;
}

void ConstructThreeStagePipelinePass::createPostLoopStorePipeline(
    mlir::ktdp::StoreOp store_op, mlir::scf::ForOp enclosing_n_loop) {
  // Gap 5: emit a ktdf.pipeline after the reduction loop for the post-loop
  // ktdp.store.  The pipeline has a compute stage (write_to_fifo of the
  // accumulated tensor) and a store stage (data_transfer from the FIFO slot to
  // the HBM output tile).
  //
  // The access tile for the store is left as-is here; it will be replaced by
  // memref.reinterpret_cast when replaceAccessTilesWithReinterpretCast runs
  // (after all pipelines have been created), consistent with how the load
  // pipelines are handled.

  mlir::MLIRContext* ctx = &getContext();
  mlir::Location loc = store_op.getLoc();

  // The tensor being stored and the access tile receiving it.
  mlir::Value data_tensor = store_op.getDataTile();  // tensor<1x64xf16>
  mlir::Value access_tile_value = store_op.getAccessTile();

  auto tensor_type =
      mlir::dyn_cast<mlir::RankedTensorType>(data_tensor.getType());
  if (!tensor_type) {
    store_op.emitError("post-loop store data is not a ranked tensor");
    signalPassFailure();
    return;
  }

  // Determine FIFO slot type: compute unit → memory (same direction as store).
  auto [store_src_attr, store_dest_attr] = getFifoAttributesForStore(store_op);
  mlir::Type elem_type = tensor_type.getElementType();
  // FIFO element count = product of tensor shape (e.g. 1*64 = 64 elements).
  int64_t fifo_elems = 1;
  for (int64_t d : tensor_type.getShape()) fifo_elems *= d;
  auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
      ctx, store_src_attr, store_dest_attr, fifo_elems, elem_type);
  auto token_type = mlir::ktdf::TokenType::get(ctx);

  // Build the source_map for the data_transfer: maps the enclosing loop IVs
  // to the access tile's coordinates. Collect the loop IVs from the outer
  // loops of the post-loop store (i.e. the N-stick loop).
  llvm::SmallVector<mlir::Value> outer_loop_ivs;
  for (mlir::Operation* p = store_op->getParentOfType<mlir::scf::ForOp>(); p;
       p = p->getParentOfType<mlir::scf::ForOp>()) {
    outer_loop_ivs.push_back(
        mlir::cast<mlir::scf::ForOp>(p).getInductionVar());
  }
  std::reverse(outer_loop_ivs.begin(), outer_loop_ivs.end());

  // Detect model-A for this store tile (HBM → model-A).
  bool is_post_loop_model_a = false;
  if (auto access_tile_op =
          access_tile_value
              .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
    mlir::Value mem_view = access_tile_op.getBase();
    if (auto cmv = mlir::dyn_cast_or_null<mlir::ktdp::ConstructMemoryViewOp>(
            mem_view.getDefiningOp())) {
      mlir::Attribute mapped_ms = mapMemorySpace(cmv.getMemorySpaceAttr());
      if (auto ms_str = mlir::dyn_cast<mlir::StringAttr>(mapped_ms)) {
        llvm::StringRef ms = ms_str.getValue();
        is_post_loop_model_a = (ms != "LX" && ms != "lrfreg");
      }
    }
  }

  // Build the source_map from the access tile's index structure.
  std::optional<mlir::AffineMap> source_map;
  if (auto access_tile_op =
          access_tile_value
              .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
    source_map = buildSourceMapFromAccessTile(access_tile_op, outer_loop_ivs,
                                              ctx, is_post_loop_model_a);
  }
  if (!source_map) {
    // Fallback: constant 0 map (all dims at offset 0 within the tile).
    unsigned memref_rank =
        mlir::dyn_cast<mlir::MemRefType>(
            mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
                access_tile_value.getDefiningOp())
                ->getResult(0)
                .getType())
            .getRank();
    llvm::SmallVector<mlir::AffineExpr> zeros(
        memref_rank, mlir::getAffineConstantExpr(0, ctx));
    source_map = mlir::AffineMap::get(outer_loop_ivs.size(), 0, zeros, ctx);
  }

  // Derive access tile sizes from the access tile type shape.
  // For model-A (1-result source_map), the dest is a 1-D stick; use just
  // the lane count (last dim).
  llvm::SmallVector<int64_t> access_tile_sizes;
  if (auto access_tile_op =
          access_tile_value
              .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
    auto at_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
        access_tile_op.getResult().getType());
    if (at_type) {
      if (source_map && source_map->getNumResults() == 1) {
        // Model A: 1-D dest — just the lane count.
        access_tile_sizes = {at_type.getShape().back()};
      } else {
        access_tile_sizes.assign(at_type.getShape().begin(),
                                 at_type.getShape().end());
      }
    }
  }
  if (access_tile_sizes.empty()) {
    for (int64_t d : tensor_type.getShape())
      access_tile_sizes.push_back(d);
  }
  llvm::SmallVector<int64_t> fifo_sizes = {fifo_elems};

  // Insert the post-loop pipeline right before the store_op.
  mlir::OpBuilder builder(store_op);

  mlir::ktdf::PipelineOp::create(
      builder, loc,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        // ktdf.private: 1 FIFO slot + 2 tokens (compute → store).
        auto private_op = mlir::ktdf::PrivateOp::create(
            builder, loc,
            mlir::TypeRange{fifo_slot_type, token_type, token_type});
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          mlir::Block* pb = &private_op.getRegion().front();
          builder.setInsertionPointToStart(pb);
          auto fifo_alloc = mlir::ktdf::FifoAllocateOp::create(
              builder, loc, mlir::TypeRange{fifo_slot_type},
              mlir::ValueRange{});
          auto tok0 = mlir::ktdf::CreateTokenOp::create(
              builder, loc, token_type);
          auto tok1 = mlir::ktdf::CreateTokenOp::create(
              builder, loc, token_type);
          mlir::ktdf::PrivateYieldOp::create(
              builder, loc,
              mlir::ValueRange{fifo_alloc.getResult(0), tok0.getResult(),
                               tok1.getResult()});
        }
        // Indices: private_op result 0 = FIFO slot, 1 = tok0, 2 = tok1.
        mlir::Value fifo_slot = private_op.getResult(0);
        mlir::Value tok0 = private_op.getResult(1);
        mlir::Value tok1 = private_op.getResult(2);

        // Stage 1 (compute): write the accumulated tensor to the FIFO.
        // This stage acts as the "compute" stage that makes the result
        // available to the store unit via the FIFO queue.
        mlir::ktdf::StageOp::create(
            builder, loc, /*depends_in=*/{}, /*depends_out=*/{tok0},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              mlir::ktdf::WriteToFifoOp::create(builder, loc, data_tensor,
                                                fifo_slot);
            })
            .setApplicableUnitsAttr(
                builder.getArrayAttr(resource_kinds_->getComputeKind()));

        // Stage 2 (store): data_transfer from FIFO to the output tile.
        mlir::ktdf::StageOp::create(
            builder, loc, /*depends_in=*/{tok0}, /*depends_out=*/{tok1},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              mlir::AffineMap null_map;
              mlir::ktdf::DataTransferOp::create(
                  builder, loc, fifo_slot, null_map, mlir::ValueRange{},
                  fifo_sizes, access_tile_value, *source_map, outer_loop_ivs,
                  access_tile_sizes);
            });
      });

  // Mark the original store op for deletion.
  ops_to_delete_.push_back(store_op.getOperation());
}

}  // namespace

void ConstructThreeStagePipelinePass::createComputeOps(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf::PrivateOp private_op) {
  LLVM_DEBUG(llvm::dbgs() << "Creating compute operations in stage 2\n");

  if (compute_ops_.size() != 1) {
    mlir::emitError(loc, "Expected exactly one compute op after fusion, got ")
        << compute_ops_.size();
    signalPassFailure();
    return;
  }

  // Get the tiled tensor type from the compute operation
  auto tiled_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(
      compute_ops_[0]->getResult(0).getType());
  if (!tiled_tensor_type) {
    compute_ops_[0]->emitError(
        "Expected RankedTensorType result from linalg operation");
    signalPassFailure();
    return;
  }

  // Find the post-tiling extract_slice that feeds each load result into the
  // linalg compute op. Its result type is the per-operand tile shape (which
  // may have lower rank than the linalg result when the operand's indexing
  // map drops iteration-space dims, e.g. a broadcast input).
  mlir::DenseMap<mlir::Value, mlir::tensor::ExtractSliceOp> load_to_extract;
  for (mlir::ktdp::LoadOp load_op : load_ops_) {
    for (mlir::Operation* user : load_op.getResult().getUsers()) {
      if (auto extract_op =
              mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
        load_to_extract[load_op.getResult()] = extract_op;
        break;
      }
    }
  }

  // Build a mapping from load results to read_from_fifo results, using the
  // per-operand tile type so the cloned linalg op's input ranks match its
  // indexing maps.
  mlir::DenseMap<mlir::Value, mlir::Value> value_map;
  mlir::IRMapping mapper;
  for (size_t i = 0; i < load_ops_.size(); ++i) {
    mlir::ktdp::LoadOp load_op = load_ops_[i];
    mlir::Value fifo_slot = private_op.getResult(i);

    auto extract_it = load_to_extract.find(load_op.getResult());
    if (extract_it == load_to_extract.end()) {
      load_op.emitError(
          "no tensor.extract_slice user found; cannot determine the "
          "per-operand "
          "tile type for ktdf.read_from_fifo");
      signalPassFailure();
      return;
    }
    mlir::Type per_operand_tile_type = extract_it->second.getResult().getType();

    auto read_fifo_op = mlir::ktdf::ReadFromFifoOp::create(
        builder, loc, per_operand_tile_type, fifo_slot);
    value_map[load_op.getResult()] = read_fifo_op.getResult();
    mapper.map(extract_it->second.getResult(), read_fifo_op.getResult());
  }

  // Clone compute operation into stage 2 and create write_to_fifo for each
  // store
  mlir::linalg::LinalgOp compute_op = compute_ops_[0];
  LLVM_DEBUG(llvm::dbgs() << "  Cloning compute op: " << compute_op->getName()
                          << "\n");

  // Map the DPS output operand.
  //
  // Elementwise case (no lrf_acc_buf_): map DPS init → tensor.empty so the
  // compute stage has a fresh output buffer.
  //
  // Reduction case (lrf_acc_buf_ set): convert the LRF alloca buffer to a
  // tensor (bufferization.to_tensor) and use it as the DPS init
  // (read-modify-write: read current acc, compute new value, write back).
  auto linalg_op =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(compute_op.getOperation());
  if (linalg_op && linalg_op.getNumDpsInits() > 0) {
    mlir::Value output_operand = linalg_op.getDpsInitOperand(0)->get();

    if (lrf_acc_buf_) {
      // Reduction: read the current accumulator value from the LRF alloca.
      auto lrf_tensor = mlir::bufferization::ToTensorOp::create(
          builder, loc, tiled_tensor_type, lrf_acc_buf_,
          /*restrict=*/true, /*writable=*/false);
      mapper.map(output_operand, lrf_tensor.getResult());
    } else {
      // Elementwise: no accumulator; use a fresh empty tensor.
      auto empty_tensor = mlir::tensor::EmptyOp::create(
          builder, loc, tiled_tensor_type.getShape(),
          tiled_tensor_type.getElementType());
      mapper.map(output_operand, empty_tensor.getResult());
    }
  }

  auto* cloned = builder.clone(*compute_op, mapper);

  // For the reduction case: write the updated accumulator back to the LRF
  // alloca so the next iteration reads the updated value.
  if (lrf_acc_buf_) {
    mlir::bufferization::MaterializeInDestinationOp::create(
        builder, loc, /*result=*/mlir::Type{}, cloned->getResult(0),
        lrf_acc_buf_, /*restrict=*/false, /*writable=*/true);
  }

  // Create ktdf.write_to_fifo for each store operation
  for (size_t i = 0; i < store_ops_.size(); ++i) {
    mlir::Value fifo_slot = private_op.getResult(load_ops_.size() + i);
    mlir::ktdf::WriteToFifoOp::create(builder, loc, cloned->getResult(0),
                                      fifo_slot);
  }
}

mlir::Value ConstructThreeStagePipelinePass::computeReinterpretCastOffset(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::SmallVector<mlir::Value>& indices,
    llvm::SmallVector<int64_t>& strides) {
  // Calculate offset from access tile indices and memory view strides.
  // For an access tile %A_view[%idx0, %idx1, ...] with strides [stride0,
  // stride1, ...], the offset is: %idx0 * stride0 + %idx1 * stride1 + ...
  // This is computed using a sequence of arith.muli and arith.addi operations.

  size_t num_indices = indices.size();

  if (num_indices == 0) {
    // No indices means offset is 0
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  }

  // Compute terms: indices[i] * strides[i] for each dimension
  // Optimizations:
  // - Skip multiplication if stride is 1
  // - Skip addition if index is constant 0
  llvm::SmallVector<mlir::Value> terms;
  for (size_t i = 0; i < num_indices; ++i) {
    if (isTargetConstant(0, indices[i])) {
      continue;
    }

    mlir::Value term;
    if (strides[i] == 1) {
      term = indices[i];
    } else {
      mlir::Value stride_const =
          mlir::arith::ConstantIndexOp::create(builder, loc, strides[i]);
      term =
          mlir::arith::MulIOp::create(builder, loc, indices[i], stride_const);
    }
    terms.push_back(term);
  }

  if (terms.empty()) {
    // All indices were constant 0
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  } else if (terms.size() == 1) {
    // Only one term, no addition needed
    return terms[0];
  } else {
    // Add all terms together
    mlir::Value offset = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
      offset = mlir::arith::AddIOp::create(builder, loc, offset, terms[i]);
    }
    return offset;
  }
}

mlir::Attribute ConstructThreeStagePipelinePass::mapMemorySpace(
    mlir::Attribute ktdp_memory_space) {
  // Get the DeviceManager analysis
  auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();

  // Get the import declaration which contains the mem_space_mapping
  auto* const device = device_manager.getOrImportDevice();
  if (!device) {
    return ktdp_memory_space;  // Fallback to original if no device found
  }

  // Get the mem_space_mapping attribute
  auto mem_space_mapping =
      device->getAttrOfType<mlir::ktdf_arch::MapAttr>("mem_space_mapping");
  if (!mem_space_mapping) {
    return ktdp_memory_space;  // No mapping found, return original
  }

  // Look up the ktdp memory space in the mapping
  auto mapped_attr =
      mem_space_mapping.getAttr<mlir::StringAttr>(ktdp_memory_space);
  if (mapped_attr) {
    return mapped_attr;  // Return the mapped string attribute
  }

  return ktdp_memory_space;  // Fallback to original if not found in mapping
}

std::pair<mlir::Attribute, mlir::Attribute>
ConstructThreeStagePipelinePass::getFifoAttributesForLoad(
    mlir::ktdp::LoadOp load_op) {
  // Load operations transfer data from memory (e.g., DDR) to compute unit
  // (e.g., SFU) Get the memory space from the load operation's access tile
  mlir::Attribute memory_space;
  auto access_tile = load_op.getAccessTile();
  if (auto construct_access_tile =
          mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
              access_tile.getDefiningOp())) {
    auto memory_view = construct_access_tile.getBase();
    if (auto construct_mem_view =
            mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(
                memory_view.getDefiningOp())) {
      memory_space = construct_mem_view.getMemorySpaceAttr();
    }
  }

  // Map the memory space to device namespace
  mlir::Attribute mapped_memory_space = mapMemorySpace(memory_space);

  // Get the compute unit
  mlir::Attribute compute_unit = resource_kinds_->getComputeKind();

  return {mapped_memory_space, compute_unit};
}

std::pair<mlir::Attribute, mlir::Attribute>
ConstructThreeStagePipelinePass::getFifoAttributesForStore(
    mlir::ktdp::StoreOp store_op) {
  // Store operations transfer data from compute unit (e.g., SFU) to memory
  // (e.g., DDR) Get the memory space from the store operation's access tile
  mlir::Attribute memory_space;
  auto access_tile = store_op.getAccessTile();
  if (auto construct_access_tile =
          mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
              access_tile.getDefiningOp())) {
    auto memory_view = construct_access_tile.getBase();
    if (auto construct_mem_view =
            mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(
                memory_view.getDefiningOp())) {
      memory_space = construct_mem_view.getMemorySpaceAttr();
    }
  }

  // Map the memory space to device namespace
  mlir::Attribute mapped_memory_space = mapMemorySpace(memory_space);

  // Get the compute unit
  mlir::Attribute compute_unit = resource_kinds_->getComputeKind();

  return {compute_unit, mapped_memory_space};
}

void ConstructThreeStagePipelinePass::replaceAccessTilesWithReinterpretCast(
    mlir::func::FuncOp func_op) {
  llvm::SmallVector<mlir::ktdp::ConstructAccessTilesOp> access_tiles;
  func_op.walk([&](mlir::ktdp::ConstructAccessTilesOp access_tile) {
    access_tiles.push_back(access_tile);
  });

  if (access_tiles.empty()) {
    return;
  }

  // Find the insertion point: right before the first loop in tiled_loops_
  // to ensure def before use
  mlir::Operation* insertion_point = nullptr;
  if (!tiled_loops_.empty()) {
    insertion_point = tiled_loops_.front();
  } else {
    // Fallback: find the last memory view operation
    func_op.walk([&](mlir::ktdp::ConstructMemoryViewOp memory_view) {
      insertion_point = memory_view.getOperation();
    });
  }

  if (!insertion_point) {
    func_op.emitError(
        "No insertion point found for reinterpret_cast operations");
    signalPassFailure();
    return;
  }

  mlir::OpBuilder builder(&getContext());

  for (mlir::ktdp::ConstructAccessTilesOp access_tile : access_tiles) {
    // Get the memory view (source memref) - first operand
    mlir::Value memory_view = access_tile.getBase();
    auto memory_view_type =
        mlir::dyn_cast<mlir::MemRefType>(memory_view.getType());
    if (!memory_view_type) {
      access_tile.emitError("Memory view is not a memref type");
      signalPassFailure();
      return;
    }

    // Get the access tile indices and base_map. base_map projects the index
    // operands onto the per-dimension coordinates of the source memref, so
    // indices.size() == base_map.getNumInputs() and base_map.getNumResults()
    // == memref rank. For the common case base_map is identity.
    llvm::SmallVector<mlir::Value> raw_indices = access_tile.getIndices();
    mlir::AffineMap base_map = access_tile.getBaseMap();

    // Get the access tile result type to determine sizes
    auto access_tile_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
        access_tile.getResult().getType());
    if (!access_tile_type) {
      access_tile.emitError("Result is not an access tile type");
      signalPassFailure();
      return;
    }

    // Get tile dimensions from the access tile type
    llvm::ArrayRef<int64_t> tile_shape = access_tile_type.getShape();
    llvm::SmallVector<int64_t> tile_dims(tile_shape.begin(), tile_shape.end());

    // Get strides from the originating ktdp.construct_memory_view's
    // static_strides attribute.  These are the logical strides as committed
    // in the KTIR (matching data_dti.json stride_map), which differ from the
    // row-major default whenever the tensor layout is non-contiguous in the
    // index space (e.g. a[2x256x64] with strides [64,128,1] vs row-major
    // [16384,64,1]).  Falling back to the memref type's layout would silently
    // produce wrong strides in those cases.
    llvm::SmallVector<int64_t> strides;
    {
      auto construct_mv_op = mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(
          memory_view.getDefiningOp());
      bool got_strides = false;
      if (construct_mv_op) {
        // static_strides is the DenseI64ArrayAttr on the op.
        auto static_strides_attr =
            construct_mv_op->getAttrOfType<mlir::DenseI64ArrayAttr>(
                "static_strides");
        if (static_strides_attr) {
          strides.assign(static_strides_attr.asArrayRef().begin(),
                         static_strides_attr.asArrayRef().end());
          got_strides = true;
        }
      }
      if (!got_strides) {
        // Fallback: try the memref type's layout, then row-major.
        if (auto strided_layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(
                memory_view_type.getLayout())) {
          strides.assign(strided_layout.getStrides().begin(),
                         strided_layout.getStrides().end());
        } else {
          int64_t stride = 1;
          for (int i = memory_view_type.getRank() - 1; i >= 0; --i) {
            strides.insert(strides.begin(), stride);
            stride *= memory_view_type.getShape()[i];
          }
        }
      }
    }

    // Insert the reinterpret_cast at the very start of the block containing
    // the access tile.  Placing it here ensures:
    //   1. All loop IVs used as access-tile indices dominate this point
    //      (they are block args of the same block or enclosing blocks).
    //   2. The reinterpret_cast dominates any ktdf.pipeline ops in the same
    //      block that were inserted at the block's start and hold a
    //      data_transfer referencing this access tile's result.
    // Using a single global insertion_point (e.g. tiled_loops_.front()) would
    // place the offset arithmetic OUTSIDE the loop that defines the IVs.
    mlir::Block* tile_block = access_tile->getBlock();
    // Find the first op in the block that has a user of this access tile's
    // result; insert just before that op so we remain as late as possible while
    // still dominating all uses.  Fall back to the block's first op.
    mlir::Operation* best_ip = &tile_block->front();
    for (mlir::Operation& op : *tile_block) {
      bool uses_tile = false;
      for (mlir::Value operand : op.getOperands()) {
        if (operand == access_tile.getResult()) { uses_tile = true; break; }
      }
      // Also check nested regions (e.g. ktdf.pipeline)
      op.walk([&](mlir::Operation* nested) {
        for (mlir::Value operand : nested->getOperands()) {
          if (operand == access_tile.getResult()) uses_tile = true;
        }
      });
      if (uses_tile) { best_ip = &op; break; }
    }
    builder.setInsertionPoint(best_ip);
    mlir::Location loc = access_tile.getLoc();

    // Move the memory view operation to the same insertion point if it lives
    // in the same block (ensures the cast source is textually before the cast).
    if (memory_view.getDefiningOp()->getBlock() == tile_block) {
      memory_view.getDefiningOp()->moveBefore(best_ip);
    }

    // Apply base_map to materialize one index per source-memref dimension.
    // Operands to expandAffineMap are dim-values followed by symbol-values; the
    // op's symbol_operands feed both base_map symbols (if any) and the
    // access_tile_set, so they are appended after the raw indices.
    llvm::SmallVector<mlir::Value> expand_operands(raw_indices.begin(),
                                                   raw_indices.end());
    mlir::ValueRange symbol_operands = access_tile.getSymbolOperands();
    expand_operands.append(symbol_operands.begin(), symbol_operands.end());
    std::optional<llvm::SmallVector<mlir::Value, 8>> per_dim_indices =
        mlir::affine::expandAffineMap(builder, loc, base_map, expand_operands);
    if (!per_dim_indices) {
      access_tile.emitError(
          "Failed to expand base_map into per-dimension "
          "indices");
      signalPassFailure();
      return;
    }
    llvm::SmallVector<mlir::Value> indices(per_dim_indices->begin(),
                                           per_dim_indices->end());

    // After base_map expansion, indices.size() == memref rank == strides.size()
    if (indices.size() != strides.size()) {
      access_tile.emitError("Number of indices (")
          << indices.size() << ") does not match number of strides ("
          << strides.size() << ")";
      signalPassFailure();
      return;
    }

    // Calculate offset from per-dim indices and memory view strides
    mlir::Value offset =
        computeReinterpretCastOffset(builder, loc, indices, strides);

    // Create sizes for reinterpret_cast
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    for (int64_t dim : tile_dims) {
      sizes.push_back(builder.getIndexAttr(dim));
    }

    // Create strides for reinterpret_cast (same as memory view)
    llvm::SmallVector<mlir::OpFoldResult> reinterpret_strides;
    for (int64_t stride : strides) {
      reinterpret_strides.push_back(builder.getIndexAttr(stride));
    }

    // Get ktdp memory space from construct_memory_view operation attribute
    auto construct_mem_view = mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(
        memory_view.getDefiningOp());
    assert(construct_mem_view &&
           "Memory view must be defined by construct_memory_view operation");
    mlir::Attribute ktdp_memory_space = construct_mem_view.getMemorySpaceAttr();

    // Map the ktdp memory space to the device namespace using mem_space_mapping
    mlir::Attribute mapped_memory_space = mapMemorySpace(ktdp_memory_space);

    // This propagates the mapped memory space to the reinterpret_cast
    mlir::MemRefType cast_source_type = mlir::MemRefType::get(
        memory_view_type.getShape(), memory_view_type.getElementType(),
        memory_view_type.getLayout(), mapped_memory_space);
    auto memory_space_cast = mlir::memref::MemorySpaceCastOp::create(
        builder, loc, cast_source_type, memory_view);

    // Model-A applies only to global (HBM) memory tiles: the L3 view start
    // address must be a compile-time constant (dcc-l3-model-a-immutable-addr)
    // and the per-tile dynamic offset goes into the composite subscript.
    // Per-core (LX) tiles use model-B (offset baked into the view start addr).
    //
    // Detect by checking whether the mapped memory space is NOT a per-core
    // scratchpad (LX) or compute-local (lrfreg) space.
    bool is_model_a_source = false;
    {
      auto mapped_str = mlir::dyn_cast<mlir::StringAttr>(mapped_memory_space);
      if (mapped_str) {
        llvm::StringRef ms = mapped_str.getValue();
        // Per-core: "LX" (local SRAM scratchpad). Compute-local: "lrfreg".
        // Everything else (e.g., "HBM", "DDR") is global → model A.
        is_model_a_source = (ms != "LX" && ms != "lrfreg");
      }
    }

    llvm::SmallVector<int64_t> result_shape;
    llvm::SmallVector<mlir::OpFoldResult> rc_sizes;
    llvm::SmallVector<mlir::OpFoldResult> rc_strides;
    if (is_model_a_source) {
      // 1-D cast: just the lane dimension (last dim of tile, stride 1).
      int64_t lane_count = tile_dims.back();
      result_shape = {lane_count};
      rc_sizes = {builder.getIndexAttr(lane_count)};
      rc_strides = {builder.getIndexAttr(1)};
    } else {
      result_shape.assign(tile_dims.begin(), tile_dims.end());
      rc_sizes = sizes;
      rc_strides = reinterpret_strides;
    }
    mlir::StridedLayoutAttr strided_layout;
    if (is_model_a_source) {
      strided_layout = mlir::StridedLayoutAttr::get(
          builder.getContext(), mlir::ShapedType::kDynamic, {1});
    } else {
      strided_layout = mlir::StridedLayoutAttr::get(
          builder.getContext(), mlir::ShapedType::kDynamic, strides);
    }
    mlir::MemRefType result_type =
        mlir::MemRefType::get(result_shape, memory_view_type.getElementType(),
                              strided_layout, mapped_memory_space);

    mlir::OpFoldResult offset_fold_result(offset);
    auto cast_op = mlir::memref::ReinterpretCastOp::create(
        builder, loc, result_type, memory_space_cast.getResult(),
        offset_fold_result, rc_sizes, rc_strides);
    // Replace access tile with reinterpret_cast
    access_tile.replaceAllUsesWith(cast_op.getResult());
    ops_to_delete_.push_back(access_tile.getOperation());
  }
}

void ConstructThreeStagePipelinePass::runOnFunc(mlir::func::FuncOp func_op) {
  LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func_op.getName()
                          << "\n");

  // Initialize const_builder_ at start of function
  if (!func_op.empty()) {
    const_builder_.emplace(&getContext());
    const_builder_->setInsertionPointToStart(&func_op.front());
  }

  // Step 1: Generalize named linalg operations and arith/math operations to
  // linalg.generic
  if (mlir::failed(generalizeLinalgArithMathOps(func_op))) {
    func_op.emitError("Failed to generalize linalg operations");
    signalPassFailure();
    return;
  }

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After generalization:\n"
                                             << func_op << "\n\n");

  // Step 2: Fuse consecutive linalg.generic operations
  if (mlir::failed(fuseLinalgOps(func_op))) {
    func_op.emitError("Failed to fuse linalg operations");
    signalPassFailure();
    return;
  }

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After fusion:\n"
                                             << func_op << "\n\n");

  // Matmul / reduction path: rewrite the K-loop tensor iter_arg accumulator
  // into a compute-local (LRF) memref RMW with an scf.if first-tile seed (the
  // ktdf.pipeline cannot carry a loop-result accumulator). The subsequent
  // pipeline construction then wraps only the HBM A/W loads and C store; the
  // LRF accumulator RMW is left in place (pass 20 lowers it to vector_load/
  // store on the lrfreg unit).
  // Detect the K-reduction loop BEFORE rewriteReductionAccumulator, while the
  // original tensor iter_arg still ties the matmul to the K-loop. After the
  // rewrite the matmul's DPS init is changed to an scf.if result, so the
  // iter_arg linkage is lost and detection would fail.
  mlir::scf::ForOp reduction_loop;
  func_op.walk([&](mlir::scf::ForOp for_op) {
    if (for_op.getNumRegionIterArgs() == 0) return;
    mlir::Value iter_arg = for_op.getRegionIterArg(0);
    if (!mlir::isa<mlir::RankedTensorType>(iter_arg.getType())) return;
    for_op.getBody()->walk([&](mlir::linalg::LinalgOp lop) {
      for (mlir::Value init : lop.getDpsInits()) {
        if (init == iter_arg) reduction_loop = for_op;
      }
    });
  });
  reduction_loop_ = reduction_loop;  // save for createLoopsFromLinalg

  // Now rewrite the accumulator (changes the matmul's DPS init, so must come
  // after detection above).
  rewriteReductionAccumulator(func_op);

  // For a reduction (matmul), collect only ops inside the K-loop body:
  // - HBM loads (A/W) live inside K-loop and feed the compute stage.
  // - The K-loop matmul is the pipelined compute op.
  // - The C store / truncf live OUTSIDE the K-loop; they must NOT be collected
  //   (the C-write is handled as raw ktdp ops that pass-3+ address-assigns).

  llvm::SmallVector<mlir::linalg::LinalgOp> linalg_ops;
  // Use reduction_loop scope if found; fall back to func-wide walk otherwise
  // (covers the elementwise/add case where there is no enclosing K-loop).
  auto collect_scope = reduction_loop_
                           ? static_cast<mlir::Operation*>(reduction_loop_)
                           : static_cast<mlir::Operation*>(func_op);
  collect_scope->walk([&](mlir::Operation* op) {
    if (auto load_op = mlir::dyn_cast<mlir::ktdp::LoadOp>(op)) {
      load_ops_.push_back(load_op);
      ops_to_delete_.push_back(op);
    } else if (auto store_op = mlir::dyn_cast<mlir::ktdp::StoreOp>(op)) {
      store_ops_.push_back(store_op);
      ops_to_delete_.push_back(op);
    } else if (auto linalg_op = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
      linalg_ops.push_back(linalg_op);
    }
  });

  // Step 3: Create loops from linalg operations
  createLoopsFromLinalg(linalg_ops);

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After loops created:\n"
                                             << func_op << "\n\n");

  // Step 4: Create pipeline if we have tiled loops
  if (!tiled_loops_.empty()) {
    auto innermost_loop = llvm::cast<mlir::scf::ForOp>(tiled_loops_.back());
    createPipeline(innermost_loop);
  }

  // Step 4b (Gap 5): For reduction ops, wrap any ktdp.store ops that live
  // OUTSIDE the reduction loop body in their own ktdf.pipeline (post-loop
  // store pipeline for the accumulated output).
  if (reduction_loop_) {
    llvm::SmallVector<std::pair<mlir::ktdp::StoreOp, mlir::scf::ForOp>>
        post_loop_stores;
    func_op.walk([&](mlir::ktdp::StoreOp store_op) {
      // Only collect stores that are NOT inside the reduction loop body.
      if (reduction_loop_->isProperAncestor(store_op)) return;
      // Find the enclosing N-stick loop (parent scf.for of the store_op).
      auto enc = store_op->getParentOfType<mlir::scf::ForOp>();
      post_loop_stores.push_back({store_op, enc});
    });
    for (auto& [s, enc_loop] : post_loop_stores) {
      createPostLoopStorePipeline(s, enc_loop);
    }
  }

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After pipeline created:\n"
                                             << func_op << "\n\n");

  // Step 5: Replace access tiles with reinterpret_cast after pipeline creation
  replaceAccessTilesWithReinterpretCast(func_op);

  // Step 6: Cleanup operations (removes original load/store/compute ops)
  cleanupOperations();

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After cleanup:\n"
                                             << func_op << "\n\n");
}

void ConstructThreeStagePipelinePass::runOnOperation() {
  if (DisableThisPass) return;

  mlir::ModuleOp module = getOperation();

  auto& devices = getAnalysis<mlir::ktdf_arch::DeviceManager>();
  auto* const device = devices.getOrImportDevice();
  if (!device) {
    LLVM_DEBUG(llvm::dbgs() << "No device found.\n");
    signalPassFailure();
    return;
  }
  resource_kinds_ = &getChildAnalysis<arch_view::ResourceKinds>(**device);

  auto compute_kind = resource_kinds_->getComputeKind();
  if (!compute_kind) {
    signalPassFailure();
    return;
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Starting ConstructThreeStagePipeline transformation\n");

  // Process each function in each nested module
  module.walk([&](mlir::func::FuncOp func_op) {
    resetState();
    runOnFunc(func_op);
  });
}

std::unique_ptr<mlir::Pass> scheduler::createConstructThreeStagePipelinePass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<ConstructThreeStagePipelinePass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> scheduler::createConstructThreeStagePipelinePass() {
  return std::make_unique<ConstructThreeStagePipelinePass>(
      SchedulerExtContext::dummyContext());
}
