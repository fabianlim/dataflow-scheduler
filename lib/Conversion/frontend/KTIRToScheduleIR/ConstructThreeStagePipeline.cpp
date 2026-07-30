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
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
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

  // Create a 3-stage pipeline inside innermost_loop, with one stage for loads,
  // computes, stores.
  void createPipeline(mlir::scf::ForOp innermost_loop);

  // Create a 2-stage pipeline for the explicit-M reduction path.
  // Emits an accumulate stage (with an explicit reduction scf.for inside) and a
  // store stage.  Called instead of createPipeline when the linalg.generic has
  // at least one reduction iterator.
  void createReductionPipeline(mlir::scf::ForOp innermost_loop);

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

  // Get the memory space attribute for compute-local storage (e.g. "lrfreg").
  // Queries the device for a memory resource with ktdf_arch.feature.local_to_compute.
  // Falls back to StringAttr "lrfreg" if no such resource is found.
  mlir::Attribute getLocalComputeStorageMemSpace();

  // Compute offset for reinterpret_cast from indices and strides
  mlir::Value computeReinterpretCastOffset(
      mlir::OpBuilder& builder, mlir::Location loc,
      llvm::SmallVector<mlir::Value>& indices,
      llvm::SmallVector<int64_t>& strides);

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

  // Tile sizes determined from linalg operation (iterator-space indexed,
  // 0 at reduction positions)
  llvm::SmallVector<int64_t> tile_sizes_;

  // Total number of elements (product of non-zero tile_sizes_)
  int64_t total_num_elements_ = 0;

  // True when the linalg op has at least one reduction iterator.  When set,
  // runOnFunc calls createReductionPipeline instead of createPipeline.
  bool has_reduction_ = false;

  // For each reduction iterator position (in iterator-space order), the full
  // dimension size from the original linalg op's input tensor.  Used by
  // createReductionPipeline to emit the explicit M loop.
  llvm::SmallVector<int64_t> reduction_dim_sizes_;

  // Operations to delete after pipeline creation
  llvm::SmallVector<mlir::Operation*> ops_to_delete_;

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
  has_reduction_ = false;
  reduction_dim_sizes_.clear();
  ops_to_delete_.clear();
  const_builder_.reset();
}

mlir::LogicalResult
ConstructThreeStagePipelinePass::generalizeLinalgArithMathOps(
    mlir::func::FuncOp func_op) {
  LDBG(1)
      << "Generalizing linalg, arith, and math operations to linalg.generic";

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
    LDBG(1) << "  Generalizing linalg op: " << linalg_op->getName() << "";
    rewriter.setInsertionPoint(linalg_op);

    llvm::FailureOr<mlir::linalg::GenericOp> generic_result =
        mlir::linalg::generalizeNamedOp(rewriter, linalg_op);

    if (mlir::failed(generic_result)) {
      linalg_op.emitError("Failed to generalize linalg operation");
      return mlir::failure();
    }
  }

  // Convert arith and math operations to linalg using elementwise patterns
  LDBG(1) << "  Converting arith/math operations";

  mlir::RewritePatternSet patterns(&getContext());
  mlir::linalg::populateElementwiseToLinalgConversionPatterns(patterns);

  if (mlir::failed(mlir::applyPatternsGreedily(func_op, std::move(patterns)))) {
    func_op.emitError(
        "Failed to convert arith/math operations to linalg.generic");
    return mlir::failure();
  }

  LDBG(1) << "  Successfully generalized compute ops to linalg.generic";
  return mlir::success();
}

mlir::LogicalResult ConstructThreeStagePipelinePass::fuseLinalgOps(
    mlir::func::FuncOp func_op) {
  LDBG(1) << "Fusing consecutive linalg.generic operations";

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
        LDBG(1) << "  Successfully fused operations";
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
      mlir::dyn_cast<mlir::ShapedType>(linalg_op->getResult(0).getType());
  assert(shaped_type && shaped_type.hasRank());

  mlir::Type elem_type = shaped_type.getElementType();

  auto simd_feature =
      resource_kinds_->getFeature<mlir::ktdf_arch::feature::SIMD>(
          resource_kinds_->getComputeKind());
  const auto vector_length =
      std::max(simd_feature.getLanes(elem_type), int64_t(1));

  llvm::ArrayRef<int64_t> output_shape = shaped_type.getShape();
  int64_t output_rank = output_shape.size();

  // Run the greedy algorithm on the output tensor shape (parallel dims only).
  // Start from rightmost dimension and multiply until we reach vector_length.
  int64_t product = 1;
  int64_t covered_dims = 0;

  for (int64_t i = output_rank - 1; i >= 0; --i) {
    product *= output_shape[i];
    covered_dims++;
    if (product >= vector_length) {
      break;
    }
  }

  // Build output_tile_sizes: 1 for uncovered dims, partial/full for covered.
  llvm::SmallVector<int64_t> output_tile_sizes(output_rank);
  for (int64_t i = 0; i < output_rank - covered_dims; ++i) {
    output_tile_sizes[i] = 1;
  }

  int64_t remaining_product = vector_length;
  for (int64_t i = output_rank - 1; i >= output_rank - covered_dims; --i) {
    int64_t dim_size = output_shape[i];

    // Use GCD to find largest value that divides both dim_size and
    // remaining_product.
    int64_t tile_size = std::gcd(dim_size, remaining_product);

    // Clamp to dimension size.
    tile_size = std::min(tile_size, dim_size);

    output_tile_sizes[i] = tile_size;
    remaining_product /= tile_size;
    if (remaining_product <= 1) remaining_product = 1;
  }

  // Map output_tile_sizes to the full iterator space.  For each iterator:
  //   - parallel   -> consume next output-dim tile size
  //   - reduction  -> insert 0, record the full dim size into
  //                   reduction_dim_sizes_ from the first input tensor shape
  const auto& iter_types = linalg_op.getIteratorTypesArray();
  int64_t num_iters = static_cast<int64_t>(iter_types.size());

  mlir::RankedTensorType input_type;
  if (!linalg_op.getDpsInputOperands().empty()) {
    input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        linalg_op.getDpsInputOperand(0)->get().getType());
  }

  llvm::SmallVector<int64_t> full_tile_sizes;
  full_tile_sizes.reserve(num_iters);
  size_t out_idx = 0;
  for (int64_t i = 0; i < num_iters; ++i) {
    if (iter_types[i] == mlir::utils::IteratorType::reduction) {
      full_tile_sizes.push_back(0);
      int64_t red_size = mlir::ShapedType::kDynamic;
      if (input_type && i < input_type.getRank()) {
        red_size = input_type.getDimSize(i);
      }
      reduction_dim_sizes_.push_back(red_size);
    } else {
      assert(out_idx < output_tile_sizes.size() &&
             "output_tile_sizes too short for parallel iterators");
      full_tile_sizes.push_back(output_tile_sizes[out_idx++]);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Output shape: [";
    for (int64_t i = 0; i < output_rank; ++i) {
      llvm::dbgs() << output_shape[i];
      if (i < output_rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n    Full tile sizes (iterator space): [";
    for (int64_t i = 0; i < num_iters; ++i) {
      llvm::dbgs() << full_tile_sizes[i];
      if (i < num_iters - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
  });

  return full_tile_sizes;
}

void ConstructThreeStagePipelinePass::annotateLoopsWithIteratorTypes(
    llvm::ArrayRef<mlir::Operation*> loops,
    mlir::linalg::GenericOp generic_op) {
  assert(generic_op);

  const auto& iterator_types = generic_op.getIteratorTypesArray();

  // `loops` has one entry per non-zero tile_size (i.e., per emitted scf.for).
  // `iterator_types` spans the full iterator space (including reduction dims
  // where tile_size==0 and no loop was emitted).  Walk iterator_types and
  // only consume a loop when the corresponding tile_size is non-zero.
  size_t loop_idx = 0;
  for (size_t iter_idx = 0;
       iter_idx < iterator_types.size() && loop_idx < loops.size();
       ++iter_idx) {
    // Skip iterator positions where no loop was emitted (tile_size == 0).
    if (iter_idx < tile_sizes_.size() && tile_sizes_[iter_idx] == 0) {
      continue;
    }

    auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(loops[loop_idx]);
    assert(for_op);

    mlir::ktdf::LoopType loop_type;
    if (iterator_types[iter_idx] == mlir::utils::IteratorType::parallel) {
      loop_type = mlir::ktdf::LoopType::ParallelLoop;
    } else if (iterator_types[iter_idx] ==
               mlir::utils::IteratorType::reduction) {
      loop_type = mlir::ktdf::LoopType::ReductionLoop;
    } else {
      for_op->emitError("Unsupported iterator type");
      signalPassFailure();
      return;
    }

    auto loop_type_attr =
        mlir::ktdf::LoopTypeAttr::get(&getContext(), loop_type);
    for_op->setAttr("loop_type", loop_type_attr);

    LDBG(1) << "  Annotated loop " << loop_idx << " (iter dim " << iter_idx
            << ") with loop_type: "
            << (loop_type == mlir::ktdf::LoopType::ParallelLoop ? "parallel"
                                                                : "reduction")
            << "";
    ++loop_idx;
  }
}


void ConstructThreeStagePipelinePass::createLoopsFromLinalg(
    llvm::ArrayRef<mlir::linalg::LinalgOp> linalg_ops) {
  LDBG(1) << "Creating SCF loops from linalg.generic operations";

  assert(linalg_ops.size() <= 1 &&
         "Currently only supporting one linalg.generic operation after fusion");

  mlir::IRRewriter rewriter(&getContext());
  for (mlir::linalg::LinalgOp linalg_op : linalg_ops) {
    LDBG(1) << "  Processing: " << linalg_op->getName() << "";

    rewriter.setInsertionPoint(linalg_op);

    // Determine tile sizes for the full iterator space (0 at reduction
    // positions, parallel dims tiled by the greedy algorithm).
    // reduction_dim_sizes_ is populated as a side effect.
    tile_sizes_ = determineTileSizes(linalg_op);

    if (tile_sizes_.empty()) {
      linalg_op.emitError("Could not determine tile sizes");
      signalPassFailure();
      return;
    }

    has_reduction_ = !reduction_dim_sizes_.empty();

    // Calculate total number of elements (product of non-zero tile_sizes_)
    total_num_elements_ = 1;
    for (int64_t dim : tile_sizes_) {
      if (dim != 0) total_num_elements_ *= dim;
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

    LDBG(1) << "  Successfully tiled linalg operation";

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

  // Add three token types for the three stages
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);

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

  // Create three tokens
  auto t1 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  auto t2 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  auto t3 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  fifo_results.push_back(t1.getResult());
  fifo_results.push_back(t2.getResult());
  fifo_results.push_back(t3.getResult());

  // Yield all results
  mlir::ktdf::PrivateYieldOp::create(builder, loc, fifo_results);

  return private_op;
}

void ConstructThreeStagePipelinePass::createPipeline(
    mlir::scf::ForOp innermost_loop) {
  LDBG(1) << "Creating ktdf.pipeline with three stages";

  compute_ops_.clear();
  innermost_loop.getBody()->walk([&](mlir::linalg::LinalgOp linalg_op) {
    compute_ops_.push_back(linalg_op);
  });

  LDBG(1) << "  Found " << compute_ops_.size() << " linalg compute operations";

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
    innermost_loop.emitError(
        "ConstructThreeStagePipeline: compute unit link does not use an "
        "ordered "
        "queue; only ordered queues are supported");
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

        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{private_op.getResult(fifo_count + 1U)},
            /*depends_out=*/{private_op.getResult(fifo_count + 2U)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Add data transfer operations in stage3 for stores
              createDataTransfers(builder, loc, private_op, tile_sizes_,
                                  /*is_load=*/false,
                                  /*private_result_offset=*/load_ops_.size());
            });
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
  LDBG(1) << "Cleaning up operations";

  for (auto* op : ops_to_delete_) {
    if (!op || !op->use_empty()) continue;
    deleteOpAndUnusedChainOfOperands(op);
  }
}

// Forward declarations for static helper functions defined later in this file.
static std::optional<mlir::AffineMap> findIndexingMapForLoadResult(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value);
static std::optional<mlir::AffineMap> findIndexingMapForStoreSource(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value);

mlir::Attribute
ConstructThreeStagePipelinePass::getLocalComputeStorageMemSpace() {
  // Walk all resource exemplars and find the first one tagged with
  // ktdf_arch.feature.local_to_compute.  Return its kind attribute as the
  // memory space string for the accumulator memref.
  for (auto& [kind, resource] : *resource_kinds_) {
    // resource is const; use const_cast to satisfy getOperation()'s non-const
    // signature — we only read the features attribute.
    mlir::Operation* resource_op =
        const_cast<mlir::ktdf_arch::Resource&>(resource).getOperation();
    if (mlir::ktdf_arch::getFeature<mlir::ktdf_arch::feature::LocalToCompute>(
            resource_op)) {
      return kind;
    }
  }
  // Fallback: no device annotation found — use the well-known literal.
  return mlir::StringAttr::get(&getContext(), "lrfreg");
}

void ConstructThreeStagePipelinePass::createReductionPipeline(
    mlir::scf::ForOp innermost_loop) {
  LDBG(1) << "Creating reduction pipeline (flat three-stage design)";

  // Collect compute ops from the innermost loop body.
  compute_ops_.clear();
  innermost_loop.getBody()->walk([&](mlir::linalg::LinalgOp linalg_op) {
    compute_ops_.push_back(linalg_op);
  });

  assert(compute_ops_.size() == 1 &&
         "expected exactly one linalg compute op after fusion");
  assert(load_ops_.size() == 1 &&
         "expected exactly one load op for reduction pipeline");
  assert(store_ops_.size() == 1 &&
         "expected exactly one store op for reduction pipeline");
  assert(reduction_dim_sizes_.size() == 1 &&
         "expected exactly one reduction dimension");

  // Fix up the scf.yield in innermost_loop to yield iter_args (same as
  // createPipeline does) so that tensor.insert_slice / original linalg can
  // be cleaned up later.
  auto yield_op =
      mlir::cast<mlir::scf::YieldOp>(innermost_loop.getBody()->getTerminator());
  for (mlir::Value operand : yield_op.getOperands()) {
    if (auto* def_op = operand.getDefiningOp()) {
      ops_to_delete_.push_back(def_op);
    }
  }
  mlir::OpBuilder yield_builder(yield_op);
  llvm::SmallVector<mlir::Value> new_yield_operands;
  for (mlir::BlockArgument iter_arg : innermost_loop.getRegionIterArgs()) {
    new_yield_operands.push_back(iter_arg);
  }
  mlir::scf::YieldOp::create(yield_builder, yield_op.getLoc(),
                             new_yield_operands);
  yield_op.erase();

  // Collect IVs from the tiled parallel loops.  tiled_loops_ has one entry per
  // non-zero tile size (i.e., the parallel dims only).  innermost_loop is the
  // last entry.
  llvm::SmallVector<mlir::Value> parallel_ivs;
  for (auto* loop_op : tiled_loops_) {
    parallel_ivs.push_back(
        mlir::cast<mlir::scf::ForOp>(loop_op).getInductionVar());
  }

  // Determine element type and shapes from the linalg compute op.
  mlir::linalg::LinalgOp linalg_op = compute_ops_[0];
  auto result_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(
      linalg_op->getResult(0).getType());
  assert(result_tensor_type && "expected ranked tensor result from linalg op");
  mlir::Type elem_type = result_tensor_type.getElementType();
  // Accumulator shape = tiled parallel output shape (same as linalg result).
  llvm::ArrayRef<int64_t> acc_shape = result_tensor_type.getShape();

  // FIFO sizes: total_num_elements_ elements per slot.
  llvm::SmallVector<int64_t> fifo_sizes = {total_num_elements_};

  // Get FIFO src/dest attributes.
  auto [load_src, load_dest] = getFifoAttributesForLoad(load_ops_[0]);
  auto [store_src, store_dest] = getFifoAttributesForStore(store_ops_[0]);

  auto load_fifo_type = mlir::ktdf::FifoSlotType::get(
      &getContext(), load_src, load_dest, total_num_elements_, elem_type);
  auto store_fifo_type = mlir::ktdf::FifoSlotType::get(
      &getContext(), store_src, store_dest, total_num_elements_, elem_type);

  mlir::ktdf::TokenType token_type =
      mlir::ktdf::TokenType::get(&getContext());

  // Accumulator memref type: parallel-tiled shape in the local-compute memory
  // space queried from the device.
  mlir::Attribute acc_mem_space = getLocalComputeStorageMemSpace();
  mlir::MemRefType acc_memref_type =
      mlir::MemRefType::get(acc_shape, elem_type,
                            mlir::MemRefLayoutAttrInterface{}, acc_mem_space);

  // Reduction loop trip count and last-iteration index (for conditional store).
  int64_t reduction_trip_count = reduction_dim_sizes_[0];
  int64_t last_iter_idx = reduction_trip_count - 1;

  // Get the linalg.generic body region to clone into the accumulate RMW
  // generic.
  auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(
      linalg_op.getOperation());
  assert(generic_op && "expected linalg.generic after tiling");

  // Original iterator types and indexing maps from the linalg.generic —
  // used verbatim in the accumulate stage's RMW generic.
  llvm::SmallVector<mlir::utils::IteratorType> orig_iter_types =
      generic_op.getIteratorTypesArray();
  auto indexing_maps_range = generic_op.getIndexingMapsArray();
  llvm::SmallVector<mlir::AffineMap> orig_indexing_maps(
      indexing_maps_range.begin(), indexing_maps_range.end());

  // Per-row memref type: the load FIFO delivers one row at a time.  Shape is
  // tile_sizes_ with reduction dim replaced by 1 (matching the row shape).
  llvm::SmallVector<int64_t> row_shape;
  for (int64_t ts : tile_sizes_) {
    row_shape.push_back(ts == 0 ? 1 : ts);
  }
  mlir::MemRefType row_memref_type =
      mlir::MemRefType::get(row_shape, elem_type);

  // Input indexing map for the load (used in both the load stage data_transfer
  // and the accumulate stage linalg.generic input).
  auto input_linalg_map =
      findIndexingMapForLoadResult(linalg_op, load_ops_[0].getResult());
  assert(input_linalg_map && "could not find input linalg map for load");

  // Output indexing map for the store.
  auto output_linalg_map =
      findIndexingMapForStoreSource(linalg_op, store_ops_[0].getDataTile());
  assert(output_linalg_map && "could not find output linalg map for store");

  // Build the flat three-stage ktdf.pipeline at the start of innermost_loop's
  // body.
  mlir::OpBuilder pipeline_builder(innermost_loop.getBodyRegion());
  mlir::Location loc = innermost_loop.getLoc();

  mlir::ktdf::PipelineOp::create(
      pipeline_builder, loc,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        // ------------------------------------------------------------------
        // ktdf.private: load_fifo, store_fifo, t0 (load->accum), t1 (accum->store)
        // ------------------------------------------------------------------
        llvm::SmallVector<mlir::Type> private_types = {load_fifo_type,
                                                        store_fifo_type,
                                                        token_type, token_type};
        auto priv = mlir::ktdf::PrivateOp::create(builder, loc, private_types);

        {
          mlir::Region& pr = priv.getRegion();
          mlir::Block* pb = &pr.front();
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(pb);

          auto load_fifo_alloc = mlir::ktdf::FifoAllocateOp::create(
              builder, loc, mlir::TypeRange{load_fifo_type},
              mlir::ValueRange{});
          auto store_fifo_alloc = mlir::ktdf::FifoAllocateOp::create(
              builder, loc, mlir::TypeRange{store_fifo_type},
              mlir::ValueRange{});
          auto t0 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
          auto t1 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);

          mlir::ktdf::PrivateYieldOp::create(
              builder, loc,
              mlir::ValueRange{load_fifo_alloc.getResult(0),
                               store_fifo_alloc.getResult(0), t0.getResult(),
                               t1.getResult()});
        }

        // Private results:
        //   [0] = load_fifo
        //   [1] = store_fifo
        //   [2] = t0  (LOAD depends_out, ACCUMULATE depends_in)
        //   [3] = t1  (ACCUMULATE depends_out, STORE depends_in)
        mlir::Value load_fifo_val = priv.getResult(0);
        mlir::Value store_fifo_val = priv.getResult(1);

        // ------------------------------------------------------------------
        // Stage 1 (LOAD): reduction scf.for containing one data_transfer per
        // row.
        // ------------------------------------------------------------------
        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{},
            /*depends_out=*/{priv.getResult(2)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              mlir::Value c0 =
                  mlir::arith::ConstantIndexOp::create(builder, loc, 0)
                      .getResult();
              mlir::Value c_red = mlir::arith::ConstantIndexOp::create(
                                      builder, loc, reduction_trip_count)
                                      .getResult();
              mlir::Value c1 =
                  mlir::arith::ConstantIndexOp::create(builder, loc, 1)
                      .getResult();

              auto load_loop =
                  mlir::scf::ForOp::create(builder, loc, c0, c_red, c1,
                                           /*iterArgs=*/{});
              load_loop->setAttr(
                  "loop_type",
                  mlir::ktdf::LoopTypeAttr::get(
                      &getContext(), mlir::ktdf::LoopType::ReductionLoop));

              {
                mlir::OpBuilder::InsertionGuard loop_guard(builder);
                builder.setInsertionPointToStart(load_loop.getBody());
                mlir::Value d_red_iv = load_loop.getInductionVar();

                // Build per-iterator-dim indices for the data_transfer source.
                llvm::SmallVector<mlir::Value> src_indices;
                size_t par_idx = 0;
                for (size_t i = 0; i < tile_sizes_.size(); ++i) {
                  if (tile_sizes_[i] == 0) {
                    src_indices.push_back(d_red_iv);
                  } else {
                    src_indices.push_back(parallel_ivs[par_idx++]);
                  }
                }

                mlir::Value input_memref = load_ops_[0].getAccessTile();
                mlir::AffineMap null_map;
                mlir::ktdf::DataTransferOp::create(
                    builder, loc, input_memref, *input_linalg_map, src_indices,
                    row_shape, load_fifo_val, null_map, mlir::ValueRange{},
                    fifo_sizes);
              }
            });

        // ------------------------------------------------------------------
        // Stage 2 (ACCUMULATE): alloc accumulator, fill zero, reduction
        // scf.for containing read_from_fifo + linalg.generic RMW +
        // conditional write_to_fifo on last iteration.
        // ------------------------------------------------------------------
        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{priv.getResult(2)},
            /*depends_out=*/{priv.getResult(3)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Allocate the accumulator in compute-local storage.
              auto acc_alloc =
                  mlir::memref::AllocOp::create(builder, loc, acc_memref_type);
              mlir::Value acc_val = acc_alloc.getResult();

              // Unconditionally fill with zero (hardware invariant).
              mlir::Value zero_val = mlir::arith::ConstantOp::create(
                  builder, loc, elem_type,
                  builder.getFloatAttr(elem_type, 0.0));
              {
                mlir::OperationState fill_state(
                    loc, mlir::linalg::FillOp::getOperationName());
                mlir::linalg::FillOp::build(builder, fill_state,
                                            mlir::ValueRange{zero_val},
                                            mlir::ValueRange{acc_val});
                builder.create(fill_state);
              }

              // Reduction loop.
              mlir::Value c0 =
                  mlir::arith::ConstantIndexOp::create(builder, loc, 0)
                      .getResult();
              mlir::Value c_red = mlir::arith::ConstantIndexOp::create(
                                      builder, loc, reduction_trip_count)
                                      .getResult();
              mlir::Value c1 =
                  mlir::arith::ConstantIndexOp::create(builder, loc, 1)
                      .getResult();

              auto acc_loop =
                  mlir::scf::ForOp::create(builder, loc, c0, c_red, c1,
                                           /*iterArgs=*/{});
              acc_loop->setAttr(
                  "loop_type",
                  mlir::ktdf::LoopTypeAttr::get(
                      &getContext(), mlir::ktdf::LoopType::ReductionLoop));

              {
                mlir::OpBuilder::InsertionGuard loop_guard(builder);
                builder.setInsertionPointToStart(acc_loop.getBody());
                mlir::Value d_red_iv = acc_loop.getInductionVar();

                // Read one row from the load FIFO.
                auto row = mlir::ktdf::ReadFromFifoOp::create(
                    builder, loc, row_memref_type, load_fifo_val);

                // RMW linalg.generic: uses original iterator_types and
                // indexing_maps from the source linalg.generic.  The output
                // operand is the accumulator memref (in-place update).
                mlir::OperationState generic_state(
                    loc, mlir::linalg::GenericOp::getOperationName());
                mlir::linalg::GenericOp::build(
                    builder, generic_state,
                    /*resultTensorTypes=*/mlir::TypeRange{},
                    /*inputs=*/mlir::ValueRange{row.getResult()},
                    /*outputs=*/mlir::ValueRange{acc_val},
                    orig_indexing_maps, orig_iter_types);
                auto* generic_raw = builder.create(generic_state);
                auto rmw_op =
                    mlir::cast<mlir::linalg::GenericOp>(generic_raw);

                // Clone the scalar body from the original linalg.generic.
                mlir::Region& rmw_region = rmw_op.getRegion();
                mlir::IRMapping mapping;
                generic_op.getRegion().cloneInto(&rmw_region, mapping);

                // On the last iteration, write the accumulator to the store
                // FIFO.  This is a peel of the final iteration's store — the
                // conditional is evaluated at runtime.
                mlir::Value c_last = mlir::arith::ConstantIndexOp::create(
                                         builder, loc, last_iter_idx)
                                         .getResult();
                mlir::Value is_last = mlir::arith::CmpIOp::create(
                    builder, loc, mlir::arith::CmpIPredicate::eq, d_red_iv,
                    c_last);
                mlir::scf::IfOp::create(
                    builder, loc, is_last,
                    [&](mlir::OpBuilder& then_builder, mlir::Location then_loc) {
                      mlir::ktdf::WriteToFifoOp::create(then_builder, then_loc,
                                                        acc_val, store_fifo_val);
                      mlir::scf::YieldOp::create(then_builder, then_loc,
                                                  mlir::ValueRange{});
                    });
              }
            })
            .setApplicableUnitsAttr(
                builder.getArrayAttr(resource_kinds_->getComputeKind()));

        // ------------------------------------------------------------------
        // Stage 3 (STORE): data_transfer from store FIFO to output HBM memref.
        // ------------------------------------------------------------------
        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{priv.getResult(3)},
            /*depends_out=*/{},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              mlir::Value output_memref = store_ops_[0].getAccessTile();

              // Store indices: parallel IVs for non-zero tile dims, c0 for
              // reduction dims.  The output linalg map spans the full iterator
              // space so we must supply one value per iterator dim.
              llvm::SmallVector<mlir::Value> store_loop_ivs;
              size_t par_idx = 0;
              for (size_t i = 0; i < tile_sizes_.size(); ++i) {
                if (tile_sizes_[i] == 0) {
                  store_loop_ivs.push_back(
                      mlir::arith::ConstantIndexOp::create(builder, loc, 0)
                          .getResult());
                } else {
                  store_loop_ivs.push_back(parallel_ivs[par_idx++]);
                }
              }

              // Output tile sizes: the accumulator shape (parallel tile dims).
              llvm::SmallVector<int64_t> output_tile_sizes(acc_shape.begin(),
                                                           acc_shape.end());

              mlir::AffineMap null_map;
              mlir::ktdf::DataTransferOp::create(
                  builder, loc, store_fifo_val, null_map, mlir::ValueRange{},
                  fifo_sizes, output_memref, *output_linalg_map, store_loop_ivs,
                  output_tile_sizes);
            });
      });
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
    getOperation()->emitError(
        "ConstructThreeStagePipeline: no compute resource kind found in device "
        "description");
    signalPassFailure();
    return;
  }
  auto compute = resource_kinds_->getResource(compute_kind);
  auto incoming = mlir::ktdf_arch::getLink(
      mlir::ktdf_arch::LinkDirection::Incoming, compute);
  auto outgoing = mlir::ktdf_arch::getLink(
      mlir::ktdf_arch::LinkDirection::Outgoing, compute);
  if (!incoming || !outgoing) {
    getOperation()->emitError(
        "ConstructThreeStagePipeline: compute resource has no incoming or "
        "outgoing link in device description");
    signalPassFailure();
    return;
  }
  const auto granularity_in =
      incoming.getProperty<mlir::ktdf_arch::TransferGranularityAttr>();
  const auto granularity_out =
      outgoing.getProperty<mlir::ktdf_arch::TransferGranularityAttr>();
  if (!granularity_in || !granularity_out) {
    getOperation()->emitError(
        "ConstructThreeStagePipeline: compute resource link is missing a "
        "TransferGranularity property");
    signalPassFailure();
    return;
  }
  const auto max_in = maxOrDefault(granularity_in.asArrayRef());
  const auto max_out = maxOrDefault(granularity_out.asArrayRef());
  if (max_in != max_out) {
    getOperation()->emitError(
        "ConstructThreeStagePipeline: incoming and outgoing transfer "
        "granularities differ (")
        << max_in << " vs " << max_out << ")";
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
    // Get the access_tile value (operand of load/store op) and the linalg
    // indexing map for the matching operand.
    mlir::Value access_tile_value;
    std::optional<mlir::AffineMap> indexing_map;
    mlir::Operation* err_anchor;
    if (is_load) {
      mlir::ktdp::LoadOp load_op = load_ops_[i];
      access_tile_value = load_op.getAccessTile();
      err_anchor = load_op.getOperation();
      indexing_map =
          findIndexingMapForLoadResult(linalg_op, load_op.getResult());
    } else {
      mlir::ktdp::StoreOp store_op = store_ops_[i];
      access_tile_value = store_op.getAccessTile();
      err_anchor = store_op.getOperation();
      indexing_map =
          findIndexingMapForStoreSource(linalg_op, store_op.getDataTile());
    }

    if (!indexing_map) {
      err_anchor->emitError(
          "could not locate matching linalg operand to project loop IVs and "
          "tile sizes through; the data_transfer rank would not match the "
          "underlying memref");
      signalPassFailure();
      return;
    }

    // Project tile_sizes through the operand's indexing map.
    llvm::SmallVector<int64_t> access_tile_sizes;
    if (mlir::failed(projectSizesThroughIndexingMap(
            *indexing_map, tile_sizes, access_tile_sizes,
            [&]() { return err_anchor->emitError(); }))) {
      signalPassFailure();
      return;
    }

    // Get the FIFO slot from ktdf.private results.
    mlir::Value fifo_slot = private_op.getResult(private_result_offset + i);

    // The fifo side gets a null AffineMap; the access-tile (memref) side
    // gets the linalg operand's indexing map directly. loop_ivs feeds the
    // map's dim inputs.
    mlir::AffineMap null_map;
    if (is_load) {
      mlir::ktdf::DataTransferOp::create(builder, loc, access_tile_value,
                                         *indexing_map, loop_ivs,
                                         access_tile_sizes, fifo_slot, null_map,
                                         mlir::ValueRange{}, fifo_sizes);
    } else {
      mlir::ktdf::DataTransferOp::create(
          builder, loc, fifo_slot, null_map, mlir::ValueRange{}, fifo_sizes,
          access_tile_value, *indexing_map, loop_ivs, access_tile_sizes);
    }
  }
}

}  // namespace

void ConstructThreeStagePipelinePass::createComputeOps(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf::PrivateOp private_op) {
  LDBG(1) << "Creating compute operations in stage 2";

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
  LDBG(1) << "  Cloning compute op: " << compute_op->getName() << "";

  // Create a dummy tensor.empty for the output operand with the tiled tensor
  // type
  auto empty_tensor =
      mlir::tensor::EmptyOp::create(builder, loc, tiled_tensor_type.getShape(),
                                    tiled_tensor_type.getElementType());

  // Map the output extract_slice to the empty tensor
  auto linalg_op =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(compute_op.getOperation());
  if (linalg_op && linalg_op.getNumDpsInits() > 0) {
    mlir::Value output_operand = linalg_op.getDpsInitOperand(0)->get();
    mapper.map(output_operand, empty_tensor.getResult());
  }

  auto* cloned = builder.clone(*compute_op, mapper);

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

    // Get strides from memory view type
    llvm::SmallVector<int64_t> strides;
    if (auto strided_layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(
            memory_view_type.getLayout())) {
      strides.assign(strided_layout.getStrides().begin(),
                     strided_layout.getStrides().end());
    } else {
      // Default strides for row-major layout
      int64_t stride = 1;
      for (int i = memory_view_type.getRank() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= memory_view_type.getShape()[i];
      }
    }

    builder.setInsertionPoint(insertion_point);
    mlir::Location loc = access_tile.getLoc();

    // Move the memory view operation right before the insertion point to ensure
    // it dominates the reinterpret_cast
    memory_view.getDefiningOp()->moveBefore(insertion_point);

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

    llvm::SmallVector<int64_t> result_shape(tile_dims.begin(), tile_dims.end());
    mlir::StridedLayoutAttr strided_layout = mlir::StridedLayoutAttr::get(
        builder.getContext(), mlir::ShapedType::kDynamic, strides);
    mlir::MemRefType result_type =
        mlir::MemRefType::get(result_shape, memory_view_type.getElementType(),
                              strided_layout, mapped_memory_space);

    mlir::OpFoldResult offset_fold_result(offset);
    auto cast_op = mlir::memref::ReinterpretCastOp::create(
        builder, loc, result_type, memory_space_cast.getResult(),
        offset_fold_result, sizes, reinterpret_strides);
    // Replace access tile with reinterpret_cast
    access_tile.replaceAllUsesWith(cast_op.getResult());
    ops_to_delete_.push_back(access_tile.getOperation());
  }
}

void ConstructThreeStagePipelinePass::runOnFunc(mlir::func::FuncOp func_op) {
  LDBG(1) << "Processing function: " << func_op.getName() << "";

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

  LDBG(1) << "After generalization:\n" << func_op << "\n";

  // Step 2: Fuse consecutive linalg.generic operations
  if (mlir::failed(fuseLinalgOps(func_op))) {
    func_op.emitError("Failed to fuse linalg operations");
    signalPassFailure();
    return;
  }

  LDBG(1) << "After fusion:\n" << func_op << "\n";

  // Collect ktdp load/store operations and linalg operations
  llvm::SmallVector<mlir::linalg::LinalgOp> linalg_ops;
  func_op.walk([&](mlir::Operation* op) {
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

  LDBG(1) << "After loops created:\n" << func_op << "\n";

  // Step 4: Create pipeline if we have tiled loops
  if (!tiled_loops_.empty()) {
    auto innermost_loop = llvm::cast<mlir::scf::ForOp>(tiled_loops_.back());
    if (has_reduction_)
      createReductionPipeline(innermost_loop);
    else
      createPipeline(innermost_loop);
  }

  LDBG(1) << "After pipeline created:\n" << func_op << "\n";

  // Step 5: Replace access tiles with reinterpret_cast after pipeline creation
  replaceAccessTilesWithReinterpretCast(func_op);

  // Step 6: Cleanup operations (removes original load/store/compute ops)
  cleanupOperations();

  LDBG(1) << "After cleanup:\n" << func_op << "\n";
}

void ConstructThreeStagePipelinePass::runOnOperation() {
  if (DisableThisPass) return;

  mlir::ModuleOp module_op = getOperation();

  auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
  auto* const device = device_manager.getOrImportDevice();
  if (!device) {
    module_op->emitError(
        "Unable to import the device specification. This could happen if the "
        "device spec file is empty or contains multiple devices");
    signalPassFailure();
    return;
  }
  resource_kinds_ =
      &getChildAnalysis<arch_view::ResourceKinds>(device->getDeclaration());

  auto compute_kind = resource_kinds_->getComputeKind();
  if (!compute_kind) {
    module_op->emitError(
        "ConstructThreeStagePipeline: no compute resource kind found in device "
        "description");
    signalPassFailure();
    return;
  }

  LDBG(1) << "Starting ConstructThreeStagePipeline transformation";

  // Process each function in each nested module
  module_op.walk([&](mlir::func::FuncOp func_op) {
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
