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

#include <functional>

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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

// ---------------------------------------------------------------------------
// Site-based pipeline construction scaffolding.
//
// A func's plan is a StructuralInfo (the loop-nest half) plus a list of
// PipelineSite (one per compute group). instantiatePipeline materializes one
// ktdf.pipeline per site. Currently only the elementwise path is exercised:
// accumulatorBuf is always null and reductionLoops is always empty, so there is
// no is_reduction branch anywhere in the emitter.
// ---------------------------------------------------------------------------

// Forward declarations of the baseline address helpers (defined later in this
// file). The elementwise composable blocks are verbatim lifts that delegate to
// these.
static std::optional<mlir::AffineMap> findIndexingMapForLoadResult(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value);
static std::optional<mlir::AffineMap> findIndexingMapForStoreSource(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value);
static mlir::LogicalResult projectSizesThroughIndexingMap(
    mlir::AffineMap indexing_map, llvm::ArrayRef<int64_t> tile_sizes,
    llvm::SmallVectorImpl<int64_t>& projected_sizes,
    mlir::function_ref<mlir::InFlightDiagnostic()> emit_error);

// The context injected into the composable address blocks. Minimal: only what
// the two elementwise blocks read.
struct AddrCtx {
  mlir::linalg::LinalgOp linalgOp;  // the fused generic (compute_ops_[0])
  mlir::ktdp::LoadOp loadOp;        // valid iff is_load
  mlir::ktdp::StoreOp storeOp;      // valid iff !is_load
  bool is_load;
  llvm::SmallVector<mlir::Value> tiledLoopIVs;  // subscript IVs
  llvm::ArrayRef<int64_t> tileSizes;            // tile_sizes_
  mlir::Operation* errAnchor;                   // diagnostic anchor
  // Reduction path only: the ktdp.construct_access_tile whose indices are the
  // per-dimension subscripts (%n, %m, %c0) and whose result shape gives the
  // transfer sizes. Null on the elementwise path.
  mlir::ktdp::ConstructAccessTilesOp accessTile;
};

struct AddressParts {
  mlir::AffineMap sourceMap;
  llvm::SmallVector<mlir::Value> subscriptIVs;
  llvm::SmallVector<int64_t> sizes;
  // The memref the transfer addresses. Seeded by the caller with the raw
  // access-tile value; a MemrefBuilder may rewrite it (reduction -> full view).
  mlir::Value sourceMemref;
};

// Source-map builder: returns nullopt on failure (caller emits the diagnostic
// on errAnchor, exactly as the baseline did).
struct SourceMapBuilder {
  virtual ~SourceMapBuilder() = default;
  virtual std::optional<mlir::AffineMap> build(AddrCtx&) = 0;
  // The subscript IVs feeding the map's dim inputs. Elementwise projects the
  // manufactured tiled-loop IVs; reduction uses the access-tile indices.
  virtual llvm::SmallVector<mlir::Value> subscripts(AddrCtx& ctx) = 0;
};

// Sizes builder: fills `out` with the projected sizes; returns failure (and has
// emitted its own diagnostic) on an unsupported map, matching the baseline.
struct SizesBuilder {
  virtual ~SizesBuilder() = default;
  virtual mlir::LogicalResult build(AddrCtx&, mlir::AffineMap,
                                    llvm::SmallVectorImpl<int64_t>&) = 0;
};

// Memref builder: given the raw access-tile value the caller seeded, returns
// the memref the transfer should address. Elementwise keeps the access-tile
// value verbatim (step 5 rewrites it into a reinterpret_cast); reduction swaps
// in the full memory view. This is what keeps createDataTransfers policy-free.
struct MemrefBuilder {
  virtual ~MemrefBuilder() = default;
  virtual mlir::Value build(mlir::OpBuilder&, mlir::Location, AddrCtx&,
                            mlir::Value seeded) = 0;
};

// Elementwise: pass the seeded access-tile value straight through.
struct MemrefPassthrough : MemrefBuilder {
  mlir::Value build(mlir::OpBuilder&, mlir::Location, AddrCtx&,
                    mlir::Value seeded) override {
    return seeded;
  }
};

// Elementwise source map = the linalg operand's indexing map (verbatim lift of
// findIndexingMapForLoadResult / findIndexingMapForStoreSource selection).
struct SourceMapFromLinalg : SourceMapBuilder {
  std::optional<mlir::AffineMap> build(AddrCtx& ctx) override {
    LLVM_DEBUG(llvm::dbgs()
               << "    SourceMapFromLinalg: building "
               << (ctx.is_load ? "load" : "store") << " source_map\n");
    if (ctx.is_load) {
      return findIndexingMapForLoadResult(ctx.linalgOp,
                                          ctx.loadOp.getResult());
    }
    return findIndexingMapForStoreSource(ctx.linalgOp,
                                         ctx.storeOp.getDataTile());
  }
  llvm::SmallVector<mlir::Value> subscripts(AddrCtx& ctx) override {
    return ctx.tiledLoopIVs;
  }
};

// Elementwise sizes = tile_sizes projected through the indexing map (verbatim
// lift of the projectSizesThroughIndexingMap call).
struct SizesByProjection : SizesBuilder {
  mlir::LogicalResult build(AddrCtx& ctx, mlir::AffineMap sourceMap,
                            llvm::SmallVectorImpl<int64_t>& out) override {
    LLVM_DEBUG(llvm::dbgs()
               << "    SizesByProjection: projecting tile sizes through the "
                  "indexing map\n");
    return projectSizesThroughIndexingMap(
        sourceMap, ctx.tileSizes, out,
        [&]() { return ctx.errAnchor->emitError(); });
  }
};

// Reduction source map = the access-tile order map (identity over the source
// memref rank), so the data_transfer addresses the FULL memory view directly
// with one subscript per memref dimension (%n, %m, %c0 for the load; %n, %c0
// for the store). No reinterpret_cast / tiled-loop projection is involved.
struct SourceMapFromAccessTile : SourceMapBuilder {
  std::optional<mlir::AffineMap> build(AddrCtx& ctx) override {
    LLVM_DEBUG(llvm::dbgs()
               << "    SourceMapFromAccessTile: identity map over the "
                  "access-tile order\n");
    if (!ctx.accessTile) return std::nullopt;
    return ctx.accessTile.getAccessTileOrder();
  }
  llvm::SmallVector<mlir::Value> subscripts(AddrCtx& ctx) override {
    llvm::SmallVector<mlir::Value> indices = ctx.accessTile.getIndices();
    return indices;
  }
};

// Reduction sizes = the access-tile result shape ([1,1,64] load / [1,64]
// store), one entry per source-memref dimension.
struct SizesFromTileShape : SizesBuilder {
  mlir::LogicalResult build(AddrCtx& ctx, mlir::AffineMap /*sourceMap*/,
                            llvm::SmallVectorImpl<int64_t>& out) override {
    LLVM_DEBUG(llvm::dbgs()
               << "    SizesFromTileShape: pulling sizes from access-tile "
                  "result shape\n");
    auto tile_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
        ctx.accessTile.getResult().getType());
    if (!tile_type) return mlir::failure();
    out.assign(tile_type.getShape().begin(), tile_type.getShape().end());
    return mlir::success();
  }
};

// Result of planAddress: distinguishes the two baseline failure modes so the
// caller can reproduce the exact diagnostics.
enum class AddressStatus { Ok, NoSourceMap, SizesFailed };

// The single per-transfer address entry point. The offset is NOT computed here
// (it stays in computeReinterpretCastOffset). subscriptIVs are always the
// tiled-loop IVs. On NoSourceMap the caller emits the "could not locate"
// diagnostic; on SizesFailed the sizes builder has already emitted its own
// diagnostic.
static AddressStatus planAddress(mlir::OpBuilder& builder, mlir::Location loc,
                                 AddrCtx& ctx, SourceMapBuilder& mapB,
                                 SizesBuilder& sizeB, MemrefBuilder& memB,
                                 AddressParts& out) {
  LLVM_DEBUG(llvm::dbgs() << "  planAddress for "
                          << (ctx.is_load ? "load" : "store") << " transfer\n");
  std::optional<mlir::AffineMap> sourceMap = mapB.build(ctx);
  if (!sourceMap) {
    LLVM_DEBUG(llvm::dbgs() << "  planAddress: no source_map (NoSourceMap)\n");
    return AddressStatus::NoSourceMap;
  }

  out.sourceMap = *sourceMap;
  out.subscriptIVs = mapB.subscripts(ctx);
  if (mlir::failed(sizeB.build(ctx, out.sourceMap, out.sizes))) {
    LLVM_DEBUG(llvm::dbgs() << "  planAddress: sizes builder failed "
                               "(SizesFailed)\n");
    return AddressStatus::SizesFailed;
  }
  out.sourceMemref = memB.build(builder, loc, ctx, out.sourceMemref);
  LLVM_DEBUG(llvm::dbgs() << "  planAddress: Ok\n");
  return AddressStatus::Ok;
}

struct SitePosition {
  mlir::Block* block;  // the loop body block to insert the pipeline into
};

struct PipelineSpec {
  llvm::SmallVector<mlir::ktdp::LoadOp> loads;
  llvm::SmallVector<mlir::ktdp::StoreOp> stores;
  mlir::linalg::LinalgOp linalgOp;
  // The reduction accumulator buffer (lrfreg), or null for elementwise. This is
  // the sole discriminator between the elementwise and reduction stage layouts.
  // For now we model a SINGLE accumulator buffer per reduction (one output
  // group); the multi-buffer extension (e.g. multiple concurrently-live
  // accumulators) is deferred.
  mlir::Value accumulatorBuf;  // currently always null (elementwise)
  SourceMapBuilder* mapBuilder;
  SizesBuilder* sizeBuilder;
  MemrefBuilder* memrefBuilder;
};

struct PipelineSite {
  SitePosition pos;
  PipelineSpec spec;
};

// Carries the per-group reduction shape discovered by findReductionLoop: the
// reused inner reduction scf.for (%m), its tensor accumulator iter_arg, and the
// fused linalg whose DPS init is fed by that iter_arg.
struct ReductionInfo {
  mlir::scf::ForOp loop;          // the %m reduction loop (reused)
  mlir::BlockArgument accIterArg; // the tensor iter_arg (%acc)
  mlir::linalg::LinalgOp linalg;  // the reduction linalg (outs = %acc)
};

struct StructuralInfo {
  llvm::SmallVector<mlir::scf::ForOp> parallelLoops;   // nest OUTSIDE the pipeline
  llvm::SmallVector<mlir::scf::ForOp> reductionLoops;  // reduction %m loops
  llvm::SmallVector<ReductionInfo> reductions;         // one per reduction group
};

// ---------------------------------------------------------------------------
// Internal stage-list model (implementation detail; NOT part of the
// PipelineSite/PipelineSpec/StructuralInfo interface).
//
// A ktdf.pipeline is a LINEAR chain of stages. instantiatePipeline derives the
// chain locally from the spec and emits it with one uniform loop, so the number
// of stages -- and hence the number of stage-linking tokens -- is data, never a
// hardcoded constant and never an `is_reduction` fork:
//
//   #tokens == #stages     (stage i's depends_out == stage i+1's depends_in)
//
// The two shapes this pass produces, both expressed as a list of StageEmit:
//   - elementwise (accumulatorBuf == null): [ Load, Compute, Store ]  (3 stages)
//   - reduction   (accumulatorBuf  set):    [ Load+Compute, Store ]   (2 stages)
//       where the first stage fuses the load transfer with the compute RMW and
//       nests structural.reductionLoops around them, because the reduction load
//       is addressed by the reduction IV and must sit inside that loop (the HBM
//       immutable-base rule keeps the IV in the data_transfer source_map, not
//       the reinterpret_cast offset, so the transfer op references the IV as an
//       SSA operand and must be in its scope). Only the reduction stage carries
//       a loop; every stage's body is otherwise the same work emitters.
//
// `isCompute` selects the stage that carries applicable_units=["SFP"] (the
// compute-role stage; §4.4 cross-pass invariant). `body` emits the stage's work
// given the private op providing the FIFO slots.
struct StageEmit {
  bool isCompute = false;
  std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ktdf::PrivateOp)>
      body;
};

struct ConstructThreeStagePipelinePass
    : public impl::ConstructThreeStagePipelinePassBase<
          ConstructThreeStagePipelinePass> {
  ConstructThreeStagePipelinePass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    ConstructThreeStagePipelinePassBase::getDependentDialects(registry);
    // The reduction path emits bufferization.to_tensor /
    // materialize_in_destination for the accumulator RMW.
    registry.insert<mlir::bufferization::BufferizationDialect>();
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

  // Materialize ONE ktdf.pipeline for `site`. The single uniform body: build
  // ktdf.private, then load/compute/store stages. Currently handles only the
  // elementwise arm. Reads from a PipelineSite + StructuralInfo rather than
  // pass member state.
  void instantiatePipeline(const PipelineSite& site,
                           const StructuralInfo& structural);

  // Detect a reduction group: an scf.for with a single tensor iter_arg whose
  // value feeds (as the DPS init) the fused linalg. Returns std::nullopt for
  // the elementwise path. Populates the reused-loop / iter_arg / linalg handles.
  std::optional<ReductionInfo> findReductionLoop(mlir::func::FuncOp func_op);

  // Materialize ONE reduction ktdf.pipeline: the OUTER private (accumulator +
  // store FIFO + 2 tokens), the ACCUMULATE stage (the reused %m loop wrapping
  // an INNER load+RMW pipeline, then the post-loop write_to_fifo), and the
  // STORE stage. Reuses createPrivateOp / createDataTransfers / the StageEmit
  // machinery for the inner pipeline.
  void instantiateReductionPipeline(const PipelineSite& site,
                                    const ReductionInfo& reduction);

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
                           const PipelineSpec& spec,
                           llvm::ArrayRef<int64_t> tile_sizes, bool is_load,
                           size_t private_result_offset);

  // Create ktdf.private operation with FIFO slots and tokens
  // Returns the created private operation. `num_tokens` is the number of
  // stage-linking tokens to allocate (one per stage in the linear chain);
  // the elementwise pipeline has 3 stages -> 3 tokens.
  mlir::ktdf::PrivateOp createPrivateOp(mlir::OpBuilder& builder,
                                        mlir::Location loc, size_t num_tokens);

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

  // Reduction path: materialize (once, memoized) the full-rank memory view of
  // an access tile's underlying memory view, in the device memory space. The
  // reduction data_transfers address this full view directly (identity map +
  // per-dim subscripts). To keep the view chain uniform for the backend, an
  // identity reinterpret_cast (offset 0, full shape) is appended so every view
  // is construct_memory_view -> memory_space_cast -> reinterpret_cast.
  mlir::Value materializeFullView(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::ktdp::ConstructAccessTilesOp access_tile);

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

  // Tile sizes determined from linalg operation
  llvm::SmallVector<int64_t> tile_sizes_;

  // Total number of elements (product of tile_sizes_)
  int64_t total_num_elements_ = 0;

  // Operations to delete after pipeline creation
  llvm::SmallVector<mlir::Operation*> ops_to_delete_;

  // Builder for constants at function start
  std::optional<mlir::OpBuilder> const_builder_;

  // True while emitting a reduction group: data_transfers address the full
  // memory view (via memory_space_cast) with the access-tile subscripts, and
  // the tiling / reinterpret_cast steps are skipped.
  bool is_reduction_ = false;

  // Memoized full-view memory_space_cast per underlying memory-view op (keyed
  // by the construct_memory_view), so multiple access tiles over the same view
  // share one cast.
  llvm::DenseMap<mlir::Operation*, mlir::Value> full_view_cache_;
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
  is_reduction_ = false;
  full_view_cache_.clear();
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
    mlir::OpBuilder& builder, mlir::Location loc, size_t num_tokens) {
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

  // Add one token type per stage (the linear stage chain links stage i's
  // depends_out to stage i+1's depends_in).
  for (size_t i = 0; i < num_tokens; ++i) {
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

  // Create one token per stage.
  for (size_t i = 0; i < num_tokens; ++i) {
    auto token = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
    fifo_results.push_back(token.getResult());
  }

  // Yield all results
  mlir::ktdf::PrivateYieldOp::create(builder, loc, fifo_results);

  return private_op;
}

std::optional<ReductionInfo>
ConstructThreeStagePipelinePass::findReductionLoop(mlir::func::FuncOp func_op) {
  std::optional<ReductionInfo> result;
  func_op.walk([&](mlir::scf::ForOp for_op) {
    if (result) return;
    // A reduction loop carries exactly one tensor iter_arg.
    if (for_op.getNumRegionIterArgs() != 1) return;
    mlir::BlockArgument acc = for_op.getRegionIterArgs().front();
    if (!mlir::isa<mlir::RankedTensorType>(acc.getType())) return;

    // The iter_arg must feed a linalg op's DPS init inside the loop body.
    for (mlir::Operation* user : acc.getUsers()) {
      auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(user);
      if (!linalg) continue;
      bool is_init = false;
      for (mlir::Value init : linalg.getDpsInits()) {
        if (init == acc) is_init = true;
      }
      if (!is_init) continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "  findReductionLoop: found reduction loop with tensor "
                    "iter_arg feeding linalg outs\n");
      result = ReductionInfo{for_op, acc, linalg};
      return;
    }
  });
  return result;
}

void ConstructThreeStagePipelinePass::instantiatePipeline(
    const PipelineSite& site, const StructuralInfo& structural) {
  const PipelineSpec& spec = site.spec;
  const bool is_reduction = static_cast<bool>(spec.accumulatorBuf);
  LLVM_DEBUG(llvm::dbgs() << "Creating ktdf.pipeline with three stages "
                          << "(accumulatorBuf "
                          << (is_reduction ? "set" : "null") << ")\n");

  // The loop body block where the pipeline is materialized, and its enclosing
  // scf.for (the innermost parallel loop).
  mlir::Block* body_block = site.pos.block;
  auto innermost_loop =
      mlir::cast<mlir::scf::ForOp>(body_block->getParentOp());

  compute_ops_.clear();
  body_block->walk([&](mlir::linalg::LinalgOp linalg_op) {
    compute_ops_.push_back(linalg_op);
  });

  LLVM_DEBUG(llvm::dbgs() << "  Found " << compute_ops_.size()
                          << " linalg compute operations\n");

  // Save original scf.yield operands to ops_to_delete_ before changing the
  // yield. This ensures tensor.insert_slice and linalg.generic get cleaned up
  // later.
  auto yield_op =
      mlir::cast<mlir::scf::YieldOp>(body_block->getTerminator());
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
    signalPassFailure();
    return;
  }

  // Derive the linear stage chain from the spec. The load/compute/store work
  // emitters are the same in both cases; only their PARTITION into stages (and
  // whether a stage nests the reduction loop) differs, and it is derived here
  // from accumulatorBuf -- no is_reduction fork in the emit loop below.
  //
  // Work emitters (shared, capture-by-value of the per-stage inputs):
  auto emit_loads = [this, &spec](mlir::OpBuilder& b, mlir::Location loc,
                                  mlir::ktdf::PrivateOp private_op) {
    createDataTransfers(b, loc, private_op, spec, tile_sizes_,
                        /*is_load=*/true, /*private_result_offset=*/0);
  };
  auto emit_compute = [this](mlir::OpBuilder& b, mlir::Location loc,
                             mlir::ktdf::PrivateOp private_op) {
    createComputeOps(b, loc, private_op);
  };
  auto emit_stores = [this, &spec](mlir::OpBuilder& b, mlir::Location loc,
                                   mlir::ktdf::PrivateOp private_op) {
    createDataTransfers(b, loc, private_op, spec, tile_sizes_,
                        /*is_load=*/false,
                        /*private_result_offset=*/spec.loads.size());
  };

  llvm::SmallVector<StageEmit> stages;
  if (!is_reduction) {
    // Elementwise: three flat stages -- Load, Compute, Store.
    stages.push_back({/*isCompute=*/false, emit_loads});
    stages.push_back({/*isCompute=*/true, emit_compute});
    stages.push_back({/*isCompute=*/false, emit_stores});
  } else {
    // Reduction (compiled but not yet reached -- site derivation still leaves
    // accumulatorBuf null): two stages. The first FUSES the load transfer and
    // the compute RMW inside the reduction loop nest; the load is addressed by
    // the reduction IV, so its data_transfer must sit inside that loop. The
    // second stage drains the settled accumulator. The reduction *mechanics*
    // (accumulator rewrite, lrfreg RMW, nesting structural.reductionLoops and
    // the post-loop write_to_fifo) are added in the following commit; this
    // arm only records the 2-stage layout so the token/stage count derivation
    // and the reviewer's mental model are complete.
    LLVM_DEBUG(llvm::dbgs()
               << "  reduction layout: fusing load+compute over "
               << structural.reductionLoops.size() << " reduction loop(s)\n");
    stages.push_back({/*isCompute=*/true,
                      [&](mlir::OpBuilder& b, mlir::Location loc,
                          mlir::ktdf::PrivateOp private_op) {
                        emit_loads(b, loc, private_op);
                        emit_compute(b, loc, private_op);
                      }});
    stages.push_back({/*isCompute=*/false, emit_stores});
  }

  LLVM_DEBUG(llvm::dbgs() << "  pipeline has " << stages.size()
                          << " stage(s) / token(s)\n");

  mlir::ktdf::PipelineOp::create(
      builder, innermost_loop.getLoc(),
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        // One stage-linking token per stage; FIFO slots come first in the
        // private results, then the tokens.
        auto private_op = createPrivateOp(builder, loc, stages.size());
        size_t fifo_count = spec.loads.size() + spec.stores.size();

        // Emit the linear chain uniformly: stage i consumes token i-1 (none for
        // the first) and produces token i.
        for (size_t i = 0; i < stages.size(); ++i) {
          const StageEmit& stage = stages[i];
          llvm::SmallVector<mlir::Value> depends_in;
          if (i > 0) {
            depends_in.push_back(private_op.getResult(fifo_count + i - 1));
          }
          mlir::Value depends_out = private_op.getResult(fifo_count + i);

          auto stage_op = mlir::ktdf::StageOp::create(
              builder, loc, depends_in, /*depends_out=*/{depends_out},
              [&](mlir::OpBuilder& builder, mlir::Location loc) {
                stage.body(builder, loc, private_op);
              });
          if (stage.isCompute) {
            stage_op.setApplicableUnitsAttr(
                builder.getArrayAttr(resource_kinds_->getComputeKind()));
          }
        }
      });
}

namespace {
// Reduction memref policy: swap the seeded access-tile value for the full
// memory-view memory_space_cast (materialized by the pass). Holds a callback so
// it does not need to be a pass member.
struct MemrefFullView : MemrefBuilder {
  std::function<mlir::Value(mlir::OpBuilder&, mlir::Location,
                            mlir::ktdp::ConstructAccessTilesOp)>
      resolve;
  mlir::Value build(mlir::OpBuilder& b, mlir::Location loc, AddrCtx& ctx,
                    mlir::Value /*seeded*/) override {
    assert(ctx.accessTile && "reduction transfer expects an access tile");
    return resolve(b, loc, ctx.accessTile);
  }
};
}  // namespace

void ConstructThreeStagePipelinePass::instantiateReductionPipeline(
    const PipelineSite& site, const ReductionInfo& reduction) {
  const PipelineSpec& spec = site.spec;
  mlir::Block* body_block = site.pos.block;  // the outer %n parallel-loop body
  mlir::scf::ForOp old_loop = reduction.loop;  // reused %m loop (mutable handle)
  mlir::Location loc = old_loop.getLoc();

  LLVM_DEBUG(llvm::dbgs()
             << "Creating reduction ktdf.pipeline (outer wrapper)\n");

  // FIFO slot / accumulator granularity = the reduction lane count (product of
  // the reduction linalg's result shape, e.g. 64).
  auto acc_tensor_ty =
      mlir::cast<mlir::RankedTensorType>(reduction.linalg->getResult(0).getType());
  mlir::Type elem_ty = acc_tensor_ty.getElementType();
  total_num_elements_ = 1;
  for (int64_t d : acc_tensor_ty.getShape()) total_num_elements_ *= d;

  // The 2-D accumulator memref shape: 1 x lanes.
  llvm::SmallVector<int64_t> acc_shape{1, total_num_elements_};

  // The reduction accumulator lives in the register file. Mark it with the
  // "lrfreg" memory space so the backend treats it as register-file-resident
  // (an unconditional-reset peel target) rather than an HBM buffer.
  // TODO: the residency should be analysis-determined; hardcoded for now.
  auto acc_memref_ty = mlir::MemRefType::get(
      acc_shape, elem_ty, mlir::MemRefLayoutAttrInterface{},
      mlir::StringAttr::get(&getContext(), "lrfreg"));

  // FIFO attributes for the (single) load and store.
  assert(spec.loads.size() == 1 && spec.stores.size() == 1 &&
         "reduction group expects one load and one store");
  auto [load_src, load_dst] = getFifoAttributesForLoad(spec.loads.front());
  auto [store_src, store_dst] = getFifoAttributesForStore(spec.stores.front());
  auto load_fifo_ty = mlir::ktdf::FifoSlotType::get(
      &getContext(), load_src, load_dst, total_num_elements_, elem_ty);
  auto store_fifo_ty = mlir::ktdf::FifoSlotType::get(
      &getContext(), store_src, store_dst, total_num_elements_, elem_ty);
  mlir::ktdf::TokenType token_ty = mlir::ktdf::TokenType::get(&getContext());

  // The reduction address policy (map/sizes from the access tile, memref = the
  // full view). Must outlive the emit below.
  SourceMapFromAccessTile red_source_map;
  SizesFromTileShape red_sizes;
  MemrefFullView red_memref;
  red_memref.resolve = [this](mlir::OpBuilder& b, mlir::Location l,
                              mlir::ktdp::ConstructAccessTilesOp at) {
    return materializeFullView(b, l, at);
  };
  PipelineSpec red_spec = spec;
  red_spec.linalgOp = reduction.linalg;
  red_spec.mapBuilder = &red_source_map;
  red_spec.sizeBuilder = &red_sizes;
  red_spec.memrefBuilder = &red_memref;

  // createDataTransfers asserts a single fused compute op; the reduction linalg
  // plays that role for its per-operand indexing (the reduction map builder
  // ignores it, but the assert / spec.linalgOp must stay consistent).
  compute_ops_.assign({reduction.linalg});

  // Reuse the %m loop: build a fresh iter_arg-free scf.for with the same bounds
  // at the old loop's position, redirect the old IV to the new one so the load
  // access-tile subscript picks up the new IV, and schedule the old loop (dead)
  // for cleanup.
  //
  // Build the ktdf.pipeline at the top of the %n body.
  mlir::OpBuilder builder(body_block, body_block->begin());

  mlir::ktdf::PipelineOp::create(builder, loc, [&](mlir::OpBuilder& b,
                                                   mlir::Location l) {
    // OUTER private: acc memref + store FIFO + 2 tokens (t0, t1).
    auto outer_private = mlir::ktdf::PrivateOp::create(
        b, l,
        mlir::TypeRange{acc_memref_ty, store_fifo_ty,
                        token_ty, token_ty});
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&outer_private.getRegion().front());
      auto acc = mlir::memref::AllocOp::create(b, l, acc_memref_ty);
      auto store_fifo = mlir::ktdf::FifoAllocateOp::create(
          b, l, mlir::TypeRange{store_fifo_ty}, mlir::ValueRange{});
      auto t0 = mlir::ktdf::CreateTokenOp::create(b, l, token_ty);
      auto t1 = mlir::ktdf::CreateTokenOp::create(b, l, token_ty);
      mlir::ktdf::PrivateYieldOp::create(
          b, l,
          mlir::ValueRange{acc.getResult(), store_fifo.getResult(0),
                           t0.getResult(), t1.getResult()});
    }
    mlir::Value acc = outer_private.getResult(0);
    mlir::Value store_fifo = outer_private.getResult(1);
    mlir::Value t0 = outer_private.getResult(2);
    mlir::Value t1 = outer_private.getResult(3);

    // ACCUMULATE stage: depends_out(t0), UNIT-LESS. Body = reused %m loop +
    // post-loop drain of the settled accumulator into the store FIFO.
    mlir::ktdf::StageOp::create(
        b, l, mlir::ValueRange{}, mlir::ValueRange{t0},
        [&](mlir::OpBuilder& sb, mlir::Location sl) {
          // Fresh iter_arg-free %m loop.
          auto new_loop = mlir::scf::ForOp::create(
              sb, sl, old_loop.getLowerBound(), old_loop.getUpperBound(),
              old_loop.getStep());
          new_loop->setAttr(
              "loop_type",
              mlir::ktdf::LoopTypeAttr::get(&getContext(),
                                            mlir::ktdf::LoopType::ReductionLoop));
          // Redirect old IV -> new IV so the load access-tile subscript is the
          // new loop's induction variable.
          old_loop.getInductionVar().replaceAllUsesWith(
              new_loop.getInductionVar());

          mlir::OpBuilder lb(new_loop.getBody(), new_loop.getBody()->begin());
          // INNER pipeline: inner private (load FIFO + 2 tokens) + inner load
          // stage + inner compute RMW stage.
          mlir::ktdf::PipelineOp::create(
              lb, sl, [&](mlir::OpBuilder& ib, mlir::Location il) {
                auto inner_private = mlir::ktdf::PrivateOp::create(
                    ib, il, mlir::TypeRange{load_fifo_ty, token_ty, token_ty});
                {
                  mlir::OpBuilder::InsertionGuard g(ib);
                  ib.setInsertionPointToStart(
                      &inner_private.getRegion().front());
                  auto load_fifo = mlir::ktdf::FifoAllocateOp::create(
                      ib, il, mlir::TypeRange{load_fifo_ty}, mlir::ValueRange{});
                  auto it0 = mlir::ktdf::CreateTokenOp::create(ib, il, token_ty);
                  auto it1 = mlir::ktdf::CreateTokenOp::create(ib, il, token_ty);
                  mlir::ktdf::PrivateYieldOp::create(
                      ib, il,
                      mlir::ValueRange{load_fifo.getResult(0), it0.getResult(),
                                       it1.getResult()});
                }
                mlir::Value load_fifo = inner_private.getResult(0);
                mlir::Value it0 = inner_private.getResult(1);
                mlir::Value it1 = inner_private.getResult(2);

                // Inner LOAD stage: depends_out(it0), UNIT-LESS. Reuses the
                // policy-free createDataTransfers with the reduction address
                // spec (rank-3 source_map, sizes [1,1,64], subscripts %n,%m,%c0).
                mlir::ktdf::StageOp::create(
                    ib, il, mlir::ValueRange{}, mlir::ValueRange{it0},
                    [&](mlir::OpBuilder& stb, mlir::Location stl) {
                      createDataTransfers(stb, stl, inner_private, red_spec,
                                          tile_sizes_, /*is_load=*/true,
                                          /*private_result_offset=*/0);
                    });

                // Inner COMPUTE RMW stage: depends_in(it0), depends_out(it1),
                // applicable_units=["SFP"]. Rebuild the linalg at 2-D 1xlanes.
                auto compute_stage = mlir::ktdf::StageOp::create(
                    ib, il, mlir::ValueRange{it0}, mlir::ValueRange{it1},
                    [&](mlir::OpBuilder& cb, mlir::Location cl) {
                      auto read = mlir::ktdf::ReadFromFifoOp::create(
                          cb, cl,
                          mlir::RankedTensorType::get(acc_shape, elem_ty),
                          load_fifo);
                      auto cur = mlir::bufferization::ToTensorOp::create(
                          cb, cl,
                          mlir::RankedTensorType::get(acc_shape, elem_ty), acc,
                          /*restrict=*/true, /*writable=*/false);
                      // 2-D identity maps, both operands parallel.
                      mlir::AffineMap id2 = mlir::AffineMap::getMultiDimIdentityMap(
                          2, &getContext());
                      llvm::SmallVector<mlir::AffineMap> maps{id2, id2};
                      llvm::SmallVector<mlir::utils::IteratorType> iters{
                          mlir::utils::IteratorType::parallel,
                          mlir::utils::IteratorType::parallel};
                      auto generic = mlir::linalg::GenericOp::create(
                          cb, cl,
                          mlir::TypeRange{
                              mlir::RankedTensorType::get(acc_shape, elem_ty)},
                          mlir::ValueRange{read.getResult()},
                          mlir::ValueRange{cur.getResult()}, maps, iters);
                      {
                        // Guard so that building the generic's body block does
                        // not leave the insertion point inside the generic
                        // region. The materialize below must be a sibling of the
                        // generic in the stage body, not nested in its region.
                        mlir::OpBuilder::InsertionGuard g(cb);
                        mlir::Block* body = cb.createBlock(
                            &generic.getRegion(), generic.getRegion().end(),
                            mlir::TypeRange{elem_ty, elem_ty},
                            llvm::SmallVector<mlir::Location>{cl, cl});
                        cb.setInsertionPointToStart(body);
                        auto sum = mlir::arith::AddFOp::create(
                            cb, cl, body->getArgument(0), body->getArgument(1));
                        mlir::linalg::YieldOp::create(cb, cl,
                                                      sum.getResult());
                      }
                      mlir::bufferization::MaterializeInDestinationOp::create(
                          cb, cl, /*result=*/mlir::Type(), generic.getResult(0),
                          acc, /*restrict=*/false, /*writable=*/true);
                    });
                compute_stage.setApplicableUnitsAttr(
                    ib.getArrayAttr(resource_kinds_->getComputeKind()));
              });
          // scf.for auto-terminator (no results) already inserted.

          // After the %m loop: read the settled accumulator and write it once
          // into the store FIFO.
          sb.setInsertionPointAfter(new_loop);
          auto settled = mlir::bufferization::ToTensorOp::create(
              sb, sl, mlir::RankedTensorType::get(acc_shape, elem_ty), acc,
              /*restrict=*/true, /*writable=*/false);
          mlir::ktdf::WriteToFifoOp::create(sb, sl, settled.getResult(),
                                            store_fifo);
        });

    // STORE stage: depends_in(t0), depends_out(t1), UNIT-LESS. Reuses
    // createDataTransfers for the rank-2 FIFO->view store.
    mlir::ktdf::StageOp::create(
        b, l, mlir::ValueRange{t0}, mlir::ValueRange{t1},
        [&](mlir::OpBuilder& stb, mlir::Location stl) {
          // The store FIFO sits at outer_private result index loads.size()
          // (== 1), exactly where createDataTransfers looks for it.
          createDataTransfers(stb, stl, outer_private, red_spec, tile_sizes_,
                              /*is_load=*/false,
                              /*private_result_offset=*/spec.loads.size());
        });
  });

  // The reused old loop is now dead; schedule it (and its access tiles / store)
  // for cleanup.
  ops_to_delete_.push_back(old_loop.getOperation());
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

void ConstructThreeStagePipelinePass::createDataTransfers(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf::PrivateOp private_op, const PipelineSpec& spec,
    llvm::ArrayRef<int64_t> tile_sizes, bool is_load,
    size_t private_result_offset) {
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
  size_t op_count = is_load ? spec.loads.size() : spec.stores.size();

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
  mlir::linalg::LinalgOp linalg_op = spec.linalgOp;

  // Create data_transfer for each operation
  for (size_t i = 0; i < op_count; ++i) {
    // Build the address context for this transfer and resolve the source_map,
    // subscript IVs, and sizes through the injected composable blocks. The
    // offset is NOT computed here (stays in computeReinterpretCastOffset).
    mlir::Value access_tile_value;
    AddrCtx ctx;
    ctx.linalgOp = linalg_op;
    ctx.is_load = is_load;
    ctx.tiledLoopIVs = loop_ivs;
    ctx.tileSizes = tile_sizes;
    if (is_load) {
      mlir::ktdp::LoadOp load_op = spec.loads[i];
      access_tile_value = load_op.getAccessTile();
      ctx.loadOp = load_op;
      ctx.errAnchor = load_op.getOperation();
    } else {
      mlir::ktdp::StoreOp store_op = spec.stores[i];
      access_tile_value = store_op.getAccessTile();
      ctx.storeOp = store_op;
      ctx.errAnchor = store_op.getOperation();
    }
    ctx.accessTile =
        access_tile_value.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();

    // Resolve source_map + subscript IVs + sizes + the memref value through the
    // injected address policy. Elementwise yields the raw access tile (step 5
    // rewrites it into a reinterpret_cast); reduction yields the full memory
    // view. createDataTransfers itself is policy-free.
    AddressParts address;
    address.sourceMemref = access_tile_value;
    AddressStatus status =
        planAddress(builder, loc, ctx, *spec.mapBuilder, *spec.sizeBuilder,
                    *spec.memrefBuilder, address);
    if (status == AddressStatus::NoSourceMap) {
      // Same diagnostic/anchor as baseline.
      ctx.errAnchor->emitError(
          "could not locate matching linalg operand to project loop IVs and "
          "tile sizes through; the data_transfer rank would not match the "
          "underlying memref");
      signalPassFailure();
      return;
    }
    if (status == AddressStatus::SizesFailed) {
      // The sizes builder already emitted its diagnostic.
      signalPassFailure();
      return;
    }

    // Get the FIFO slot from ktdf.private results.
    mlir::Value fifo_slot = private_op.getResult(private_result_offset + i);

    // The fifo side gets a null AffineMap; the access-tile (memref) side
    // gets the resolved source_map. subscriptIVs feed the map's dim inputs.
    mlir::AffineMap null_map;
    if (is_load) {
      mlir::ktdf::DataTransferOp::create(
          builder, loc, address.sourceMemref, address.sourceMap,
          address.subscriptIVs, address.sizes, fifo_slot, null_map,
          mlir::ValueRange{}, fifo_sizes);
    } else {
      mlir::ktdf::DataTransferOp::create(
          builder, loc, fifo_slot, null_map, mlir::ValueRange{}, fifo_sizes,
          address.sourceMemref, address.sourceMap, address.subscriptIVs,
          address.sizes);
    }
  }
}

mlir::Value ConstructThreeStagePipelinePass::materializeFullView(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdp::ConstructAccessTilesOp access_tile) {
  mlir::Value memory_view = access_tile.getBase();
  mlir::Operation* view_op = memory_view.getDefiningOp();
  assert(view_op && "access tile base must have a defining op");

  auto cached = full_view_cache_.find(view_op);
  if (cached != full_view_cache_.end()) return cached->second;

  auto view_type = mlir::cast<mlir::MemRefType>(memory_view.getType());
  auto construct_mem_view =
      mlir::cast<mlir::ktdp::ConstructMemoryViewOp>(view_op);
  mlir::Attribute mapped_memory_space =
      mapMemorySpace(construct_mem_view.getMemorySpaceAttr());

  // Insert the cast at the function-constant point so it dominates every
  // enclosing loop, right after the view op (which the const builder keeps at
  // the function head).
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(view_op);
  mlir::MemRefType cast_type = mlir::MemRefType::get(
      view_type.getShape(), view_type.getElementType(), view_type.getLayout(),
      mapped_memory_space);
  auto cast = mlir::memref::MemorySpaceCastOp::create(builder, loc, cast_type,
                                                      memory_view);

  // Uniform view invariant: every memory view feeding a data_transfer is a
  // construct_memory_view -> memory_space_cast -> reinterpret_cast chain, so
  // the backend LogicalMemoryViewBuilder sees one chain shape. For the whole
  // view the reinterpret_cast is an identity slice: offset 0, full shape, full
  // strides. The per-tile subscript rides on the data_transfer source_map, not
  // on the view. The explicit strided layout (offset 0) makes the result type
  // differ from the memory_space_cast result type, so the identity cast is not
  // folded away as a no-op by canonicalization.
  llvm::SmallVector<int64_t> view_strides = getStridesFromMemRefType(view_type);
  llvm::SmallVector<mlir::OpFoldResult> rc_sizes;
  for (int64_t dim : view_type.getShape())
    rc_sizes.push_back(builder.getIndexAttr(dim));
  llvm::SmallVector<mlir::OpFoldResult> rc_strides;
  for (int64_t stride : view_strides)
    rc_strides.push_back(builder.getIndexAttr(stride));
  mlir::StridedLayoutAttr rc_layout = mlir::StridedLayoutAttr::get(
      builder.getContext(), /*offset=*/0, view_strides);
  mlir::MemRefType rc_type = mlir::MemRefType::get(
      view_type.getShape(), view_type.getElementType(), rc_layout,
      mapped_memory_space);
  auto rc = mlir::memref::ReinterpretCastOp::create(
      builder, loc, rc_type, cast.getResult(),
      /*offset=*/builder.getIndexAttr(0), rc_sizes, rc_strides);

  full_view_cache_[view_op] = rc.getResult();
  return rc.getResult();
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

  // Single top-level dispatch: a reduction group reuses the pre-existing loop
  // nest (no tiling) and emits the nested reduction pipeline; the elementwise
  // path is 100% unchanged below.
  std::optional<ReductionInfo> reduction = findReductionLoop(func_op);
  is_reduction_ = static_cast<bool>(reduction);

  if (is_reduction_) {
    LLVM_DEBUG(llvm::dbgs()
               << "  dispatch: reduction group (reusing loop nest, no tiling)\n");
    PipelineSite site;
    // The site sits in the block that holds the reused %m loop (the outer
    // parallel-loop body).
    site.pos.block = reduction->loop->getBlock();
    site.spec.loads = load_ops_;
    site.spec.stores = store_ops_;
    site.spec.linalgOp = reduction->linalg;
    site.spec.accumulatorBuf = {};
    instantiateReductionPipeline(site, *reduction);

    DEBUG_WITH_TYPE(VerboseDebug,
                    llvm::dbgs() << "After accumulator collapse / pipeline "
                                    "creation:\n"
                                 << func_op << "\n\n");

    // Reduction access tiles are addressed via the full view, not
    // reinterpret_cast; they are cleaned up as dead below.
    cleanupOperations();

    DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After cleanup:\n"
                                               << func_op << "\n\n");
    return;
  }

  // Step 3: Create loops from linalg operations
  createLoopsFromLinalg(linalg_ops);

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After loops created:\n"
                                             << func_op << "\n\n");

  // Step 4: Derive the pipeline site(s) and instantiate them uniformly.
  if (!tiled_loops_.empty()) {
    auto innermost_loop = llvm::cast<mlir::scf::ForOp>(tiled_loops_.back());

    // The fused compute op whose indexing maps describe each transfer.
    mlir::linalg::LinalgOp fused_generic;
    innermost_loop.getBody()->walk([&](mlir::linalg::LinalgOp linalg_op) {
      fused_generic = linalg_op;
    });

    // Elementwise composable address blocks (must outlive instantiatePipeline).
    SourceMapFromLinalg elementwiseSourceMap;
    SizesByProjection elementwiseSizes;
    MemrefPassthrough elementwiseMemref;

    // The loop-nest half of the func plan: the manufactured tiling nest are the
    // parallel loops; there are no reduction loops on the elementwise path.
    StructuralInfo structural;
    for (auto* loop_op : tiled_loops_) {
      structural.parallelLoops.push_back(
          llvm::cast<mlir::scf::ForOp>(loop_op));
    }

    // One site per compute group. Elementwise: pos = innermost tiled loop body,
    // accumulatorBuf null, reductionLoops empty.
    llvm::SmallVector<PipelineSite> sites;
    PipelineSite site;
    site.pos.block = innermost_loop.getBody();
    site.spec.loads = load_ops_;
    site.spec.stores = store_ops_;
    site.spec.linalgOp = fused_generic;
    site.spec.accumulatorBuf = {};
    site.spec.mapBuilder = &elementwiseSourceMap;
    site.spec.sizeBuilder = &elementwiseSizes;
    site.spec.memrefBuilder = &elementwiseMemref;
    sites.push_back(site);

    LLVM_DEBUG(llvm::dbgs()
               << "Derived " << sites.size() << " pipeline site(s); "
               << structural.parallelLoops.size() << " parallel loop(s), "
               << structural.reductionLoops.size() << " reduction loop(s)\n");

    for (const PipelineSite& s : sites) instantiatePipeline(s, structural);
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
