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
//
// DataTransferAlignment: make strided loads from global views contiguous.
//
// A transpose or permute of stick-tiled tensors reaches this pass as a copy
// from a global view x to a global view y through two per-core staging
// buffers, A and B, one stick of y per trip of a loop i:
//
//   stage L3LU:  for i: x[p = i, q = q0 : +64]  -> A[i]
//   stage ...:   for i: A[i] -> FIFO -> ... -> FIFO -> B[i]
//   stage L3SU:  for i: B[i] -> y[p = i, q = q0 : +64]
//
// A stick is a run of `block` elements, the default compute unit's SIMD lanes
// for the element type; q is the dim each transfer moves its stick along, and
// p is x's stride-1 dim. When x's stride along q is not 1, each load gathers
// its stick from `block` places in memory.
//
// The pass rewrites two transfers, each in place:
//
//   - the load reads stick i of x along p instead:
//       x[p = 0 : +64, q = q0 + i] -> A[i]
//   - the hop that reads A gets a loop c over q. Its read of A becomes a splat
//     of element i of A[c], and its write to B an extract into lane c of B[i]:
//       for i: for c: A[c][i] -> FIFO (splat) -> ... -> B[i][c] (extract)
//
// A[c][i] = x[p = i, q = q0 + c], which is element c of stick i of y, so B
// ends up holding what it held before. B, the store and every loop are left
// as they are.
//
// This needs the loop over p to walk exactly one stick, 0 to `block`. Tiling
// of that loop is not supported yet: when loop tiling has tiled it, the load
// is left strided, with a debug message.
//
// Only a load that feeds a plain copy is rewritten: A has no reader but the
// hop, and the hop only passes the stick through. A strided load feeding
// anything else (x + y^T) needs a transpose stage of its own, which this pass
// does not build; it is an error.
//
// Ordering: after stage-coarsening + canonicalize, before double-buffering.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define PASS_NAME "data-transfer-alignment"
#define DEBUG_TYPE PASS_NAME

namespace scheduler {
#define GEN_PASS_DEF_DATATRANSFERALIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Whether `value` is a memref in a core's own scratchpad (a staging buffer).
bool isScratchpad(mlir::Value value,
                  const scheduler::arch_view::MemoryTree& memory_tree) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type) return false;
  auto kind =
      mlir::dyn_cast_or_null<mlir::ktdf_arch::KindAttr>(type.getMemorySpace());
  return kind && memory_tree.isPerCoreScratchPadMemory(kind);
}

/// The index constant `value` at the start of `block`, reusing one already
/// there so each value is defined once.
mlir::Value indexConstant(mlir::Block* block, int64_t value,
                          mlir::PatternRewriter& rewriter) {
  for (auto cst : block->getOps<mlir::arith::ConstantIndexOp>())
    if (cst.value() == value) return cst;
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(block);
  return mlir::arith::ConstantIndexOp::create(rewriter, rewriter.getUnknownLoc(),
                                              value);
}

/// The one user of `value` other than `user`, if it is an `OpT`; null when
/// there is none, more than one, or it is another op.
template <typename OpT>
OpT soleOtherUser(mlir::Value value, mlir::Operation* user) {
  OpT other;
  for (mlir::Operation* op : value.getUsers()) {
    if (op == user) continue;
    if (other && op != other.getOperation()) return {};
    other = mlir::dyn_cast<OpT>(op);
    if (!other) return {};
  }
  return other;
}

/// The dim a transfer side moves its elements along: the last one whose size
/// is not 1. nullopt when every size is 1.
std::optional<int64_t> movingDim(llvm::ArrayRef<mlir::OpFoldResult> sizes) {
  std::optional<int64_t> dim;
  for (auto [d, size] : llvm::enumerate(sizes))
    if (mlir::getConstantIntValue(size) != 1) dim = d;
  return dim;
}

/// The operand dim `dim` of a transfer side is indexed by, when the map gives
/// it a plain operand; null otherwise.
mlir::Value indexAt(mlir::AffineMap map, mlir::ValueRange indices,
                    int64_t dim) {
  auto expr = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(dim));
  return expr ? indices[expr.getPosition()] : mlir::Value();
}

/// Whether `value` is the IV of a loop from 0 to `extent` with step 1.
bool walksOneStick(mlir::Value value, int64_t extent) {
  auto loop = mlir::scf::getForInductionVarOwner(value);
  return loop && mlir::isConstantIntValue(loop.getLowerBound(), 0) &&
         mlir::isConstantIntValue(loop.getUpperBound(), extent) &&
         mlir::isConstantIntValue(loop.getStep(), 1);
}

/// One trip of the hop as a single element: dim `stick` of the A or B side
/// takes the index of `from` (or 0), dim `from` takes `c`, every size is 1.
/// `indices` gains `c` as a new last operand.
mlir::AffineMap oneElement(mlir::AffineMap map, int64_t from, int64_t stick,
                           unsigned c_position) {
  llvm::SmallVector<mlir::AffineExpr> results(map.getResults());
  mlir::AffineExpr c = mlir::getAffineDimExpr(c_position, map.getContext());
  results[stick] = from == stick ? c : results[from];
  if (from != stick) results[from] = c;
  return mlir::AffineMap::get(c_position + 1, 0, results, map.getContext());
}

/// Re-sticks a strided load from a global view and the hop that reads its
/// staging buffer; see the file comment.
struct RestickLoad : mlir::OpRewritePattern<mlir::ktdf::DataTransferOp> {
  RestickLoad(mlir::MLIRContext* ctx,
              const scheduler::arch_view::MemoryTree& memory_tree,
              mlir::ktdf_arch::ExecutionUnitOp compute, bool& failed)
      : OpRewritePattern(ctx),
        memory_tree(memory_tree),
        compute(compute),
        failed(failed) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::DataTransferOp load,
      mlir::PatternRewriter& rewriter) const override {
    if (!load.isSourceMemRef() || !load.isDestMemRef() ||
        isScratchpad(load.getSource(), memory_tree) ||
        !isScratchpad(load.getDestination(), memory_tree))
      return mlir::failure();
    auto x = mlir::cast<mlir::MemRefType>(load.getSource().getType());
    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    std::optional<int64_t> q = movingDim(load.getMixedSourceSizes());
    if (!q || mlir::failed(x.getStridesAndOffset(strides, offset)) ||
        strides[*q] == 1)
      return mlir::failure();  // already moves a contiguous stick

    // From here on the load needs re-sticking: anything this pattern cannot
    // rewrite is an error, except a tiled loop over p.
    auto fail = [&](mlir::Operation* at, const llvm::Twine& msg) {
      failed = true;
      at->emitError(PASS_NAME ": " + msg);
      return mlir::failure();
    };
    std::optional<int64_t> p;
    for (auto [dim, stride] : llvm::enumerate(strides))
      if (stride == 1 && x.getDimSize(dim) != 1) p = dim;
    if (!p) return fail(load, "the source view has no dim with stride 1");

    // Only fp16 sticks have been run on the device; other element sizes need
    // their own transfer modes checked first.
    auto element_bytes = scheduler::tryGetSizeInBytes(x.getElementType());
    if (!element_bytes || *element_bytes != 2)
      return fail(load, "only 2-byte elements are supported");
    const int64_t block =
        compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(
            x.getElementType());
    for (auto [dim, size] : llvm::enumerate(load.getMixedSourceSizes()))
      if (mlir::getConstantIntValue(size) != ((int64_t)dim == *q ? block : 1))
        return fail(load, "the load does not move one stick of " +
                              llvm::Twine(block) + " elements along dim " +
                              llvm::Twine(*q));

    // The loop over p must walk exactly one stick.
    mlir::AffineMap x_map = *load.getSourceMap();
    mlir::Value iv = indexAt(x_map, load.getSourceIndices(), *p);
    if (!iv || !walksOneStick(iv, block)) {
      LDBG() << "skipping a strided load: dim " << *p
             << " is not walked by a loop from 0 to " << block
             << " (tiling of the stick dim is not supported yet): " << load;
      return mlir::failure();
    }

    // A's loop dim the loop over p indexes.
    std::optional<int64_t> kp;
    for (int64_t k = 0; k < load.getDestMap()->getNumResults(); ++k)
      if (indexAt(*load.getDestMap(), load.getDestIndices(), k) == iv) kp = k;
    if (!kp)
      return fail(load,
                  "no dim of the staging buffer is indexed by the loop over "
                  "dim " +
                      llvm::Twine(*p));

    // The plain copy A -> FIFO, passed through unchanged, -> B.
    auto not_a_copy = [&]() {
      return fail(load, "the load does not feed a plain copy (A -> FIFO, "
                        "read_from_fifo, write_to_fifo, FIFO -> B, with "
                        "nothing else reading A or the FIFOs); re-sticking it "
                        "needs its own transpose stage, which is not "
                        "supported");
    };
    auto from_a =
        soleOtherUser<mlir::ktdf::DataTransferOp>(load.getDestination(), load);
    if (!from_a || from_a.getSource() != load.getDestination() ||
        !from_a.isDestFifo())
      return not_a_copy();
    mlir::Value fifo = from_a.getDestination();
    auto into_b = soleOtherUser<mlir::ktdf::DataTransferOp>(fifo, from_a);
    if (!into_b) {
      auto read = soleOtherUser<mlir::ktdf::ReadFromFifoOp>(fifo, from_a);
      if (!read || !read->hasOneUse()) return not_a_copy();
      auto write =
          mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(*read->getUsers().begin());
      if (!write || write.getData() != read.getResult()) return not_a_copy();
      fifo = write.getFifoSlot();
      into_b = soleOtherUser<mlir::ktdf::DataTransferOp>(fifo, write);
    }
    if (!into_b || into_b.getSource() != fifo || !into_b.isDestMemRef() ||
        !isScratchpad(into_b.getDestination(), memory_tree))
      return not_a_copy();
    auto hop = from_a->getParentOfType<mlir::ktdf::PipelineOp>();
    if (into_b->getParentOfType<mlir::ktdf::PipelineOp>() != hop)
      return fail(from_a, "the copy's A -> B hop is not one pipeline");

    // The hop's own loop over p, and the dims it moves its stick along.
    std::optional<int64_t> a_stick = movingDim(from_a.getMixedSourceSizes());
    std::optional<int64_t> b_stick = movingDim(into_b.getMixedDestSizes());
    mlir::Value hop_iv =
        indexAt(*from_a.getSourceMap(), from_a.getSourceIndices(), *kp);
    if (!a_stick || !b_stick || !hop_iv || !walksOneStick(hop_iv, block))
      return fail(from_a,
                  "the hop does not move one stick of A per trip of a loop "
                  "from 0 to " +
                      llvm::Twine(block));

    // The load reads stick i of x along p: dim q takes i, dim p starts at 0.
    llvm::SmallVector<mlir::AffineExpr> x_results(x_map.getResults());
    x_results[*q] = x_results[*q] + x_results[*p];
    x_results[*p] = rewriter.getAffineConstantExpr(0);
    llvm::SmallVector<int64_t> x_sizes(*load.getStaticSourceSizes());
    std::swap(x_sizes[*p], x_sizes[*q]);
    rewriter.modifyOpInPlace(load, [&] {
      load.setSourceMap(mlir::AffineMap::get(x_map.getNumDims(), 0, x_results,
                                             rewriter.getContext()));
      load.setStaticSourceSizes(x_sizes);
    });

    // The hop moves element hop_iv of A[c] into lane c of B[hop_iv].
    mlir::Block* entry =
        &load->getParentOfType<mlir::func::FuncOp>().getBody().front();
    rewriter.setInsertionPoint(hop);
    auto c_loop = mlir::scf::ForOp::create(
        rewriter, hop.getLoc(), indexConstant(entry, 0, rewriter),
        indexConstant(entry, block, rewriter), indexConstant(entry, 1, rewriter));
    rewriter.moveOpBefore(hop, c_loop.getBody()->getTerminator());
    mlir::Value c = c_loop.getInductionVar();

    rewriter.modifyOpInPlace(from_a, [&] {
      mlir::AffineMap map = *from_a.getSourceMap();
      from_a.setSourceMap(
          oneElement(map, *kp, *a_stick, from_a.getSourceIndices().size()));
      from_a.getSourceIndicesMutable().append(c);
      from_a.getSourceSizesMutable().clear();
      from_a.setStaticSourceSizes(
          llvm::SmallVector<int64_t>(map.getNumResults(), 1));
      from_a->setDiscardableAttr("transfer_mode",
                                 rewriter.getStringAttr("splat"));
    });
    rewriter.modifyOpInPlace(into_b, [&] {
      mlir::AffineMap map = *into_b.getDestMap();
      into_b.setDestMap(
          oneElement(map, *b_stick, *b_stick, into_b.getDestIndices().size()));
      into_b.getDestIndicesMutable().append(c);
      into_b.getDestSizesMutable().clear();
      into_b.setStaticDestSizes(
          llvm::SmallVector<int64_t>(map.getNumResults(), 1));
      into_b->setDiscardableAttr("transfer_mode",
                                 rewriter.getStringAttr("extract"));
    });
    return mlir::success();
  }

  const scheduler::arch_view::MemoryTree& memory_tree;
  mutable mlir::ktdf_arch::ExecutionUnitOp compute;
  bool& failed;
};

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* device = device_manager.getOrImportDevice();
    if (!device) {
      getOperation()->emitError(PASS_NAME ": failed to import device spec");
      return signalPassFailure();
    }
    auto& memory_tree =
        device_manager.getOrCreateView<scheduler::arch_view::MemoryTree>(
            *device);
    const auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
    if (!default_device) return signalPassFailure();
    mlir::ktdf_arch::Mapping mapping(default_device.getRef());
    auto compute = mapping.byKind().getDefaultCompute();
    if (!compute) {
      getOperation()->emitError(PASS_NAME
                                ": no (unambiguous) default compute resource");
      return signalPassFailure();
    }

    bool failed = false;
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RestickLoad>(&getContext(), memory_tree, compute, failed);
    mlir::walkAndApplyPatterns(getOperation(), std::move(patterns));
    if (failed) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
