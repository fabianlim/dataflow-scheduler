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
// DataTransferAlignment: make strided copies between global views legal.
//
// A transpose or permute of stick-tiled tensors reaches this pass as a copy
// from a global view x to a global view y through two per-core staging
// buffers, A and B:
//
//   stage L3LU:  x  -> A     one stick of y per transfer
//   stage ...:   A  -> B     a nested pipeline moving one stick at a time
//   stage L3SU:  B  -> y     one stick of y per transfer
//
// Every transfer carries one index per view dim, and x and y index the same
// loop dims in the same order. A stick is a run of `block` contiguous
// elements, the default compute unit's SIMD lanes for the element type. Each
// transfer moves one stick along dim q, y's stride-1 dim. Let p be x's
// stride-1 dim.
//
//   - p == q: x's sticks are y's sticks. Each transfer already moves a whole
//     contiguous stick; nothing is widened.
//   - p != q: a stick of x runs along p, a stick of y along q. The pass widens
//     the copy to `block` x `block` tiles, one stick along p and along q: the
//     load and store move a whole tile, the point loops over dim p are
//     removed, and the nested pipeline transposes each tile element by
//     element (splat out of A, extract into B). The views are not rewritten.
//
// A transfer pairs the elements of its two sides in stride order, stride-1
// dim fastest. The widened load from x then fills A with the tile transposed,
// and the splat/extract transposes it back.
//
// Only a pure copy is widened in place: A, B and the FIFOs between them must
// have no other reader. A strided load that feeds anything else (x + y^T)
// needs its own transpose stage, which this pass does not build; it is an
// error.
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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "data-transfer-alignment"
#define DEBUG_TYPE PASS_NAME

namespace scheduler {
#define GEN_PASS_DEF_DATATRANSFERALIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

//===----------------------------------------------------------------------===//
// Views
//===----------------------------------------------------------------------===//

/// Whether `value` is a memref in a core's own scratchpad (a staging buffer).
bool isScratchpad(mlir::Value value,
                  const scheduler::arch_view::MemoryTree& memory_tree) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type) return false;
  auto kind =
      mlir::dyn_cast_or_null<mlir::ktdf_arch::KindAttr>(type.getMemorySpace());
  return kind && memory_tree.isPerCoreScratchPadMemory(kind);
}

/// The static shape and strides of the memref `value`; failure if it is not
/// a memref, its layout has no strides, or any of them is dynamic.
struct Layout {
  llvm::SmallVector<int64_t> sizes, strides;
};
mlir::FailureOr<Layout> layoutOf(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape()) return mlir::failure();
  Layout layout{llvm::to_vector(type.getShape()), {}};
  int64_t offset;
  if (mlir::failed(type.getStridesAndOffset(layout.strides, offset)) ||
      llvm::any_of(layout.strides, mlir::ShapedType::isDynamic))
    return mlir::failure();
  return layout;
}

/// The dim with stride 1 and more than one element, if any. A size-1 dim is
/// skipped: its stride never multiplies a non-zero index.
std::optional<int64_t> unitStrideDim(const Layout& layout) {
  for (auto [dim, size_stride] :
       llvm::enumerate(llvm::zip(layout.sizes, layout.strides))) {
    auto [size, stride] = size_stride;
    if (stride == 1 && size != 1) return dim;
  }
  return std::nullopt;
}

/// The index constant `value` at the start of `block`, reusing one already
/// there so each value is defined once.
mlir::Value indexConstant(mlir::Block* block, int64_t value,
                          mlir::OpBuilder& builder) {
  for (auto cst : block->getOps<mlir::arith::ConstantIndexOp>())
    if (cst.value() == value) return cst;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  return mlir::arith::ConstantIndexOp::create(
      builder, builder.getUnknownLoc(), value);
}

//===----------------------------------------------------------------------===//
// Finding a re-stick
//===----------------------------------------------------------------------===//

/// One strided copy x -> y found in a ktdf.pipeline, and what widening it
/// changes. Built by `findRestick`, consumed by `widenRestick`.
struct Restick {
  mlir::ktdf::DataTransferOp load;    // x -> A
  mlir::ktdf::DataTransferOp from_a;  // A -> FIFO, in a nested pipeline
  mlir::ktdf::DataTransferOp into_b;  // FIFO -> B, in the same one
  mlir::ktdf::DataTransferOp store;   // B -> y
  int64_t p = 0;      // x's stride-1 dim; never q
  int64_t q = 0;      // y's stride-1 dim, the one each transfer moves along
  int64_t rank = 0;   // rank of x's and y's views
};

/// Elements in one stick of `transfer`'s element type: the SIMD lanes of
/// `compute` for it; 0 if it has none.
int64_t stickSize(mlir::ktdf::DataTransferOp transfer,
                  mlir::ktdf_arch::ExecutionUnitOp compute) {
  mlir::Type element_type =
      mlir::cast<mlir::MemRefType>(transfer.getSource().getType())
          .getElementType();
  return compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(
      element_type);
}

/// What indexes slot p of the load: the IV of the loop that walks dim p one
/// element at a time (the point loop), either directly or through
/// `linearize_index [tile_iv : tile], [point_iv : 1]`, whose `tile` is the
/// reserve_size placeholder of a tiled loop over dim p. Null fields when the
/// slot has neither form.
struct SlotP {
  mlir::Value point_iv;
  mlir::ktdf::TilingReserveSizeOp tile;  // null when dim p is not tiled
};
SlotP slotPOf(mlir::ktdf::DataTransferOp load, int64_t p);

/// The ktdf.data_transfer ops directly inside `stage`, outside any nested
/// pipeline.
llvm::SmallVector<mlir::ktdf::DataTransferOp> stageTransfers(
    mlir::ktdf::StageOp stage) {
  llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
  stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(op)) return mlir::WalkResult::skip();
    if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op))
      transfers.push_back(transfer);
    return mlir::WalkResult::advance();
  });
  return transfers;
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

/// The slot `load` moves its elements along, the last one whose size is not
/// 1, when x's stride in it is not 1: the load needs re-sticking. nullopt
/// otherwise.
std::optional<int64_t> restickSlot(mlir::ktdf::DataTransferOp load) {
  std::optional<int64_t> q;
  for (auto [slot, size] : llvm::enumerate(load.getMixedSourceSizes()))
    if (mlir::getConstantIntValue(size) != 1) q = slot;
  auto x = layoutOf(load.getSource());
  if (!q || mlir::failed(x) || x->strides[*q] == 1) return std::nullopt;
  return q;
}

/// The index value at memref dim `dim` of one transfer side, when the side's
/// map gives that dim a plain loop index; null otherwise.
mlir::Value indexAt(mlir::AffineMap map, mlir::ValueRange indices,
                    int64_t dim) {
  auto expr = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(dim));
  return expr ? indices[expr.getPosition()] : mlir::Value();
}

/// Looks in `pipeline` for a copy that needs re-sticking: a transfer from a
/// global view x into a scratchpad buffer that moves its elements along a dim
/// of x whose stride is not 1.
/// Returns nullopt when there is none (nothing to do), failure after emitting
/// an error when there is one this pass cannot handle.
mlir::FailureOr<std::optional<Restick>> findRestick(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::ktdf_arch::ExecutionUnitOp compute) {
  auto fail = [](mlir::Operation* at, const llvm::Twine& msg) {
    mlir::InFlightDiagnostic diag = at->emitError(PASS_NAME ": ");
    diag << msg;
    return diag;
  };

  // The load: a transfer x -> A that moves its elements along a dim of x
  // whose stride is not 1.
  mlir::ktdf::DataTransferOp load;
  int64_t q = 0;
  for (auto stage : pipeline.getStages()) {
    for (auto transfer : stageTransfers(stage)) {
      if (!transfer.isSourceMemRef() ||
          isScratchpad(transfer.getSource(), memory_tree) ||
          !isScratchpad(transfer.getDestination(), memory_tree))
        continue;
      std::optional<int64_t> slot = restickSlot(transfer);
      if (!slot) continue;  // already moves contiguous sticks
      if (load)
        return fail(transfer, "a second load in the pipeline needs "
                              "re-sticking; only one per pipeline is "
                              "supported");
      load = transfer;
      q = *slot;
    }
  }
  if (!load) return std::optional<Restick>();

  // From here on this is a re-stick: every unmet expectation is an error.
  auto x = layoutOf(load.getSource());
  std::optional<int64_t> p = unitStrideDim(*x);
  if (!p) return fail(load, "the source view has no dim with stride 1");
  int64_t rank = x->sizes.size();

  // The copy A -> FIFO -> FIFO -> B -> y the load feeds. Widening edits each
  // of these in place, so none may have another reader.
  auto not_a_copy = [&]() {
    return fail(load, "the load does not feed a plain copy to a global view "
                      "(A -> FIFO, read_from_fifo, write_to_fifo, FIFO -> B, "
                      "B -> y, with nothing else reading A, the FIFOs or B); "
                      "re-sticking it needs its own transpose stage, which "
                      "is not supported");
  };
  auto from_a = soleOtherUser<mlir::ktdf::DataTransferOp>(
      load.getDestination(), load);
  if (!from_a || from_a.getSource() != load.getDestination() ||
      !from_a.isDestFifo())
    return not_a_copy();
  // The FIFO A's sticks go into is either read by B's transfer directly, or
  // passed on unchanged by read_from_fifo + write_to_fifo.
  mlir::Value fifo_in = from_a.getDestination(), fifo_out = fifo_in;
  auto into_b = soleOtherUser<mlir::ktdf::DataTransferOp>(fifo_in, from_a);
  if (!into_b) {
    auto read = soleOtherUser<mlir::ktdf::ReadFromFifoOp>(fifo_in, from_a);
    if (!read || !read->hasOneUse()) return not_a_copy();
    auto write =
        mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(*read->getUsers().begin());
    if (!write || write.getData() != read.getResult()) return not_a_copy();
    fifo_out = write.getFifoSlot();
    into_b = soleOtherUser<mlir::ktdf::DataTransferOp>(fifo_out, write);
  }
  if (!into_b || into_b.getSource() != fifo_out ||
      !into_b.isDestMemRef() ||
      !isScratchpad(into_b.getDestination(), memory_tree))
    return not_a_copy();
  auto store = soleOtherUser<mlir::ktdf::DataTransferOp>(
      into_b.getDestination(), into_b);
  if (!store || store.getSource() != into_b.getDestination() ||
      !store.isDestMemRef() || isScratchpad(store.getDestination(), memory_tree))
    return not_a_copy();
  auto hop = from_a->getParentOfType<mlir::ktdf::PipelineOp>();
  if (into_b->getParentOfType<mlir::ktdf::PipelineOp>() != hop ||
      hop == pipeline)
    return fail(load, "the copy's A -> B hop is not in a pipeline nested in "
                      "the re-stick's");

  auto y = layoutOf(store.getDestination());
  if (mlir::failed(y) || (int64_t)y->sizes.size() != rank)
    return fail(store, "the destination view is not a static view of rank ")
           << rank;
  if (y->strides[q] != 1)
    return fail(store, "the destination view's dim ")
           << q << ", which each transfer moves along, does not have stride 1";

  // Only fp16 sticks have been run on the device; other element sizes need
  // their own transfer modes checked first.
  mlir::Type element_type =
      mlir::cast<mlir::MemRefType>(load.getSource().getType()).getElementType();
  auto element_bytes = scheduler::tryGetSizeInBytes(element_type);
  if (!element_bytes || *element_bytes != 2)
    return fail(load, "only 2-byte elements are supported, got ")
           << (element_bytes ? (int64_t)*element_bytes : -1) << " bytes";

  // The load must already move one stick along dim q.
  int64_t block = stickSize(load, compute);
  if (block <= 1)
    return fail(load, "the device has no SIMD lanes for ") << element_type;
  if (mlir::getConstantIntValue(load.getMixedSourceSizes()[q]) != block)
    return fail(load, "the load's size in slot ")
           << q << " is not one stick (" << block << " elements)";

  int64_t extent = x->sizes[*p];
  if (extent % block != 0)
    return fail(load, "dim ") << *p << " has extent " << extent
                              << ", not a multiple of the stick size "
                              << block;

  // Each transfer moves one stick: size `block` in slot q, 1 elsewhere.
  for (auto [transfer, sizes] :
       {std::pair{load, load.getMixedSourceSizes()},
        std::pair{store, store.getMixedDestSizes()}}) {
    for (auto [slot, size] : llvm::enumerate(sizes)) {
      int64_t want = (int64_t)slot == q ? block : 1;
      if (mlir::getConstantIntValue(size) != want)
        return fail(transfer, "global-side size in slot ")
               << slot << " is not " << want
               << "; the pass expects one stick per transfer";
    }
  }


  // The loop over dim p, from what indexes slot p of the load.
  SlotP slot_p = slotPOf(load, *p);
  if (!slot_p.point_iv)
    return fail(load, "slot ")
           << *p << " of the load is not indexed by a loop over dim " << *p
           << " (expected a loop IV or `linearize_index [tile_iv : tile], "
              "[point_iv : 1]` with a ktdf.tiling.reserve_size tile)";
  // Not tiled: the loop walks the whole dim, so the dim must be one stick.
  if (!slot_p.tile && extent != block)
    return fail(load, "dim ")
           << *p << " is walked by an untiled loop over " << extent
           << " elements; only one stick (" << block << ") is supported";
  return std::optional<Restick>(
      Restick{load, from_a, into_b, store, *p, q, rank});
}

SlotP slotPOf(mlir::ktdf::DataTransferOp load, int64_t p) {
  mlir::Value index =
      indexAt(load.getSourceMap().value(), load.getSourceIndices(), p);
  if (!index) return {};
  if (mlir::scf::getForInductionVarOwner(index)) return {index, {}};
  auto lin = index.getDefiningOp<mlir::ktdf::TilingLinearizeIndexOp>();
  if (!lin || lin.getIvs().size() != 2 ||
      !mlir::scf::getForInductionVarOwner(lin.getIvs()[1]))
    return {};
  auto tile =
      lin.getStrides()[0].getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
  if (!tile) return {};
  return {lin.getIvs()[1], tile};
}

//===----------------------------------------------------------------------===//
// Widening a re-stick
//===----------------------------------------------------------------------===//

/// The backing memref.alloc of a staging buffer, through the ktdf.private
/// result it is yielded as.
struct StagingBuffer {
  mlir::Value value;  // the ktdf.private result the transfers use
  mlir::ktdf::PrivateOp private_op;
  unsigned result = 0;
  mlir::memref::AllocOp alloc;
};

mlir::FailureOr<StagingBuffer> stagingBufferOf(mlir::Value value) {
  StagingBuffer buffer;
  buffer.value = value;
  buffer.private_op = value.getDefiningOp<mlir::ktdf::PrivateOp>();
  if (!buffer.private_op) return mlir::failure();
  buffer.result = mlir::cast<mlir::OpResult>(value).getResultNumber();
  buffer.alloc = buffer.private_op.getYieldOp()
                     .getOperand(buffer.result)
                     .getDefiningOp<mlir::memref::AllocOp>();
  if (!buffer.alloc) return mlir::failure();
  return buffer;
}

/// Gives each dim in `new_sizes` of `buffer` that static size, dropping the
/// dynamic size operand of any dim that had one.
void resizeBuffer(StagingBuffer& buffer,
                  llvm::ArrayRef<std::pair<int64_t, int64_t>> new_sizes,
                  mlir::OpBuilder& builder) {
  mlir::MemRefType type = buffer.alloc.getType();
  llvm::SmallVector<int64_t> shape(type.getShape());
  for (auto [dim, size] : new_sizes) shape[dim] = size;
  llvm::SmallVector<mlir::Value> dynamic_sizes;
  unsigned next_dynamic = 0;
  for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
    if (!mlir::ShapedType::isDynamic(extent)) continue;
    mlir::Value operand = buffer.alloc.getDynamicSizes()[next_dynamic++];
    if (mlir::ShapedType::isDynamic(shape[dim]))
      dynamic_sizes.push_back(operand);
  }
  auto new_type = mlir::MemRefType::get(shape, type.getElementType(),
                                        mlir::MemRefLayoutAttrInterface{},
                                        type.getMemorySpace());
  builder.setInsertionPoint(buffer.alloc);
  auto new_alloc = mlir::memref::AllocOp::create(
      builder, buffer.alloc.getLoc(), new_type, dynamic_sizes);
  buffer.alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
  buffer.alloc.erase();
  buffer.alloc = new_alloc;
  buffer.private_op.getResult(buffer.result).setType(new_type);
}

/// Widens the re-stick `plan` to `block` x `block` tiles. See the file
/// comment for the shape of the result.
mlir::LogicalResult widenRestick(Restick plan,
                                 mlir::ktdf_arch::ExecutionUnitOp compute,
                                 mlir::OpBuilder& builder) {
  mlir::MLIRContext* ctx = plan.load.getContext();
  const int64_t p = plan.p, q = plan.q, block = stickSize(plan.load, compute);
  mlir::Location loc = plan.load.getLoc();

  // The constants every step uses, once each, at the top of the function so
  // they dominate every loop of the kernel.
  mlir::Block* entry =
      &plan.load->getParentOfType<mlir::func::FuncOp>().getBody().front();
  mlir::Value c0 = indexConstant(entry, 0, builder);
  mlir::Value c1 = indexConstant(entry, 1, builder);
  mlir::Value c_block = indexConstant(entry, block, builder);

  auto a = stagingBufferOf(plan.load.getDestination());
  auto b = stagingBufferOf(plan.store.getSource());
  if (mlir::failed(a) || mlir::failed(b))
    return plan.load->emitError(PASS_NAME
                                ": a staging buffer is not a ktdf.private "
                                "alloc");
  // A buffer index lists the stage's point loops first, then one dim per view
  // dim: view dim d is buffer dim lead + d.
  const int64_t lead = a->alloc.getType().getRank() - plan.rank;
  if (lead < 0 || b->alloc.getType().getRank() - plan.rank != lead)
    return plan.load->emitError(PASS_NAME
                                ": staging buffers must have the view rank "
                                "plus the same number of loop dims");

  mlir::ktdf::DataTransferOp from_a = plan.from_a, into_b = plan.into_b;
  auto hop = from_a->getParentOfType<mlir::ktdf::PipelineOp>();

  SlotP slot_p = slotPOf(plan.load, p);

  // The buffer dim that holds the point loop over dim p, if the point loop
  // is inside the pipeline. Every stage indexes it with its own copy of that
  // loop.
  std::optional<int64_t> kp;
  for (int64_t k = 0; k < lead; ++k)
    if (indexAt(plan.load.getDestMap().value(), plan.load.getDestIndices(),
                k) == slot_p.point_iv)
      kp = k;

  // 1. The point loops over dim p: the one indexing slot p of the load and, in
  //    each stage, the one indexing buffer dim kp. The widened transfers cover
  //    the whole stick, so step 6 removes them.
  llvm::SetVector<mlir::scf::ForOp> point_loops;
  point_loops.insert(mlir::scf::getForInductionVarOwner(slot_p.point_iv));
  if (kp) {
    llvm::SmallVector<mlir::ktdf::DataTransferOp> on_buffers;
    plan.load->getParentOfType<mlir::ktdf::PipelineOp>()->walk(
        [&](mlir::ktdf::DataTransferOp op) { on_buffers.push_back(op); });
    for (auto op : on_buffers) {
      for (bool source : {true, false}) {
        mlir::Value side = source ? op.getSource() : op.getDestination();
        if (side != a->value && side != b->value) continue;
        auto map = source ? op.getSourceMap() : op.getDestMap();
        mlir::Value iv =
            indexAt(*map, source ? op.getSourceIndices() : op.getDestIndices(),
                    *kp);
        auto loop = iv ? mlir::scf::getForInductionVarOwner(iv)
                       : mlir::scf::ForOp();
        if (!loop)
          return op->emitError(PASS_NAME ": staging-buffer dim ")
                 << *kp << " is not indexed by a loop";
        point_loops.insert(loop);
      }
    }
  }

  // The point loops run their body once per element along dim p; removing
  // them is right only if they move no data but the copy's. Any other data
  // reaches a stage through a transfer or a FIFO, so those are what is
  // checked. Checked before anything is changed.
  for (mlir::scf::ForOp loop : point_loops) {
    mlir::Operation* other = nullptr;
    loop->walk([&](mlir::Operation* op) {
      bool ours = true;
      if (auto t = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op))
        ours = t == plan.load || t == from_a || t == into_b || t == plan.store;
      else if (auto r = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op))
        ours = r.getFifoSlot() == from_a.getDestination();
      else if (auto w = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op))
        ours = w.getFifoSlot() == into_b.getSource();
      if (!ours) other = op;
    });
    if (other)
      return other->emitError(PASS_NAME
                              ": this op shares a point loop over the stick "
                              "dim with a re-stick, which removes the loop");
  }

  // 2. A re-stick tile is one stick along dim p, so a tiled loop over dim p
  //    has no tile size to choose: its placeholder becomes `block`. Every op
  //    stage coarsening derived from it (the tile loop's trip count, the
  //    linearize_index strides, the buffers' sizes) follows. The placeholder
  //    exists only because loop tiling runs before this pass and tiles every
  //    loop of a nest alike.
  if (slot_p.tile) {
    slot_p.tile.replaceAllUsesWith(c_block);
    slot_p.tile.erase();
  }

  // 3. The buffers hold one tile per point of the remaining loops: view dim p
  //    grows to `block`, the point dim over p shrinks to 1.
  llvm::SmallVector<std::pair<int64_t, int64_t>> new_sizes = {
      {lead + p, block}};
  if (kp) new_sizes.push_back({*kp, 1});
  resizeBuffer(*a, new_sizes, builder);
  resizeBuffer(*b, new_sizes, builder);

  // 4. The load and store move a whole tile: size `block` in slot p as well
  //    as in slot q, on both sides. Only static sizes change, so the dynamic
  //    size operands stay as they are.
  for (auto [op, global_is_source] :
       {std::pair{plan.load, true}, std::pair{plan.store, false}}) {
    auto src = op.getStaticSourceSizes(), dst = op.getStaticDestSizes();
    if (!src || !dst)
      return op->emitError(PASS_NAME ": the transfer has no static sizes");
    llvm::SmallVector<int64_t> global(global_is_source ? *src : *dst);
    llvm::SmallVector<int64_t> buffer(global_is_source ? *dst : *src);
    if (mlir::ShapedType::isDynamic(buffer[lead + p]))
      return op->emitError(PASS_NAME ": the staging-buffer size of dim ")
             << p << " is dynamic";
    global[p] = block;
    buffer[lead + p] = block;
    op.setStaticSourceSizes(global_is_source ? global : buffer);
    op.setStaticDestSizes(global_is_source ? buffer : global);
  }

  // 5. The copy's A -> B hop transposes each tile one element at a time:
  //    element (r, c) of A goes to (c, r) of B. Wrap the pipeline holding the
  //    hop in loops r, c over [0, block). Its read from A becomes a
  //    one-element splat at (lead + p, lead + q) = (r, c); its write to B a
  //    one-element extract at (c, r).
  builder.setInsertionPoint(hop);
  auto r_loop = mlir::scf::ForOp::create(builder, loc, c0, c_block, c1);
  builder.setInsertionPoint(r_loop.getBody()->getTerminator());
  auto c_loop = mlir::scf::ForOp::create(builder, loc, c0, c_block, c1);
  hop->moveBefore(c_loop.getBody()->getTerminator());
  mlir::Value r = r_loop.getInductionVar(), c = c_loop.getInductionVar();

  for (mlir::ktdf::DataTransferOp op : {from_a, into_b}) {
    bool reads_buffer = op == from_a;
    mlir::AffineMap map = reads_buffer ? *op.getSourceMap() : *op.getDestMap();
    mlir::ValueRange indices =
        reads_buffer ? op.getSourceIndices() : op.getDestIndices();
    // One index per buffer dim through an identity map: the point-loop
    // indices stay, every view dim is 0 except the two the element is at.
    llvm::SmallVector<mlir::Value> element(lead + plan.rank, c0);
    for (int64_t k = 0; k < lead; ++k) {
      element[k] = indexAt(map, indices, k);
      if (!element[k])
        return op->emitError(PASS_NAME ": staging-buffer dim ")
               << k << " is not indexed by a loop";
    }
    element[lead + p] = reads_buffer ? r : c;
    element[lead + q] = reads_buffer ? c : r;
    auto identity = mlir::AffineMap::getMultiDimIdentityMap(element.size(), ctx);
    llvm::SmallVector<int64_t> ones(element.size(), 1);
    if (reads_buffer) {
      op.getSourceIndicesMutable().assign(element);
      op.getSourceSizesMutable().clear();
      op.setSourceMap(identity);
      op.setStaticSourceSizes(ones);
    } else {
      op.getDestIndicesMutable().assign(element);
      op.getDestSizesMutable().clear();
      op.setDestMap(identity);
      op.setStaticDestSizes(ones);
    }
    op->setDiscardableAttr(
        "transfer_mode",
        mlir::StringAttr::get(ctx, reads_buffer ? "splat" : "extract"));
  }

  // 6. Remove the point loops over dim p: their body runs once, at point 0.
  mlir::IRRewriter rewriter(builder);
  for (mlir::scf::ForOp loop : point_loops) {
    loop.setUpperBound(c1);
    if (mlir::failed(loop.promoteIfSingleIteration(rewriter)))
      return loop->emitError(PASS_NAME
                             ": the point loop over the stick dim does not "
                             "start at 0 with step 1");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

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

    // Outermost pipelines only: a re-stick's nested pipeline is part of it.
    llvm::SmallVector<mlir::ktdf::PipelineOp> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(pipeline);
          return mlir::WalkResult::skip();
        });

    mlir::OpBuilder builder(&getContext());
    for (auto pipeline : pipelines) {
      auto plan = findRestick(pipeline, memory_tree, compute);
      if (mlir::failed(plan)) return signalPassFailure();
      if (!*plan) continue;
      LDBG(1) << "re-stick: p = " << (*plan)->p << ", q = " << (*plan)->q
              << " at " << pipeline.getLoc();
      if (mlir::failed(widenRestick(**plan, compute, builder)))
        return signalPassFailure();
    }

  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
