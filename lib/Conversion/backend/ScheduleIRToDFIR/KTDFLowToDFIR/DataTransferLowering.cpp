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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Create a vectorchain.shuffle that broadcasts src_vec (vector<src_elements x
/// T>) to vector<dst_elements x T> using indices [0..src_elements-1] repeated
/// (dst_elements / src_elements) times.
mlir::Value insertSplatShuffle(mlir::PatternRewriter& rewriter,
                               mlir::Location loc, mlir::Value src_vec,
                               int64_t src_elements, int64_t dst_elements) {
  auto src_vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  auto elem_type = src_vec_type.getElementType();

  // Build indices [0, 1, ..., src_elements-1]
  llvm::SmallVector<mlir::Attribute> index_attrs;
  for (int64_t i = 0; i < src_elements; ++i) {
    index_attrs.push_back(rewriter.getIntegerAttr(rewriter.getI32Type(), i));
  }
  auto indices_attr = rewriter.getArrayAttr(index_attrs);

  int32_t repetition = static_cast<int32_t>(dst_elements / src_elements);
  auto result_type = mlir::VectorType::get({dst_elements}, elem_type);

  return mlir::vectorchain::ShuffleOp::create(
             rewriter, loc, result_type, src_vec,
             /*mask=*/nullptr,
             /*dbgName=*/nullptr, indices_attr,
             rewriter.getI32IntegerAttr(repetition))
      .getOutput();
}

/// Collect the positions and extents of the non-unit entries of `sizes`,
/// excluding the innermost (vector) dimension. Outermost first.
void collectNonUnitOuterDims(llvm::ArrayRef<int64_t> sizes,
                             llvm::SmallVectorImpl<unsigned>& positions,
                             llvm::SmallVectorImpl<int64_t>& extents) {
  for (unsigned i = 0, e = sizes.size() ? sizes.size() - 1 : 0; i < e; ++i) {
    if (sizes[i] != 1) {
      positions.push_back(i);
      extents.push_back(sizes[i]);
    }
  }
}

/// Number of AGEN time dimensions used to describe a transfer that spans more
/// than one hardware vector. Transfers walking a single dimension pin the
/// second one to a single step.
constexpr unsigned kNumWideTransferTimeDims = 2;

/// The time-dimension description of an AGEN composite transfer.
struct TimeDimensions {
  mlir::IntegerSet set;
  mlir::AffineMap order;
  mlir::AffineMap load_addr_map;
  mlir::AffineMap store_addr_map;
};

/// Build the AGEN time dimensions of a composite transfer.
///
/// `extents` holds one entry per dimension the transfer walks over time,
/// outermost first, and `src_positions`/`dst_positions` give the index position
/// that dimension occupies on the load and on the store side. The k-th walked
/// dimension becomes time dimension `dk`, contributes `0 <= dk <= extent-1` to
/// the time set and appears as the sole non-zero result of the corresponding
/// address map. Those maps are *added* to the base index, so the base access
/// maps and their operands need no adjustment.
///
/// `extents` is empty when the transfer fits in a single hardware vector; a
/// single time step pinned to zero is emitted instead.
TimeDimensions buildTimeDimensions(mlir::MLIRContext* context,
                                   llvm::ArrayRef<int64_t> extents,
                                   llvm::ArrayRef<unsigned> src_positions,
                                   llvm::ArrayRef<unsigned> dst_positions,
                                   unsigned src_rank, unsigned dst_rank) {
  const unsigned num_walked = extents.size();
  assert(src_positions.size() == num_walked &&
         dst_positions.size() == num_walked &&
         "expected one index position per walked dimension on each side");
  const unsigned num_time_dims =
      num_walked ? kNumWideTransferTimeDims : unsigned(1);
  assert(num_walked <= num_time_dims && "too many walked dimensions");

  // time_set: the walked dimensions span their extent, the remaining ones are
  // pinned to a single step.
  llvm::SmallVector<mlir::AffineExpr> constraints;
  llvm::SmallVector<bool> eq_flags;
  for (unsigned dim = 0; dim < num_time_dims; ++dim) {
    auto expr = mlir::getAffineDimExpr(dim, context);
    if (dim < num_walked) {
      constraints.push_back(expr);
      eq_flags.push_back(false);
      constraints.push_back(
          mlir::getAffineConstantExpr(extents[dim] - 1, context) - expr);
      eq_flags.push_back(false);
    } else {
      constraints.push_back(expr);
      eq_flags.push_back(true);
    }
  }

  // time_order: pinned dimensions first, then the walked ones innermost.
  llvm::SmallVector<mlir::AffineExpr> order_exprs;
  for (unsigned dim = num_walked; dim < num_time_dims; ++dim) {
    order_exprs.push_back(mlir::getAffineDimExpr(dim, context));
  }
  for (unsigned dim = 0; dim < num_walked; ++dim) {
    order_exprs.push_back(mlir::getAffineDimExpr(dim, context));
  }

  // The address maps have one result per memref dimension; the walked
  // dimensions sit at the positions their side reported, everything else is a
  // zero offset from the base address.
  auto makeAddrMap = [&](unsigned rank, llvm::ArrayRef<unsigned> positions) {
    llvm::SmallVector<mlir::AffineExpr> results(
        rank, mlir::getAffineConstantExpr(0, context));
    for (auto [dim, position] : llvm::enumerate(positions)) {
      assert(position < rank && "walked dimension outside the memref rank");
      results[position] = mlir::getAffineDimExpr(dim, context);
    }
    auto map = mlir::AffineMap::get(num_time_dims, 0, results, context);
    // A misplaced time dimension silently transfers the wrong addresses, so
    // check the shape of the map that was just built.
    assert(map.getNumResults() == rank &&
           "time address map result count must match the memref rank");
    for (auto [dim, position] : llvm::enumerate(positions)) {
      assert(map.getResult(position) == mlir::getAffineDimExpr(dim, context) &&
             "time dimension is not at the reported result position");
    }
    return map;
  };

  return TimeDimensions{
      mlir::IntegerSet::get(num_time_dims, 0, constraints, eq_flags),
      mlir::AffineMap::get(num_time_dims, 0, order_exprs, context),
      makeAddrMap(src_rank, src_positions),
      makeAddrMap(dst_rank, dst_positions)};
}

/// Pattern to lower ktdf.data_transfer operations
struct LowerDataTransferPattern
    : public mlir::OpRewritePattern<mlir::ktdf::DataTransferOp> {
  LowerDataTransferPattern(mlir::MLIRContext* context,
                           const ResourceToUnits& components,
                           arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context),
        components_(components),
        resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::DataTransferOp data_transfer_op,
      mlir::PatternRewriter& rewriter) const override {
    auto src = data_transfer_op.getSource();
    auto dst = data_transfer_op.getDestination();

    bool src_is_fifo = data_transfer_op.isSourceFifo();
    bool dst_is_fifo = data_transfer_op.isDestFifo();

    // Extract FIFO types if applicable
    // Determine transfer type based on source and destination
    auto transfer_type_or = getDataTransferType(src_is_fifo, dst_is_fifo);
    if (mlir::failed(transfer_type_or)) {
      data_transfer_op.emitError(
          "Unsupported data transfer: FIFO to FIFO transfers are not allowed");
      return mlir::failure();
    }
    auto transfer_type = *transfer_type_or;

    // Get source and destination indices
    auto src_indices = data_transfer_op.getSourceIndices();
    auto dst_indices = data_transfer_op.getDestIndices();

    // Get static sizes
    assert(data_transfer_op.hasAllStaticSourceSizes() &&
           "Expected static source sizes");
    assert(data_transfer_op.hasAllStaticDestSizes() &&
           "Expected static dest sizes");

    auto src_static_sizes = *data_transfer_op.getStaticSourceSizes();
    auto dst_static_sizes = *data_transfer_op.getStaticDestSizes();

    // Calculate total elements from static sizes
    int64_t src_total_elements = 1;
    for (int64_t size : src_static_sizes) {
      src_total_elements *= size;
    }

    int64_t dst_total_elements = 1;
    for (int64_t size : dst_static_sizes) {
      dst_total_elements *= size;
    }

    // For splat/pad transfers the source is smaller than the destination —
    // the hardware replicates or zero-pads to fill the vector. Skip the
    // equality check and use the destination size as the transfer width.
    auto transfer_mode_attr =
        data_transfer_op->getDiscardableAttr("transfer_mode");
    bool is_broadcast_transfer =
        transfer_mode_attr &&
        (llvm::cast<mlir::StringAttr>(transfer_mode_attr).getValue() ==
             "splat" ||
         llvm::cast<mlir::StringAttr>(transfer_mode_attr).getValue() == "pad");

    if (is_broadcast_transfer) {
      if (src_total_elements > dst_total_elements) {
        data_transfer_op.emitError(
            "source total elements must not exceed destination for "
            "splat/pad transfer");
        return mlir::failure();
      }
    } else if (src_total_elements != dst_total_elements) {
      data_transfer_op.emitError(
          "source and destination total elements must match");
      return mlir::failure();
    }

    int64_t total_elements = dst_total_elements;

    // Get element type (from memref or FIFO slot)
    mlir::Type elem_type;
    if (src_is_fifo) {
      elem_type =
          llvm::cast<mlir::ktdf::FifoSlotType>(src.getType()).getElementType();
    } else {
      elem_type = llvm::cast<mlir::MemRefType>(src.getType()).getElementType();
    }

    // Hardware vector width (lanes for this element type). An AGEN transfer
    // moves at most one hardware vector per time step, so any request wider
    // than this must be split across several transfers.
    const auto compute_kind = resource_kinds_.getComputeKind();
    if (!compute_kind) {
      data_transfer_op.emitError(
          "cannot determine the hardware vector width: the architecture "
          "declares no compute resource kind");
      return mlir::failure();
    }
    const int64_t vector_width = std::max(
        resource_kinds_.getFeature<mlir::ktdf_arch::feature::SIMD>(compute_kind)
            .getLanes(elem_type),
        int64_t(1));

    auto vector_type = mlir::VectorType::get({total_elements}, elem_type);

    // Handle different transfer types
    switch (transfer_type) {
      case DataTransferType::kLoadAndStore: {
        // Both are memrefs
        auto src_memref = src;
        auto dst_memref = dst;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned src_num_dims = src_memref_type.getRank();
        unsigned dst_num_dims = dst_memref_type.getRank();

        auto src_map = data_transfer_op.getSourceMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(src_num_dims,
                                                    rewriter.getContext()));
        auto dst_map = data_transfer_op.getDestMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(dst_num_dims,
                                                    rewriter.getContext()));

        return lowerAsLoadAndStore(
            rewriter, data_transfer_op, src_memref, dst_memref, src_indices,
            dst_indices, src_static_sizes, dst_static_sizes, src_num_dims,
            dst_num_dims, vector_type, src_map, dst_map, vector_width);
      }

      case DataTransferType::kLoadAndSend: {
        // Source is memref, destination is FIFO
        auto src_memref = src;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        unsigned num_dims = src_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto src_map = data_transfer_op.getSourceMap().value_or(identity_map);

        auto dst_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(dst.getType());
        return lowerAsLoadAndSend(rewriter, data_transfer_op, src_memref,
                                  src_indices, src_static_sizes, num_dims,
                                  vector_type, src_map, dst_fifo_slot_type,
                                  is_broadcast_transfer, src_total_elements);
      }

      case DataTransferType::kReceiveAndStore: {
        // Source is FIFO, destination is memref
        auto dst_memref = dst;
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned num_dims = dst_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto dst_map = data_transfer_op.getDestMap().value_or(identity_map);

        auto src_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(src.getType());
        return lowerAsReceiveAndStore(rewriter, data_transfer_op, dst_memref,
                                      dst_indices, dst_static_sizes, num_dims,
                                      vector_type, dst_map, src_fifo_slot_type);
      }
    }

    return mlir::failure();
  }

 private:
  const ResourceToUnits& components_;
  arch_view::ResourceKinds& resource_kinds_;

  /// Lower as CompositeLoadAndStore
  mlir::LogicalResult lowerAsLoadAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::Value dst_memref, mlir::ValueRange src_indices,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      llvm::ArrayRef<int64_t> dst_static_sizes, unsigned src_num_dims,
      unsigned dst_num_dims, mlir::VectorType vector_type,
      mlir::AffineMap src_map, mlir::AffineMap dst_map,
      int64_t vector_width) const {
    // Sizes describing the elements covered by one AGEN vector transfer. They
    // are narrowed below when the request is wider than one hardware vector.
    llvm::SmallVector<int64_t> load_sizes(src_static_sizes);
    llvm::SmallVector<int64_t> store_sizes(dst_static_sizes);
    mlir::VectorType load_iv_type = vector_type;

    // Dimensions the transfer walks over time, outermost first: their extents
    // and the index position they occupy on each side. Empty when the request
    // fits in a single hardware vector.
    llvm::SmallVector<int64_t> time_extents;
    llvm::SmallVector<unsigned> src_time_dims;
    llvm::SmallVector<unsigned> dst_time_dims;

    // An AGEN composite transfer moves one hardware vector per time step. A
    // request wider than that becomes a single transfer whose time dimensions
    // walk the non-unit outer dimensions.
    if (vector_type.getNumElements() > vector_width) {
      if (mlir::failed(collectTransferTimeDims(
              data_transfer_op, src_static_sizes, dst_static_sizes,
              vector_width, src_time_dims, dst_time_dims, time_extents))) {
        return mlir::failure();
      }
      load_iv_type =
          mlir::VectorType::get({vector_width}, vector_type.getElementType());
      load_sizes.assign(src_static_sizes.size(), 1);
      load_sizes.back() = vector_width;
      store_sizes.assign(dst_static_sizes.size(), 1);
      store_sizes.back() = vector_width;
    }

    // Build src_load_set from source sizes
    auto src_load_set =
        scheduler::buildIntegerSetFromSizes(rewriter.getContext(), load_sizes);

    // Build dst_store_set from destination sizes
    auto dst_store_set =
        scheduler::buildIntegerSetFromSizes(rewriter.getContext(), store_sizes);

    // load_order and store_order must match their respective set
    // dimensionality.
    auto load_order = mlir::AffineMap::getMultiDimIdentityMap(
        src_num_dims, rewriter.getContext());
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        dst_num_dims, rewriter.getContext());

    // Time dimensions: a single pinned step for a transfer of at most one
    // vector, one dimension per walked outer dimension otherwise.
    auto time_dims =
        buildTimeDimensions(rewriter.getContext(), time_extents, src_time_dims,
                            dst_time_dims, src_num_dims, dst_num_dims);

    // Create the CompositeLoadAndStoreOp
    mlir::agen::CompositeLoadAndStoreOp::create(
        rewriter, data_transfer_op.getLoc(), src_memref, dst_memref,
        /*dbgName=*/nullptr, src_map, src_indices, dst_map, dst_indices,
        src_load_set, load_order, dst_store_set, store_order, {},
        time_dims.set, time_dims.order, time_dims.load_addr_map,
        time_dims.store_addr_map, load_iv_type);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Validate a memref-to-memref transfer that is wider than one hardware
  /// vector and report the dimensions it has to walk over time: their extents
  /// and the index position they occupy on the source and on the destination
  /// side, outermost first.
  mlir::LogicalResult collectTransferTimeDims(
      mlir::ktdf::DataTransferOp data_transfer_op,
      llvm::ArrayRef<int64_t> src_static_sizes,
      llvm::ArrayRef<int64_t> dst_static_sizes, int64_t vector_width,
      llvm::SmallVectorImpl<unsigned>& src_time_dims,
      llvm::SmallVectorImpl<unsigned>& dst_time_dims,
      llvm::SmallVectorImpl<int64_t>& time_extents) const {
    int64_t total_elements = 1;
    for (int64_t size : dst_static_sizes) total_elements *= size;

    if (src_static_sizes.empty() || dst_static_sizes.empty()) {
      data_transfer_op.emitError()
          << "data transfer of " << total_elements
          << " elements exceeds the hardware vector width of " << vector_width
          << " but has no dimensions to split";
      return mlir::failure();
    }

    if (src_static_sizes.back() != vector_width ||
        dst_static_sizes.back() != vector_width) {
      data_transfer_op.emitError()
          << "data transfer of " << total_elements
          << " elements exceeds the hardware vector width of " << vector_width
          << "; splitting requires the innermost source and destination sizes "
             "to equal the vector width, but they are "
          << src_static_sizes.back() << " and " << dst_static_sizes.back();
      return mlir::failure();
    }

    if (total_elements % vector_width != 0) {
      data_transfer_op.emitError()
          << "data transfer of " << total_elements
          << " elements is not a multiple of the hardware vector width of "
          << vector_width << "; partial vector transfers are not supported";
      return mlir::failure();
    }

    llvm::SmallVector<int64_t> dst_extents;
    collectNonUnitOuterDims(src_static_sizes, src_time_dims, time_extents);
    collectNonUnitOuterDims(dst_static_sizes, dst_time_dims, dst_extents);

    if (time_extents != dst_extents) {
      data_transfer_op.emitError()
          << "source and destination non-unit outer sizes must match to split "
             "a transfer of "
          << total_elements << " elements across multiple vectors";
      return mlir::failure();
    }

    if (time_extents.size() > kNumWideTransferTimeDims) {
      return data_transfer_op.emitError()
             << "transfer requires " << time_extents.size()
             << " AGEN time dimensions; only 1 and 2 are device-verified";
    }

    return mlir::success();
  }

  /// Lower as vector_load and send (L1 to FIFO)
  mlir::LogicalResult lowerAsLoadAndSend(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::ValueRange src_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap src_map,
      mlir::ktdf::FifoSlotType dst_fifo_slot_type, bool is_splat,
      int64_t src_total_elements) const {
    // Build load_set from source sizes
    auto load_set = scheduler::buildIntegerSetFromSizes(rewriter.getContext(),
                                                        src_static_sizes);

    // Build load_order
    auto load_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // When splat: load only src_total_elements, then shuffle to full width.
    if (is_splat) {
      if (vector_type.getNumElements() % src_total_elements != 0) {
        data_transfer_op.emitError(
            "dst_total_elements must be divisible by src_total_elements "
            "for splat transfer");
        return mlir::failure();
      }
    }
    auto load_type = is_splat
                         ? mlir::VectorType::get({src_total_elements},
                                                 vector_type.getElementType())
                         : vector_type;

    // Create vector_load operation
    auto vector_load_op = mlir::agen::VectorLoadOp::create(
        rewriter, data_transfer_op.getLoc(), load_type, src_memref,
        /*dbgName=*/nullptr, src_map, src_indices, load_set, load_order);

    // For splat: broadcast loaded vector to full destination width.
    mlir::Value send_value = vector_load_op.getResult();
    if (is_splat) {
      send_value =
          insertSplatShuffle(rewriter, data_transfer_op.getLoc(), send_value,
                             src_total_elements, vector_type.getNumElements());
    }

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the destination unit from the FIFO dest attribute
    auto dest_unit_result = resolveUnitFromFifoAttr(
        dst_fifo_slot_type.getDest(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(dest_unit_result)) {
      return mlir::failure();
    }
    mlir::Value dest_unit = *dest_unit_result;

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, data_transfer_op.getLoc(),
                                   dest_unit, send_value,
                                   /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as receive and vector_store (FIFO to L1)
  mlir::LogicalResult lowerAsReceiveAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value dst_memref,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> dst_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap dst_map,
      mlir::ktdf::FifoSlotType src_fifo_slot_type) const {
    // Build store_set from destination sizes
    auto store_set = scheduler::buildIntegerSetFromSizes(rewriter.getContext(),
                                                         dst_static_sizes);

    // Build store_order
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the source unit from the FIFO src attribute
    auto src_unit_result = resolveUnitFromFifoAttr(
        src_fifo_slot_type.getSrc(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(src_unit_result)) {
      return mlir::failure();
    }
    mlir::Value src_unit = *src_unit_result;

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, data_transfer_op.getLoc(), vector_type, src_unit,
        /*dbgName=*/nullptr);

    // Create vector_store operation
    mlir::agen::VectorStoreOp::create(
        rewriter, data_transfer_op.getLoc(), receive_op.getData(), dst_memref,
        /*dbgName=*/nullptr, dst_map, dst_indices, store_set, store_order);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }
};

}  // namespace

void scheduler::populateDataTransferLoweringPatterns(
    mlir::RewritePatternSet& patterns, const ResourceToUnits& components,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerDataTransferPattern>(patterns.getContext(), components,
                                         resource_kinds);
}
