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
// Lowering compute operations (linalg.generic, arith, math) into DFIR.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/OpaqueTemplates.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Pattern to lower linalg.generic compute operations
struct LowerLinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LowerLinalgGenericPattern(mlir::MLIRContext* context,
                            arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const override {
    // Buffer-semantics path: init operand is a memref accumulator.
    if (generic_op.hasPureBufferSemantics())
      return lowerMemRefGenericOp(generic_op, rewriter);

    if (!generic_op.hasPureTensorSemantics() ||
        generic_op.getNumResults() != 1) {
      return mlir::failure();
    }

    // If the generic has any reduction dimensions, delegate to the dedicated
    // reduction lowering path before the elementwise path touches block args.
    for (auto iter_type : generic_op.getIteratorTypesArray()) {
      if (iter_type == mlir::utils::IteratorType::reduction)
        return lowerReductionGenericOp(generic_op, rewriter);
    }

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) {
      return mlir::failure();
    }

    // Replace block arguments with generic inputs.  Any input that is a
    // constant tensor (e.g. a dense<0.0>) is converted to an equivalent
    // arith.constant with vector type first so that vectorchain.binary always
    // receives vector-typed operands.
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      mlir::Value converted =
          convertConstTensorInputToVector(input, generic_op, rewriter);
      block_arg.replaceAllUsesWith(converted);
    }

    // Identity affine map used as op_specific_map for binary ops.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    // Collect compute operations to replace
    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator()) {
      ops_to_lower.push_back(&op);
    }

    // Process and lower compute operations via visitors
    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::mul);
              })
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::add);
              })
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::sub);
              })
              .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max);
              })
              .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min);
              })
              .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::abs_max);
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });

      if (mlir::failed(result)) return mlir::failure();
    }

    // Replace the generic op with the yield operand
    rewriter.replaceOp(generic_op, yield_op.getOperand(0));
    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;

  // Lowers a linalg.generic with buffer semantics (memref init operand).
  // The resulting accumulated vector is written back to the output buffer via
  // agen.vector_store.
  mlir::LogicalResult lowerMemRefGenericOp(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    mlir::Location loc = generic_op.getLoc();

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) return mlir::failure();

    // Replace input block arguments with their corresponding linalg ins
    // operands.
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs()))
      block_arg.replaceAllUsesWith(input);

    // The output block argument represents the current accumulator value held
    // in the output memref.  Emit an agen.vector_load to read it into a vector,
    // then replace all uses of the output block arg with that loaded vector.
    mlir::Value out_memref = generic_op.getDpsInitOperand(0)->get();
    auto out_memref_type = mlir::cast<mlir::MemRefType>(out_memref.getType());
    auto acc_vec_type =
        getFlattenedVectorType(out_memref_type, resource_kinds_);
    if (!acc_vec_type) return mlir::failure();
    rewriter.setInsertionPoint(generic_op);
    body.getArguments().back().replaceAllUsesWith(
        scheduler::emitVectorLoad(rewriter, loc, acc_vec_type, out_memref));

    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator())
      ops_to_lower.push_back(&op);

    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::mul);
              })
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::add);
              })
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::sub);
              })
              .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max);
              })
              .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min);
              })
              .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::abs_max);
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });
      if (mlir::failed(result)) return mlir::failure();
    }

    // Write the vectorchain.binary result back to the output buffer.
    scheduler::emitVectorStore(rewriter, loc, yield_op.getOperand(0),
                               out_memref);

    rewriter.eraseOp(generic_op);
    return mlir::success();
  }

  // Lowers a linalg.generic that has one or more reduction dimensions.
  //
  // One scf.for loop is emitted per reduction dimension (outermost first),
  // carrying the output vector as an iter_arg accumulator.  Each innermost
  // iteration extracts a parallel-shaped slice from the input tensor (size 1
  // along every reduction dim, full extent along parallel dims), then
  // accumulates it into the current accumulator via vectorchain.binary.
  //
  // The body op (addf / mulf / subf) determines the binary operator; the
  // existing lowerXxxFOp helpers are reused for the accumulation step.
  mlir::LogicalResult lowerReductionGenericOp(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    mlir::Location loc = generic_op.getLoc();

    // Require exactly one body op (plus the linalg.yield terminator).
    mlir::Block& body = generic_op.getRegion().front();
    llvm::SmallVector<mlir::Operation*> body_ops;
    for (mlir::Operation& op : body.without_terminator())
      body_ops.push_back(&op);
    if (body_ops.size() != 1)
      return generic_op.emitError(
          "reduction linalg.generic body must have exactly one compute op");

    // Map body op kind to the vectorchain binary operator.
    mlir::vectorchain::VectorChainBinaryOperator binary_kind;
    if (mlir::isa<mlir::arith::AddFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::add;
    else if (mlir::isa<mlir::arith::MulFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::mul;
    else if (mlir::isa<mlir::arith::SubFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::sub;
    else if (mlir::isa<mlir::arith::MaximumFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::max;
    else if (mlir::isa<mlir::arith::MinimumFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::min;
    else if (mlir::isa<mlir::arith::MaxNumFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::abs_max;
    else
      return body_ops[0]->emitError(
          "unsupported reduction body op in linalg.generic");

    // Collect reduction dim indices and their sizes from the input type.
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic_op.getDpsInputOperand(0)->get().getType());
    if (!input_type) return mlir::failure();

    const auto iterator_types = generic_op.getIteratorTypesArray();
    llvm::SmallVector<int64_t> red_dims;
    for (int64_t i = 0; i < static_cast<int64_t>(iterator_types.size()); ++i) {
      if (iterator_types[i] == mlir::utils::IteratorType::reduction)
        red_dims.push_back(i);
    }

    // The output type has only parallel dims and always fits in vector_length.
    auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic_op.getDpsInitOperand(0)->get().getType());
    if (!output_type) return mlir::failure();

    mlir::VectorType vec_type =
        getFlattenedVectorType(output_type, resource_kinds_);
    if (!vec_type) return mlir::failure();

    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    rewriter.setInsertionPoint(generic_op);

    // Build the initial accumulator from the init operand.  For a tensor.empty
    // (undefined init) use a zero vector; for a constant tensor reshape it.
    mlir::Value init = generic_op.getDpsInitOperand(0)->get();
    mlir::Value acc =
        convertConstTensorInputToVector(init, generic_op, rewriter);
    if (acc.getType() != vec_type) {
      // Non-constant init (e.g. tensor.empty) — zero is the correct identity
      // for reductions that start with an uninitialised accumulator.
      acc = mlir::arith::ConstantOp::create(rewriter, loc, vec_type,
                                            rewriter.getZeroAttr(vec_type));
    }

    // Emit one scf.for per reduction dimension (outermost first), each
    // carrying the accumulator as its single iter_arg.
    //
    // The input to the linalg.generic comes from a ktdf.read_from_fifo whose
    // result type includes all dims (parallel + reduction).  Rather than
    // tensor.extract_slice-ing that large tensor inside the loop (which would
    // require LowerReadFromFifoPattern to lower a tensor larger than
    // vector_length), we instead emit one new ktdf.read_from_fifo per loop
    // iteration — each producing the parallel-only shaped slice directly.
    // The original read_from_fifo is replaced by the new ones so it is erased.
    mlir::Value input = generic_op.getDpsInputOperand(0)->get();
    auto read_from_fifo = mlir::dyn_cast_or_null<mlir::ktdf::ReadFromFifoOp>(
        input.getDefiningOp());
    if (!read_from_fifo)
      return generic_op.emitError(
          "reduction linalg.generic input must be produced by "
          "ktdf.read_from_fifo");

    // Parallel-only result type for each per-step read.
    llvm::SmallVector<int64_t> parallel_shape;
    for (int64_t i = 0; i < static_cast<int64_t>(iterator_types.size()); ++i) {
      if (iterator_types[i] != mlir::utils::IteratorType::reduction)
        parallel_shape.push_back(input_type.getShape()[i]);
    }
    auto parallel_type = mlir::RankedTensorType::get(
        parallel_shape, input_type.getElementType());

    mlir::Value cur_acc = acc;

    // Build nested scf.for loops, one per reduction dimension.
    llvm::SmallVector<mlir::scf::ForOp> for_ops;
    for (int64_t red_dim : red_dims) {
      mlir::Value lb = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      mlir::Value ub = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, input_type.getShape()[red_dim]);
      mlir::Value step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto for_op = mlir::scf::ForOp::create(rewriter, loc, lb, ub, step,
                                             mlir::ValueRange{cur_acc});
      for_ops.push_back(for_op);
      rewriter.setInsertionPointToStart(for_op.getBody());
      cur_acc = for_op.getRegionIterArgs()[0];
    }

    // Innermost body: emit a new read_from_fifo producing parallel_type,
    // then accumulate via vectorchain.binary.
    auto new_read = mlir::ktdf::ReadFromFifoOp::create(
        rewriter, loc, parallel_type, read_from_fifo.getFifoSlot());

    // The parallel slice fits in vector_length.
    auto binary = mlir::vectorchain::BinaryOp::create(
        rewriter, loc, vec_type, cur_acc, new_read.getResult(),
        /*mask=*/nullptr, /*dbgName=*/nullptr, binary_kind, identity_map);

    // Yield the new accumulator up through each loop level.
    mlir::Value result = binary.getData();
    for (auto for_op : llvm::reverse(for_ops)) {
      mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{result});
      result = for_op.getResult(0);
      rewriter.setInsertionPointAfter(for_op);
    }

    // Replace the linalg.generic result and erase the original read_from_fifo
    // (which is now unused).
    rewriter.replaceOp(generic_op, result);
    rewriter.eraseOp(read_from_fifo);
    return mlir::success();
  }

  /// Converts a constant tensor input of a linalg.generic to a vector-typed
  /// arith.constant, preserving all element values.  This is needed because
  /// linalg.generic inputs can be constant tensors (e.g. a dense<0.0>), while
  /// vectorchain.binary requires vector operands.
  ///
  /// Only arith.constant ops whose value is a DenseElementsAttr are handled;
  /// any other input (non-constant tensors, vectors, scalars) is returned
  /// unchanged.
  mlir::Value convertConstTensorInputToVector(
      mlir::Value input, mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    // Only act on tensor-typed inputs — vectors and scalars pass through.
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensor_type) return input;

    // Must be a constant op with a dense attribute to convert.
    auto const_op =
        mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(input.getDefiningOp());
    if (!const_op) return input;
    auto dense_attr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (!dense_attr) return input;

    // Determine the target vector type (same element type, flattened shape).
    auto vector_type = getFlattenedVectorType(tensor_type, resource_kinds_);
    if (!vector_type) return input;

    // Re-materialise the constant with the vector type, preserving all element
    // values by reinterpreting the same dense data into the flat vector shape.
    auto vec_attr = dense_attr.reshape(vector_type);
    rewriter.setInsertionPoint(generic_op);
    return mlir::arith::ConstantOp::create(rewriter, const_op.getLoc(),
                                           vector_type, vec_attr)
        .getResult();
  }

  // Unified helper: lowers any two-operand arith float op to
  // vectorchain.binary with the given binary_kind.
  mlir::LogicalResult lowerBinaryFOp(
      mlir::Operation* op, mlir::Value lhs, mlir::Value rhs,
      mlir::PatternRewriter& rewriter, mlir::AffineMap identity_map,
      mlir::vectorchain::VectorChainBinaryOperator binary_kind) const {
    auto vector_type = getFlattenedVectorType(lhs.getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, op->getLoc(), vector_type, lhs, rhs,
        /*mask=*/nullptr, /*dbgName=*/nullptr, binary_kind, identity_map);

    rewriter.replaceOp(op, binary_op.getData());
    return mlir::success();
  }
};

/// Pattern to lower linalg.fill into:
///   vectorchain.constant_bitstream {value = [0x0]} : vector<1xT>
///   vectorchain.shuffle ... {indices = [0 : i32], repetition = N}
///       : vector<1xT>, vector<NxT>
///
/// Buffer semantics (memref output): the shuffle result is written to the
/// output memref via agen.vector_store and the fill is erased.
///
/// Tensor semantics (tensor output): the shuffle result directly replaces
/// the fill result (consumed by downstream vectorchain / FIFO ops).
///
/// Only zero fill values are supported. N and T are derived from the output
/// type shape and element type.
struct LowerLinalgFillPattern
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  LowerLinalgFillPattern(mlir::MLIRContext* context,
                         arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::FillOp fill_op,
      mlir::PatternRewriter& rewriter) const override {
    // linalg.fill must have exactly one input (the fill scalar).
    if (fill_op.getInputs().size() != 1) return mlir::failure();
    mlir::Value fill_val = fill_op.getInputs()[0];
    auto const_op = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
        fill_val.getDefiningOp());
    if (!const_op) return mlir::failure();
    auto scalar_attr = mlir::dyn_cast<mlir::TypedAttr>(const_op.getValue());
    if (!scalar_attr) return mlir::failure();

    // Derive output vector type from the output operand (memref or tensor).
    mlir::Value out_operand = fill_op.getOutputs()[0];
    mlir::VectorType out_vec_type =
        getFlattenedVectorType(out_operand.getType(), resource_kinds_);
    if (!out_vec_type) return mlir::failure();

    mlir::Location loc = fill_op.getLoc();
    int64_t total_elements = out_vec_type.getNumElements();
    mlir::Type elem_type = out_vec_type.getElementType();

    rewriter.setInsertionPoint(fill_op);

    // Only zero fills are supported.
    if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(scalar_attr)) {
      if (!fa.getValue().isZero())
        return fill_op.emitError(
            "linalg.fill lowering only supports zero fill values");
    } else if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(scalar_attr)) {
      if (!ia.getValue().isZero())
        return fill_op.emitError(
            "linalg.fill lowering only supports zero fill values");
    } else {
      return fill_op.emitError(
          "linalg.fill constant value must be integer or float");
    }

    // Step 1: vectorchain.constant_bitstream {value = [0x0]} : vector<1xT>
    mlir::VectorType seed_type = mlir::VectorType::get({1}, elem_type);
    mlir::ArrayAttr value_attr = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getI64Type(), 0)});
    auto bitstream = mlir::vectorchain::ConstantBitstreamOp::create(
        rewriter, loc, seed_type, value_attr);

    // Step 2: vectorchain.shuffle — splat to vector<NxT>
    mlir::ArrayAttr indices_attr = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getI32Type(), 0)});
    auto shuffle = mlir::vectorchain::ShuffleOp::create(
        rewriter, loc, out_vec_type, bitstream.getResult(),
        /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
        /*mask=*/nullptr, /*dbgName=*/nullptr, indices_attr,
        static_cast<uint32_t>(total_elements));

    // Step 3a (tensor): replace the fill result directly with the vector.
    if (!fill_op.getResultTensors().empty()) {
      rewriter.replaceOp(fill_op, shuffle.getOutput());
      return mlir::success();
    }

    // Step 3b (memref): write the filled vector into the output memref.
    scheduler::emitVectorStore(rewriter, loc, shuffle.getOutput(), out_operand);

    rewriter.eraseOp(fill_op);
    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;
};

/// Lowers an on-stick (lane-axis) reduction, carried as `ktdf.opaque
/// "LANE_REDUCE"`, into an in-register reduction: the lanes of one vector
/// register are collapsed, so unlike lowerReductionGenericOp's cross-register
/// form it needs no loop and no FIFO.
///
/// Runs memory-to-memory, not as an SSA chain: an arithmetic operand must be
/// something an address generator can name -- a load, or a lane permutation of
/// one -- which another arithmetic op's result is not. The staging buffer is the
/// extra input operand when the op carries one, else the input rewritten in
/// place; a vector input with no buffer cannot be lowered.
///
/// Phase 1 is a butterfly over each group of kGroupSize lanes, using two fixed
/// permutations:
///
///   p1 = [2, 3, 0, 1, 6, 7, 6, 7]      p2 = [4, 5, 3, 3, 5, 5, 6, 7]
///
///   v1 = shuffle(v,  p1) + v
///   v2 = v1 + shuffle(v1, p2)
///   v3 = shuffle(v2, p1) + shuffle(v2, p2)
///
/// leaving group g's sum in v3[8g] and junk in the group's other lanes. Each
/// stage loads what the previous stored, which is also what orders them.
///
/// Two operand rules are load-bearing. The backend fuses each shuffle into its
/// consumer as a per-operand NFWD modifier and SILENTLY drops any it cannot
/// place, so violating either yields a wrong sum whose total still looks right:
///   * p1 fuses only onto operand 0, p2 only onto operand 1 (the `nfwd0` and
///     `nfwd2` ports). Hence `shuffle(v, p1) + v`, not `v + shuffle(v, p1)` --
///     do not normalise these to a uniform order.
///   * Each operand needs its own load, which is why stage 3 loads twice for
///     what is arithmetically the same value.
///
/// Phase 2 broadcasts each group's lane 0 across its group, then one
/// scan_with_gap with gap = kGroupSize combines the groups, leaving the total at
/// lane 0 and the rest unspecified. The broadcast is its own stage because the
/// scan has no operand slot to fold a permutation into.
///
/// A narrower result is absorbed here rather than by reducing a linalg
/// dimension: both widths reduce at the input width, and a closing shuffle reads
/// lane 0 once for a single-element result, once per lane for a full-width one.
struct LowerLaneReducePattern
    : public mlir::OpRewritePattern<mlir::ktdf::OpaqueOp> {
  LowerLaneReducePattern(mlir::MLIRContext* context,
                         arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::OpaqueOp opaque_op,
      mlir::PatternRewriter& rewriter) const override {
    if (opaque_op.getTemplateName() != mlir::ktdf::kLaneReduceTemplateName)
      return mlir::failure();

    if (opaque_op.getInputs().empty() || opaque_op->getNumResults() != 1)
      return opaque_op.emitError(
          "lane reduction expects at least one input and exactly one result");

    // Only the first input carries data. A further memref input is the staging
    // buffer, threaded onto the op so that address assignment has already
    // placed it by the time this runs.
    mlir::Value input = opaque_op.getInputs().front();
    mlir::Value staging;
    for (mlir::Value extra : opaque_op.getInputs().drop_front()) {
      if (mlir::isa<mlir::MemRefType>(extra.getType())) {
        staging = extra;
        break;
      }
    }

    // Three input forms reach this arm:
    //
    //   vector -- already in a register; combine its lanes as is.
    //   memref -- a preceding reduction's running total; load it first.
    //   anything else -- defer. A tensor operand means the producer is not
    //             lowered yet; the greedy driver revisits once it is. Letting
    //             one through would hand a tensor to the shuffles.
    //
    // The reduction runs at the input width throughout -- that is the register
    // whose lanes are being combined.
    auto input_memref_type = mlir::dyn_cast<mlir::MemRefType>(input.getType());
    auto vec_type = mlir::dyn_cast<mlir::VectorType>(input.getType());
    if (!vec_type && input_memref_type)
      vec_type = getFlattenedVectorType(input_memref_type, resource_kinds_);
    if (!vec_type) return mlir::failure();

    mlir::VectorType result_vec_type = getFlattenedVectorType(
        opaque_op->getResult(0).getType(), resource_kinds_);
    if (!result_vec_type) return mlir::failure();
    if (result_vec_type.getElementType() != vec_type.getElementType())
      return opaque_op.emitError(
          "lane reduction input and result must have the same element type");

    // The butterfly reduces groups of kGroupSize lanes and the scan then
    // combines kGroupSize groups, so together they cover kGroupSize^2 lanes.
    const int64_t num_lanes = vec_type.getNumElements();
    if (num_lanes != kGroupSize * kGroupSize)
      return opaque_op.emitError("lane reduction requires a ")
             << kGroupSize * kGroupSize << "-lane vector, got " << num_lanes;

    // Only two result widths are meaningful: the single reduced value, or the
    // whole register it sits in. Anything between names lanes the reduction says
    // nothing about, so the shuffle's result would be partly undefined.
    const int64_t num_result_lanes = result_vec_type.getNumElements();
    if (num_result_lanes != 1 && num_result_lanes != num_lanes)
      return opaque_op.emitError(
                 "lane reduction result must hold either 1 element or the "
                 "full ")
             << num_lanes << " of the input, got " << num_result_lanes;

    // Pick the buffer the stages are staged through, and reject the one input
    // form that leaves none: a register with no buffer alongside it.
    if (!staging) {
      if (!input_memref_type)
        return opaque_op.emitError(
            "lane reduction of a value already in a register needs a buffer "
            "operand to stage its intermediates through");
      staging = input;
    }
    mlir::VectorType staging_vec_type = getFlattenedVectorType(
        mlir::cast<mlir::MemRefType>(staging.getType()), resource_kinds_);
    if (!staging_vec_type || staging_vec_type != vec_type)
      return opaque_op.emitError(
                 "lane reduction staging buffer must hold exactly the register "
                 "being reduced, ")
             << vec_type << ", got " << staging.getType();

    mlir::Location loc = opaque_op.getLoc();
    rewriter.setInsertionPoint(opaque_op);

    // Seed the staging buffer. A register input has to be written out first; a
    // buffer input that is not itself the staging buffer is copied across; and a
    // buffer input that *is* the staging buffer already holds the value.
    if (!input_memref_type) {
      scheduler::emitVectorStore(rewriter, loc, input, staging);
    } else if (input != staging) {
      scheduler::emitVectorStore(
          rewriter, loc,
          scheduler::emitVectorLoad(rewriter, loc, vec_type, input), staging);
    }

    static constexpr int64_t kP1[kGroupSize] = {2, 3, 0, 1, 6, 7, 6, 7};
    static constexpr int64_t kP2[kGroupSize] = {4, 5, 3, 3, 5, 5, 6, 7};
    static constexpr int64_t kBroadcastLane0[kGroupSize] = {0, 0, 0, 0,
                                                            0, 0, 0, 0};

    auto shuffle = [&](mlir::Value in,
                       llvm::ArrayRef<int64_t> indices) -> mlir::Value {
      llvm::SmallVector<mlir::Attribute> index_attrs;
      index_attrs.reserve(indices.size());
      for (int64_t index : indices)
        index_attrs.push_back(
            mlir::IntegerAttr::get(rewriter.getI32Type(), index));
      return mlir::vectorchain::ShuffleOp::create(
                 rewriter, loc, vec_type, in, /*variable=*/mlir::ValueRange{},
                 /*pad=*/mlir::ValueRange{}, /*mask=*/nullptr,
                 /*dbgName=*/nullptr, rewriter.getArrayAttr(index_attrs),
                 static_cast<uint32_t>(kGroupSize))
          .getOutput();
    };

    // All lane movement happens in the shuffles, so the arithmetic op must do
    // no cross-lane work of its own: identity op_specific_map.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());
    auto add = [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
      return mlir::vectorchain::BinaryOp::create(
                 rewriter, loc, vec_type, lhs, rhs, /*mask=*/nullptr,
                 /*dbgName=*/nullptr,
                 mlir::vectorchain::VectorChainBinaryOperator::add,
                 identity_map)
          .getData();
    };

    auto load = [&]() -> mlir::Value {
      return scheduler::emitVectorLoad(rewriter, loc, vec_type, staging);
    };
    auto store = [&](mlir::Value value) {
      scheduler::emitVectorStore(rewriter, loc, value, staging);
    };

    // Phase 1: butterfly, reducing each group of kGroupSize lanes into lane 0 of
    // that group. Each stage reloads what the previous one stored.
    //
    // Every value gets its own statement: C++ argument evaluation order is
    // unspecified, so inlining these would make the emitted op order
    // compiler-dependent. Per the operand rules above, p1 goes on operand 0, p2
    // on operand 1, and each operand gets its own load.
    mlir::Value s1_shuffled = shuffle(load(), kP1);
    mlir::Value s1_plain = load();
    store(add(s1_shuffled, s1_plain));

    mlir::Value s2_shuffled = shuffle(load(), kP2);
    mlir::Value s2_plain = load();
    store(add(s2_plain, s2_shuffled));

    mlir::Value s3_lhs = shuffle(load(), kP1);
    mlir::Value s3_rhs = shuffle(load(), kP2);
    store(add(s3_lhs, s3_rhs));

    // Phase 2: broadcast each group's lane 0 across its group, then combine the
    // groups. This eval_order leaves the total at lane 0, as the store path
    // expects; it does not match the op's own prefix-scan description, so do not
    // "correct" it from that text.
    mlir::Value v3 = load();
    store(shuffle(v3, kBroadcastLane0));

    mlir::Value broadcast = load();
    auto scan = mlir::vectorchain::ScanWithGapOp::create(
        rewriter, loc, vec_type, broadcast, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::add,
        llvm::APInt(64, kGroupSize),
        mlir::vectorchain::VectorChainScanOpEvalOrders::left_to_right);
    store(scan.getOutput());

    mlir::Value total = load();

    // One shuffle reading lane 0, where the total landed, repeated as many times
    // as the result holds. A full-width result must replicate the total rather
    // than pass the register through: leaving the butterfly's partial sums in
    // lanes 1..63 would contradict what that width claims.
    const int64_t repetition = num_result_lanes;
    mlir::Value result = mlir::vectorchain::ShuffleOp::create(
                             rewriter, loc, result_vec_type, total,
                             /*variable=*/mlir::ValueRange{},
                             /*pad=*/mlir::ValueRange{}, /*mask=*/nullptr,
                             /*dbgName=*/nullptr,
                             rewriter.getArrayAttr({mlir::IntegerAttr::get(
                                 rewriter.getI32Type(), 0)}),
                             static_cast<uint32_t>(repetition))
                             .getOutput();

    rewriter.replaceOp(opaque_op, result);
    return mlir::success();
  }

 private:
  /// Lanes reduced per butterfly group, and equally the number of groups the
  /// scan combines.  Both permutations above are indexed within one group.
  static constexpr int64_t kGroupSize = 8;

  arch_view::ResourceKinds& resource_kinds_;
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext(),
                                          resource_kinds);
  patterns.add<LowerLinalgFillPattern>(patterns.getContext(), resource_kinds);
  patterns.add<LowerLaneReducePattern>(patterns.getContext(), resource_kinds);
}
