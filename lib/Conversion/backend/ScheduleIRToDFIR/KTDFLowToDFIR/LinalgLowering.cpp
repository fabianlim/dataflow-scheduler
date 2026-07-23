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
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
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
              // arith.mulf %lhs, %rhs -> vectorchain.binary {binary_op = mul}
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp mulf_op) {
                return lowerMulFOp(mulf_op, rewriter, identity_map);
              })
              // arith.addf %lhs, %rhs -> vectorchain.binary {binary_op = add}
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp addf_op) {
                return lowerAddFOp(addf_op, rewriter, identity_map);
              })
              // arith.subf %lhs, %rhs -> vectorchain.binary {binary_op = sub}
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp subf_op) {
                return lowerSubFOp(subf_op, rewriter, identity_map);
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

  // arith.mulf visitor: lowers to vectorchain.binary {binary_op = mul}
  mlir::LogicalResult lowerMulFOp(mlir::arith::MulFOp mulf_op,
                                  mlir::PatternRewriter& rewriter,
                                  mlir::AffineMap identity_map) const {
    auto vector_type =
        getFlattenedVectorType(mulf_op.getLhs().getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, mulf_op.getLoc(), vector_type, mulf_op.getLhs(),
        mulf_op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::mul, identity_map);

    rewriter.replaceOp(mulf_op, binary_op.getData());
    return mlir::success();
  }

  // arith.addf visitor: lowers to vectorchain.binary {binary_op = add}
  mlir::LogicalResult lowerAddFOp(mlir::arith::AddFOp addf_op,
                                  mlir::PatternRewriter& rewriter,
                                  mlir::AffineMap identity_map) const {
    auto vector_type =
        getFlattenedVectorType(addf_op.getLhs().getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, addf_op.getLoc(), vector_type, addf_op.getLhs(),
        addf_op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::add, identity_map);

    rewriter.replaceOp(addf_op, binary_op.getData());
    return mlir::success();
  }

  // arith.subf visitor: lowers to vectorchain.binary {binary_op = sub}
  mlir::LogicalResult lowerSubFOp(mlir::arith::SubFOp subf_op,
                                  mlir::PatternRewriter& rewriter,
                                  mlir::AffineMap identity_map) const {
    auto vector_type =
        getFlattenedVectorType(subf_op.getLhs().getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, subf_op.getLoc(), vector_type, subf_op.getLhs(),
        subf_op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::sub, identity_map);

    rewriter.replaceOp(subf_op, binary_op.getData());
    return mlir::success();
  }
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext(),
                                          resource_kinds);
}
