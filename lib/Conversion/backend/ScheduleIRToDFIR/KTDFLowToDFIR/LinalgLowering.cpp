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
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
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

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) {
      return mlir::failure();
    }

    // Replace block arguments with generic inputs
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      block_arg.replaceAllUsesWith(input);
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

/// Build an IntegerSet from a static shape. Each dimension of size 1 is
/// constrained to dk == 0; each dimension of size n > 1 is constrained to
/// 0 <= dk <= n-1 (two inequalities). The set has `rank` dimensions and no
/// symbols.
static mlir::IntegerSet buildSetFromShape(mlir::MLIRContext* ctx,
                                          llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<mlir::AffineExpr> exprs;
  llvm::SmallVector<bool> eq_flags;
  for (unsigned i = 0; i < shape.size(); ++i) {
    auto dim = mlir::getAffineDimExpr(i, ctx);
    int64_t sz = shape[i];
    if (sz == 1) {
      exprs.push_back(dim);
      eq_flags.push_back(true);
    } else {
      exprs.push_back(dim);
      eq_flags.push_back(false);
      exprs.push_back(mlir::getAffineConstantExpr(sz - 1, ctx) - dim);
      eq_flags.push_back(false);
    }
  }
  return mlir::IntegerSet::get(shape.size(), 0, exprs, eq_flags);
}

/// Pattern to lower linalg.fill with buffer semantics on an lrfreg-backed
/// memref (after get_logical_memory_view resolution).
///
///   linalg.fill ins(%scalar : f16) outs(%view : memref<..., ...>)
///
/// Lowers to:
///   %zero_vec = arith.constant dense<0.0> : vector<N x f16>
///   agen.vector_store %zero_vec, %view[%c0, ..., %c0] {store_set, store_order}
struct LowerLinalgFillPattern
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  LowerLinalgFillPattern(mlir::MLIRContext* context,
                         arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::FillOp fill_op,
      mlir::PatternRewriter& rewriter) const override {
    if (!fill_op.hasPureBufferSemantics()) return mlir::failure();

    mlir::Value output = fill_op.output();

    auto memref_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    if (!memref_type) return mlir::failure();

    // Compute total elements from memref shape.
    auto shape = memref_type.getShape();
    int64_t total_elements = 1;
    for (int64_t dim : shape) {
      total_elements *= dim;
    }

    auto elem_type = memref_type.getElementType();
    auto vector_type = mlir::VectorType::get({total_elements}, elem_type);

    // Build a dense zero vector constant.
    auto zero_attr = mlir::DenseElementsAttr::get(
        vector_type, llvm::ArrayRef<mlir::Attribute>{
                         rewriter.getFloatAttr(elem_type, 0.0)});
    auto zero_vec =
        mlir::arith::ConstantOp::create(rewriter, fill_op.getLoc(), zero_attr);

    // Build affine map (identity) and integer set from the memref shape.
    unsigned rank = memref_type.getRank();
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        rank, rewriter.getContext());
    auto store_set = buildSetFromShape(rewriter.getContext(), shape);

    // Build zero index operands for the store address.
    auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, fill_op.getLoc(), 0);
    llvm::SmallVector<mlir::Value> indices(rank, c0.getResult());

    mlir::agen::VectorStoreOp::create(
        rewriter, fill_op.getLoc(), zero_vec.getResult(), output,
        /*dbgName=*/nullptr, store_order, indices, store_set, store_order);

    rewriter.eraseOp(fill_op);
    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;
};

/// Pattern to lower a buffer-semantics linalg.generic reduction accumulate:
///
///   linalg.generic {iterator_types = [..., "reduction", ...]}
///     ins(%row : memref<...>) outs(%acc : memref<...>) {
///   ^bb0(%in: f16, %out: f16):
///     %r = arith.addf %in, %out : f16
///     linalg.yield %r : f16
///   }
///
/// Lowers to:
///   %in_vec  = agen.vector_load  %row[%c0,...] {load_set, load_order}
///   %acc_vec = agen.vector_load  %acc[%c0,...] {load_set, load_order}
///   %sum     = vectorchain.binary %in_vec, %acc_vec {binary_op = add}
///   agen.vector_store %sum, %acc[%c0,...] {store_set, store_order}
struct LowerLinalgGenericBufferAccumulatePattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LowerLinalgGenericBufferAccumulatePattern(
      mlir::MLIRContext* context, arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const override {
    // Must be buffer-semantics with no SSA result.
    if (!generic_op.hasPureBufferSemantics()) return mlir::failure();
    if (generic_op.getNumResults() != 0) return mlir::failure();

    // Must have at least one reduction iterator.
    auto iter_types = generic_op.getIteratorTypesArray();
    bool has_reduction = llvm::any_of(iter_types, [](mlir::utils::IteratorType t) {
      return t == mlir::utils::IteratorType::reduction;
    });
    if (!has_reduction) return mlir::failure();

    // Body must be: arith.addf + linalg.yield.
    mlir::Block& body = generic_op.getRegion().front();
    auto* term = body.getTerminator();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(term);
    if (!yield_op || yield_op.getNumOperands() != 1) return mlir::failure();

    auto* add_op_base = yield_op.getOperand(0).getDefiningOp();
    auto addf_op = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(add_op_base);
    if (!addf_op) return mlir::failure();

    // Exactly one input (ins[0]) and one output (outs[0]).
    if (generic_op.getNumDpsInputs() != 1 ||
        generic_op.getNumDpsInits() != 1)
      return mlir::failure();

    mlir::Value input = generic_op.getDpsInputs()[0];
    mlir::Value acc   = generic_op.getDpsInits()[0];

    auto in_memref_type  = mlir::dyn_cast<mlir::MemRefType>(input.getType());
    auto acc_memref_type = mlir::dyn_cast<mlir::MemRefType>(acc.getType());
    if (!in_memref_type || !acc_memref_type) return mlir::failure();

    // Compute total elements from the input shape.
    auto in_shape = in_memref_type.getShape();
    int64_t total_elements = 1;
    for (int64_t dim : in_shape) total_elements *= dim;

    auto elem_type  = in_memref_type.getElementType();
    auto vector_type = mlir::VectorType::get({total_elements}, elem_type);

    auto loc = generic_op.getLoc();
    auto c0  = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    // Build affine map and integer set for input load.
    unsigned in_rank  = in_memref_type.getRank();
    unsigned acc_rank = acc_memref_type.getRank();
    auto in_load_order = mlir::AffineMap::getMultiDimIdentityMap(
        in_rank, rewriter.getContext());
    auto in_load_set = buildSetFromShape(rewriter.getContext(), in_shape);
    llvm::SmallVector<mlir::Value> in_indices(in_rank, c0.getResult());

    // Build affine map and integer set for accumulator load/store.
    auto acc_shape = acc_memref_type.getShape();
    auto acc_order = mlir::AffineMap::getMultiDimIdentityMap(
        acc_rank, rewriter.getContext());
    auto acc_set = buildSetFromShape(rewriter.getContext(), acc_shape);
    llvm::SmallVector<mlir::Value> acc_indices(acc_rank, c0.getResult());

    // agen.vector_load from input row.
    auto in_vec = mlir::agen::VectorLoadOp::create(
        rewriter, loc, vector_type, input,
        /*dbgName=*/nullptr, in_load_order, in_indices, in_load_set,
        in_load_order);

    // agen.vector_load from accumulator.
    auto acc_vec = mlir::agen::VectorLoadOp::create(
        rewriter, loc, vector_type, acc,
        /*dbgName=*/nullptr, acc_order, acc_indices, acc_set, acc_order);

    // vectorchain.binary {binary_op = add}.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());
    auto sum = mlir::vectorchain::BinaryOp::create(
        rewriter, loc, vector_type, in_vec.getResult(), acc_vec.getResult(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::add, identity_map);

    // agen.vector_store result into accumulator.
    mlir::agen::VectorStoreOp::create(
        rewriter, loc, sum.getData(), acc,
        /*dbgName=*/nullptr, acc_order, acc_indices, acc_set, acc_order);

    rewriter.eraseOp(generic_op);
    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext(),
                                          resource_kinds);
  patterns.add<LowerLinalgFillPattern>(patterns.getContext(), resource_kinds);
  patterns.add<LowerLinalgGenericBufferAccumulatePattern>(patterns.getContext(),
                                                          resource_kinds);
}
