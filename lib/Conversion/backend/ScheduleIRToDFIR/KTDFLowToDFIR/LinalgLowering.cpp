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
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Build IntegerSet: d0 in [0, 63] (all 64 lanes of the lrfreg vector).
static mlir::IntegerSet buildAccLoadStoreSet(mlir::MLIRContext* ctx) {
  auto d0 = mlir::getAffineDimExpr(0, ctx);
  // d0 >= 0  &&  -d0 + 63 >= 0
  return mlir::IntegerSet::get(1, 0, {d0, -d0 + 63}, {/*eq=*/false, /*eq=*/false});
}

/// Emit agen.vector_load from a 1-D lrfreg memref view at index 0.
static mlir::Value emitAccVectorLoad(mlir::OpBuilder& builder,
                                     mlir::Location loc, mlir::Value view) {
  auto* ctx = builder.getContext();
  auto memref_type = mlir::cast<mlir::MemRefType>(view.getType());
  assert(memref_type.getRank() == 1 && "expected 1-D lrfreg view");
  int64_t num_elems = memref_type.getShape()[0];
  auto elem_type = memref_type.getElementType();
  auto vector_type = mlir::VectorType::get({num_elems}, elem_type);
  auto identity1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
  auto load_set = buildAccLoadStoreSet(ctx);
  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  return mlir::agen::VectorLoadOp::create(builder, loc, vector_type, view,
                                          /*dbgName=*/nullptr, identity1d,
                                          mlir::ValueRange{c0}, load_set,
                                          identity1d)
      .getResult();
}

/// Emit agen.vector_store of `vec` to a 1-D lrfreg memref view at index 0.
static void emitAccVectorStore(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value vec, mlir::Value view) {
  auto* ctx = builder.getContext();
  auto identity1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
  auto store_set = buildAccLoadStoreSet(ctx);
  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::agen::VectorStoreOp::create(builder, loc, vec, view,
                                    /*dbgName=*/nullptr, identity1d,
                                    mlir::ValueRange{c0}, store_set, identity1d);
}

/// Pattern to lower linalg.generic compute operations
struct LowerLinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LowerLinalgGenericPattern(mlir::MLIRContext* context,
                            const SchedulerExtContext& scheduler_ctx,
                            arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context),
        scheduler_ctx_(scheduler_ctx),
        resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const override {
    if (!generic_op.hasPureTensorSemantics() ||
        generic_op.getNumResults() != 1) {
      return mlir::failure();
    }

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = llvm::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) {
      return mlir::failure();
    }

    // Detect the reduction accumulator: the op has exactly one DPS init defined
    // by a bufferization.to_tensor of the lrfreg accumulator view.
    // NOTE: by the time this pattern runs, buildLogicalMemoryViews (earlier in
    // this same pass) has already replaced the "lrfreg"-space
    // unrealized_conversion_cast with a dataflow.get_logical_memory_view of PLAIN
    // memref type (the memory space is stripped). So we cannot test the memref's
    // memory space; instead we recognize the accumulator by its backing unit:
    // to_tensor(get_logical_memory_view(get_local_unit {name="lrfreg"}, ...)).
    mlir::bufferization::ToTensorOp accToTensor;
    mlir::Value acc_view;
    if (generic_op.getNumDpsInits() == 1) {
      mlir::Value init_val = generic_op.getDpsInits()[0];
      if (auto tt =
              init_val.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
        mlir::Value buf = tt.getBuffer();
        if (auto view =
                buf.getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>()) {
          if (auto lu = view.getFromUnit()
                            .getDefiningOp<mlir::dataflow::GetLocalUnitOp>()) {
            if (lu.getName() == "lrfreg") {
              accToTensor = tt;
              acc_view = buf;
            }
          }
        }
      }
    }

    bool is_reduction = static_cast<bool>(accToTensor);

    // Replace block arguments with generic inputs
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      block_arg.replaceAllUsesWith(input);
    }

    // Replace the outs (DPS init) block args.
    // For a reduction: load the accumulator from lrfreg and use that as the
    // outs block arg replacement (the acc_vec from agen.vector_load).
    // For elementwise: use the DPS init value directly (outs block arg is
    // unused, so this is a no-op).
    unsigned num_outputs = generic_op.getNumDpsInits();

    // For the reduction case we emit the vector_load before the block-arg
    // replacement so we can use it as the outs replacement.
    mlir::Value acc_vec;
    if (is_reduction) {
      rewriter.setInsertionPoint(generic_op);
      acc_vec = emitAccVectorLoad(rewriter, generic_op.getLoc(), acc_view);
      // Replace outs block arg with acc_vec.
      body.getArguments()
          .drop_front(num_inputs)
          .take_front(num_outputs)[0]
          .replaceAllUsesWith(acc_vec);
    } else {
      for (auto [block_arg, init] :
           llvm::zip(
               body.getArguments().drop_front(num_inputs).take_front(
                   num_outputs),
               generic_op.getDpsInits())) {
        block_arg.replaceAllUsesWith(init);
      }
    }

    // Identity affine map used as op_specific_map for binary ops.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    // Collect compute operations to replace
    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator()) {
      ops_to_lower.push_back(&op);
    }

    // Process and lower compute operations
    rewriter.setInsertionPoint(generic_op);

    // Fused multiply-accumulate:  %m = mulf %a,%b ; %s = addf %m,%c ; yield %s
    // -> single vectorchain.multiply_and_accumulate %a,%b,%c [%mask] {reduction_map}.
    // A matmul/contraction generalizes to exactly this body; emitting two chained
    // vectorchain.binary ops instead crashes the backend PE result-forwarding.
    if (ops_to_lower.size() == 2) {
      auto mulf = llvm::dyn_cast<mlir::arith::MulFOp>(ops_to_lower[0]);
      auto addf = llvm::dyn_cast<mlir::arith::AddFOp>(ops_to_lower[1]);
      if (mulf && addf && yield_op.getOperand(0) == addf.getResult()) {
        mlir::Value acc;
        if (addf.getLhs() == mulf.getResult()) {
          acc = addf.getRhs();
        } else if (addf.getRhs() == mulf.getResult()) {
          acc = addf.getLhs();
        }
        if (acc) {
          auto vector_type = getFlattenedVectorType(mulf.getLhs().getType(),
                                                    resource_kinds_);
          if (!vector_type) return mlir::failure();

          mlir::MLIRContext* ctx = rewriter.getContext();
          // reduction_map: identity over the single (lane) dim — elementwise per
          // output lane, no cross-lane reduction (matches the matmul target).
          mlir::AffineMap reduction_map =
              mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
          // mask_set: mirror the pristine matmul_ep target's create_affine_mask
          // (dcc_backend ignores it numerically here; kept for fidelity).
          auto d0 = mlir::getAffineDimExpr(0, ctx);
          mlir::IntegerSet mask_set = mlir::IntegerSet::get(
              /*dimCount=*/1, /*symbolCount=*/0,
              {d0 - 64, (-d0) + 63}, {/*eq=*/false, /*eq=*/false});
          auto mask_type = mlir::VectorType::get(
              llvm::cast<mlir::VectorType>(vector_type).getShape(),
              rewriter.getI1Type());
          auto mask = mlir::vectorchain::CreateAffineMaskOp::create(
              rewriter, generic_op.getLoc(), mask_type,
              /*mask_parameter=*/mlir::Value(),
              mlir::IntegerSetAttr::get(mask_set));
          auto mac = mlir::vectorchain::MultiplyAndAccumulateOp::create(
              rewriter, generic_op.getLoc(), vector_type, mulf.getLhs(),
              mulf.getRhs(), acc, mask.getData(), /*dbgName=*/nullptr,
              reduction_map);

          if (is_reduction) {
            // Emit vector_store for the accumulator result.
            mlir::Value new_acc = mac.getData();
            emitAccVectorStore(rewriter, generic_op.getLoc(), new_acc, acc_view);
            // Erase site #3: the materialize_in_destination whose dest is
            // acc_view and whose source is the generic op's result.
            mlir::Value generic_result = generic_op.getResult(0);
            for (mlir::OpOperand& use :
                 llvm::make_early_inc_range(generic_result.getUses())) {
              if (auto mid = mlir::dyn_cast<
                      mlir::bufferization::MaterializeInDestinationOp>(
                      use.getOwner())) {
                if (mid.getDest() == acc_view) {
                  rewriter.eraseOp(mid);
                  break;
                }
              }
            }
            rewriter.eraseOp(generic_op);
            if (accToTensor.getResult().use_empty())
              rewriter.eraseOp(accToTensor);
          } else {
            rewriter.replaceOp(generic_op, mac.getData());
          }
          return mlir::success();
        }
      }
    }

    // Plain binary path (single op: addf, mulf, subf).
    // Lower each body op to vectorchain.binary.
    for (mlir::Operation* op : ops_to_lower) {
      // arith.mulf %lhs, %rhs -> vectorchain.binary {binary_op = mul}
      if (auto mulf_op = llvm::dyn_cast<mlir::arith::MulFOp>(op)) {
        auto vector_type =
            getFlattenedVectorType(mulf_op.getLhs().getType(), resource_kinds_);
        if (!vector_type) return mlir::failure();

        auto binary_op = mlir::vectorchain::BinaryOp::create(
            rewriter, mulf_op.getLoc(), vector_type, mulf_op.getLhs(),
            mulf_op.getRhs(),
            /*mask=*/nullptr, /*dbgName=*/nullptr,
            mlir::vectorchain::VectorChainBinaryOperator::mul, identity_map);

        rewriter.replaceOp(mulf_op, binary_op.getData());
        continue;
      }

      // arith.addf %lhs, %rhs -> vectorchain.binary {binary_op = add}
      if (auto addf_op = llvm::dyn_cast<mlir::arith::AddFOp>(op)) {
        auto vector_type =
            getFlattenedVectorType(addf_op.getLhs().getType(), resource_kinds_);
        if (!vector_type) return mlir::failure();

        auto binary_op = mlir::vectorchain::BinaryOp::create(
            rewriter, addf_op.getLoc(), vector_type, addf_op.getLhs(),
            addf_op.getRhs(),
            /*mask=*/nullptr, /*dbgName=*/nullptr,
            mlir::vectorchain::VectorChainBinaryOperator::add, identity_map);

        rewriter.replaceOp(addf_op, binary_op.getData());
        continue;
      }

      // arith.subf %lhs, %rhs -> vectorchain.binary {binary_op = sub}
      if (auto subf_op = llvm::dyn_cast<mlir::arith::SubFOp>(op)) {
        auto vector_type =
            getFlattenedVectorType(subf_op.getLhs().getType(), resource_kinds_);
        if (!vector_type) return mlir::failure();

        auto binary_op = mlir::vectorchain::BinaryOp::create(
            rewriter, subf_op.getLoc(), vector_type, subf_op.getLhs(),
            subf_op.getRhs(),
            /*mask=*/nullptr, /*dbgName=*/nullptr,
            mlir::vectorchain::VectorChainBinaryOperator::sub, identity_map);

        rewriter.replaceOp(subf_op, binary_op.getData());
        continue;
      }
    }

    // After the binary body ops are lowered, the yield operand is now a
    // vectorchain result (or the original value for elementwise no-body cases).
    mlir::Value result = yield_op.getOperand(0);

    if (is_reduction) {
      // Emit vector_store for the new accumulator.
      emitAccVectorStore(rewriter, generic_op.getLoc(), result, acc_view);
      // Erase site #3: the materialize_in_destination for the generic result.
      mlir::Value generic_result = generic_op.getResult(0);
      for (mlir::OpOperand& use :
           llvm::make_early_inc_range(generic_result.getUses())) {
        if (auto mid = mlir::dyn_cast<
                mlir::bufferization::MaterializeInDestinationOp>(
                use.getOwner())) {
          if (mid.getDest() == acc_view) {
            rewriter.eraseOp(mid);
            break;
          }
        }
      }
      rewriter.eraseOp(generic_op);
      if (accToTensor.getResult().use_empty())
        rewriter.eraseOp(accToTensor);
    } else {
      // Replace the generic op with the yield operand
      rewriter.replaceOp(generic_op, result);
    }
    return mlir::success();
  }

 private:
  const SchedulerExtContext& scheduler_ctx_;
  arch_view::ResourceKinds& resource_kinds_;
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns, const SchedulerExtContext& scheduler_ctx,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext(), scheduler_ctx,
                                          resource_kinds);
}
