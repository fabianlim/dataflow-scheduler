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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ReductionLowering.h"

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace scheduler;

namespace {

// Returns true if `view` is a get_logical_memory_view rooted at the lrfreg
// register-file local unit (get_local_unit {name="lrfreg"}). This mirrors the
// structural accumulator recognition in the RMW-writeback half.
bool isLrfregAccumulatorView(mlir::Value view) {
  auto lmv = view.getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
  if (!lmv) return false;
  auto lu = lmv.getFromUnit().getDefiningOp<mlir::dataflow::GetLocalUnitOp>();
  return lu && lu.getName() == "lrfreg";
}

// Visits ops directly in `loop`'s body but not inside any nested scf.for, so an
// enclosing loop is not mistaken for the innermost accumulation loop.
template <typename OpT, typename FnT>
void walkOwnBody(mlir::scf::ForOp loop, FnT fn) {
  loop.getBody()->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (op != loop.getOperation() && mlir::isa<mlir::scf::ForOp>(op))
      return mlir::WalkResult::skip();
    if (auto casted = mlir::dyn_cast<OpT>(op)) fn(casted);
    return mlir::WalkResult::advance();
  });
}

}  // namespace

// Peel the first iteration of each reduction accumulation loop out of the loop
// and into an unconditional init store before the loop.
//
// A reduction accumulation loop is identified STRUCTURALLY: its body contains a
// vectorchain.binary and an agen.vector_load/store pair that read/write a view
// rooted at the lrfreg accumulator (get_local_unit {name="lrfreg"}). This is
// the accumulate-in-place register-file RMW; it is the exact dual of the
// RMW-writeback detection, which recognizes the same accumulator by its backing
// local unit.
//
// The loop body is:  def_immutable_mapping → query_map → receive → load(acc)
//                    → binary(recv, load) → store(acc).
// After peeling:
//   [before loop] def_immutable_mapping → query_map → receive → store(acc)  // init = reset
//   loop lb → lb + step
//   [loop body unchanged, runs lb+step..ub]
//
// The unconditional init-store is required because a conditional store inside
// scf.if is not honored by hardware. Invariant: a loop is detected as a
// reduction accumulation loop iff it must be peeled — every lrfreg
// accumulate-in-place loop requires the unconditional-reset peel.
void scheduler::peelReductionComputeLoop(mlir::func::FuncOp func) {
  auto* ctx = func.getContext();
  mlir::OpBuilder builder(ctx);

  // Collect reduction accumulation loops: a vectorchain.binary body whose
  // vector_load/store target the lrfreg accumulator view.
  llvm::SmallVector<mlir::scf::ForOp> compute_loops;
  func.walk([&](mlir::scf::ForOp for_op) {
    // Match only the innermost accumulation loop: inspect ops in this loop's
    // own body, not those nested in a deeper scf.for.
    bool has_binary = false;
    walkOwnBody<mlir::vectorchain::BinaryOp>(
        for_op, [&](mlir::vectorchain::BinaryOp) { has_binary = true; });
    if (!has_binary) return;

    bool touches_lrfreg_acc = false;
    walkOwnBody<mlir::agen::VectorLoadOp>(for_op, [&](mlir::agen::VectorLoadOp op) {
      if (isLrfregAccumulatorView(op.getMemRef())) touches_lrfreg_acc = true;
    });
    walkOwnBody<mlir::agen::VectorStoreOp>(
        for_op, [&](mlir::agen::VectorStoreOp op) {
          if (isLrfregAccumulatorView(op.getMemRef())) touches_lrfreg_acc = true;
        });
    assert((!touches_lrfreg_acc || has_binary) &&
           "reduction accumulation loop must have a vectorchain.binary body");
    if (touches_lrfreg_acc) compute_loops.push_back(for_op);
  });

  for (mlir::scf::ForOp loop : compute_loops) {
    mlir::Block* body = loop.getBody();

    // Find the receive op — its channel arg chain (def_immutable_mapping +
    // query_map) will be cloned before the loop.
    mlir::dataflow::ReceiveOp recv_op;
    body->walk([&](mlir::dataflow::ReceiveOp op) {
      if (!recv_op) recv_op = op;
    });
    if (!recv_op) continue;

    // The channel comes from a query_map(def_immutable_mapping, %arg0).
    mlir::Value channel = recv_op.getFromUnit();
    auto qmap_op =
        mlir::dyn_cast_or_null<mlir::uniform::QueryMapOp>(
            channel.getDefiningOp());
    if (!qmap_op) continue;
    mlir::Value map_val = qmap_op.getOperand(0);
    auto dim_op =
        mlir::dyn_cast_or_null<mlir::uniform::DefImmutableMappingOp>(
            map_val.getDefiningOp());
    if (!dim_op) continue;

    // Find the lrfreg accumulator view — it's the memref operand of the
    // vector_load in the loop body.
    mlir::agen::VectorLoadOp load_op;
    body->walk([&](mlir::agen::VectorLoadOp op) {
      if (!load_op) load_op = op;
    });
    if (!load_op) continue;
    mlir::Value acc_view = load_op.getMemRef();

    // Insert the peeled iteration just before the loop.
    builder.setInsertionPoint(loop);
    mlir::Location loc = loop.getLoc();

    // Clone: def_immutable_mapping
    mlir::IRMapping irmap;
    auto peel_dim_op = builder.clone(*dim_op.getOperation(), irmap);

    // Clone: query_map (the key is %arg0 of the enclosing program_unit)
    mlir::Value pu_arg0 = qmap_op.getKey();
    auto peel_qmap = mlir::uniform::QueryMapOp::create(
        builder, loc, qmap_op.getResult().getType(),
        peel_dim_op->getResult(0), pu_arg0);

    // Receive
    auto peel_recv = mlir::dataflow::ReceiveOp::create(
        builder, loc, recv_op.getResult().getType(), peel_qmap.getResult());

    // Init store: acc := received  (no load, no binary — this IS the reset)
    auto identity1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
    int64_t num_lanes =
        mlir::cast<mlir::MemRefType>(acc_view.getType()).getShape()[0];
    mlir::IntegerSet lane_set = buildLaneIntegerSet(ctx, num_lanes);
    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    mlir::agen::VectorStoreOp::create(builder, loc, peel_recv.getData(),
                                      acc_view, /*dbgName=*/nullptr,
                                      identity1d, mlir::ValueRange{c0},
                                      lane_set, identity1d);

    // Advance loop lower bound by one step (lb → lb + step).
    mlir::Value new_lb = mlir::arith::AddIOp::create(
        builder, loc, loop.getLowerBound(), loop.getStep());
    loop.setLowerBound(new_lb);
  }
}

void scheduler::emitReductionWriteBack(
    mlir::PatternRewriter& rewriter, mlir::linalg::GenericOp generic_op,
    mlir::bufferization::ToTensorOp acc_to_tensor, mlir::Value result,
    mlir::Value acc_view) {
  // Emit vector_store for the new accumulator.
  emitLrfregVectorStore(rewriter, generic_op.getLoc(), result, acc_view);
  // Erase the materialize_in_destination bridge op for the generic result.
  mlir::Value generic_result = generic_op.getResult(0);
  for (mlir::OpOperand& use :
       llvm::make_early_inc_range(generic_result.getUses())) {
    if (auto mid =
            mlir::dyn_cast<mlir::bufferization::MaterializeInDestinationOp>(
                use.getOwner())) {
      if (mid.getDest() == acc_view) {
        rewriter.eraseOp(mid);
        break;
      }
    }
  }
  rewriter.eraseOp(generic_op);
  if (acc_to_tensor.getResult().use_empty())
    rewriter.eraseOp(acc_to_tensor);
}
