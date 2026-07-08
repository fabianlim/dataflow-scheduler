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
/// KTDFLowToDFIR pass: creates one dataflow.program_unit per component type
/// from the KTDFLowering execute_on / signal IR.
///
//
//===----------------------------------------------------------------------===//

#include <mlir/Transforms/RegionUtils.h>

#include "Ktdp/KtdpDialect.hpp"
#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LogicalMemoryViewBuilder.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/OperationLowerings.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/PreludeWorkPartition.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ProgramUnitBuilder.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/QueryMapArithCollapse.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "ktdflowering-to-dfir"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTDFLOWTODFIRPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// Peel the first iteration of each compute reduction loop (identified by the
// ktdf.reduction_accumulator attribute + a vectorchain.binary body) out of the
// loop and into an unconditional init store before the loop.
//
// The loop body is:  def_immutable_mapping → query_map → receive → load(acc)
//                    → binary(recv, load) → store(acc).
// After peeling:
//   [before loop] def_immutable_mapping → query_map → receive → store(acc)  // init = reset
//   loop lb → lb + step
//   [loop body unchanged, runs lb+step..ub]
//
// This is the reset fix: an unconditional init-store replaces the prior
// conditional-store seed, which the hardware ignored (see §9.12 design doc).
// Only the COMPUTE loop (body has vectorchain.binary) is peeled; transfer loops
// that share the ktdf.reduction_accumulator attr are left unchanged.
static void peelReductionComputeLoop(mlir::func::FuncOp func) {
  auto* ctx = func.getContext();
  mlir::OpBuilder builder(ctx);

  // Collect compute reduction loops (ktdf.reduction_accumulator + binary body).
  llvm::SmallVector<mlir::scf::ForOp> compute_loops;
  func.walk([&](mlir::scf::ForOp for_op) {
    if (!for_op->hasAttr("ktdf.reduction_accumulator")) return;
    bool has_binary = false;
    for_op.getBody()->walk([&](mlir::vectorchain::BinaryOp) {
      has_binary = true;
    });
    if (has_binary) compute_loops.push_back(for_op);
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
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, ctx);
    // lane_set: d0 in [0, 63]  (same as buildAccLoadStoreSet in LinalgLowering)
    mlir::IntegerSet lane_set = mlir::IntegerSet::get(
        1, 0, {d0, -d0 + 63}, {false, false});
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

// Run the full canonicalization pattern set (all loaded dialects + registered
// ops) with region simplification (DCE) on `func`. Mirrors the upstream
// Canonicalizer pass; best-effort (non-convergence is not a failure).
static void canonicalizeFunc(mlir::func::FuncOp func) {
  mlir::MLIRContext* ctx = func.getContext();
  mlir::RewritePatternSet patterns(ctx);
  for (mlir::Dialect* dialect : ctx->getLoadedDialects()) {
    dialect->getCanonicalizationPatterns(patterns);
  }
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperations()) {
    op.getCanonicalizationPatterns(patterns, ctx);
  }
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  // Default GreedyRewriteConfig enables region simplification (DCE).
  (void)mlir::applyPatternsGreedily(func, frozen);
}

struct KTDFLowToDFIRPass
    : public impl::KTDFLowToDFIRPassBase<KTDFLowToDFIRPass> {
  KTDFLowToDFIRPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Construct MemoryTree from DeviceManager once for the whole module
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME << "] No device found\n");
      return;
    }
    scheduler::arch_view::MemoryTree memory_tree(*device);
    auto& resource_kinds = getChildAnalysis<arch_view::ResourceKinds>(**device);

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    for (auto func : funcs) {
      LLVM_DEBUG(llvm::dbgs() << "Running " << PASS_NAME << " on "
                              << func.getName() << "\n");

      auto split = partitionPreludeAndWork(func);
      if (split.work_ops.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  no work ops; skipping function\n");
        continue;
      }

      ResourceToUnits components;
      if (mlir::failed(discoverUnitTypes(split.work_ops, components))) {
        return signalPassFailure();
      }
      if (components.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  no ktdf_lowering.execute_on ops; skipping function\n");
      } else {
        if (mlir::failed(buildProgramUnits(func, split.work_ops, components))) {
          return signalPassFailure();
        }
        if (mlir::failed(scheduler::collapseArithOverQueryMaps(func))) {
          return signalPassFailure();
        }
      }

      // Clean up side-effect-free ops left dead after extraction (e.g. ops that
      // were cloned into child modules but whose originals are now unused in
      // the top-level module). runRegionDCE recurses into nested regions, so
      // passing the module's regions covers child modules created above.
      mlir::IRRewriter rewriter(module.getContext());
      (void)mlir::runRegionDCE(rewriter, func.getBody());

      if (mlir::failed(
              buildLogicalMemoryViews(func, memory_tree, scheduler_ctx_))) {
        return signalPassFailure();
      }

      // Collapse arith over query_map AGAIN: buildLogicalMemoryViews may create
      // new arith.addi(query_map, constant) patterns (e.g. base_addr + offset)
      // that did not exist during the first collapse pass above.
      if (mlir::failed(scheduler::collapseArithOverQueryMaps(func))) {
        return signalPassFailure();
      }

      // Run operation lowerings after program units have been created
      LLVM_DEBUG(llvm::dbgs() << "Running operation lowerings for "
                              << func.getName() << "\n");
      if (mlir::failed(runOperationLowerings(func, schedulerExtContext(),
                                             components, resource_kinds))) {
        return signalPassFailure();
      }

      // Peel the first iteration of each compute reduction loop so the
      // lrfreg accumulator is initialised by an unconditional store (the
      // reset).  Must run after runOperationLowerings (which produces the
      // final receive / vector_load / binary / vector_store ops) and before
      // canonicalizeFunc (which may fold the new lb arithmetic).
      peelReductionComputeLoop(func);

      // Final cleanup: canonicalize + DCE the fully-lowered function so the
      // emitted DFIR has no dead/duplicate/foldable leftovers (duplicate
      // folded constants from query-map collapse, addi(const,const), dead
      // NoMemoryEffect ops).
      canonicalizeFunc(func);
    }
  }

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTDFLowToDFIRPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<KTDFLowToDFIRPass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> scheduler::createKTDFLowToDFIRPass() {
  return std::make_unique<KTDFLowToDFIRPass>(
      SchedulerExtContext::dummyContext());
}
