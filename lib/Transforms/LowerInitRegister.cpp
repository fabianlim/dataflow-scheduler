//===-- LowerInitRegister.cpp -----------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep

namespace scheduler {
#define GEN_PASS_DEF_LOWERINITREGISTERPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace mlir;

namespace scheduler {
namespace {

/// Gets the closest stage or pipeline enclosing \p op, or null if there is
/// none.
auto getEnclosingScope(Operation* op) -> Operation* {
  for (auto* parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<ktdf::StageOp, ktdf::PipelineOp>(parent)) return parent;
  }
  return nullptr;
}

/// Whether \p stage may run a fill on \p unit: it lists the unit among its
/// applicable units, or declares none.
auto stageRunsOn(ktdf::StageOp stage, StringRef unit) -> bool {
  const auto units = stage.getApplicableUnitsAttr();
  if (!units) return true;
  return llvm::any_of(units, [&](Attribute attr) {
    const auto name = llvm::dyn_cast<StringAttr>(attr);
    return name && name.getValue() == unit;
  });
}

/// Replaces \p init with an allocation filled with its value.
auto lowerInitRegister(ktdf::InitRegisterOp init, RewriterBase& rewriter)
    -> LogicalResult {
  auto* const scope = getEnclosingScope(init);
  auto stage = llvm::dyn_cast_or_null<ktdf::StageOp>(scope);
  if (scope && !stage) {
    return init.emitOpError("must be inside a stage of its pipeline");
  }
  if (stage && !stageRunsOn(stage, init.getUnit())) {
    return init.emitOpError("unit \"")
           << init.getUnit()
           << "\" is not an applicable unit of the enclosing stage";
  }

  const auto loc = init.getLoc();
  rewriter.setInsertionPoint(init);
  auto value = arith::ConstantOp::create(rewriter, loc, init.getValue());
  auto reg = memref::AllocOp::create(rewriter, loc, init.getResult().getType());

  if (stage) {
    // The enclosing stage runs the fill, right after the allocation.
    linalg::FillOp::create(rewriter, loc, ValueRange{value.getResult()},
                           ValueRange{reg.getResult()});
  } else {
    // A pipeline of one stage on the unit runs the fill, right after the
    // allocation.
    ktdf::PipelineOp::create(
        rewriter, loc, [&](OpBuilder& builder, Location /*loc*/) {
          auto fill_stage = ktdf::StageOp::create(
              builder, loc, ValueRange{}, ValueRange{},
              [&](OpBuilder& builder, Location /*loc*/) {
                linalg::FillOp::create(builder, loc,
                                       ValueRange{value.getResult()},
                                       ValueRange{reg.getResult()});
              });
          fill_stage.setApplicableUnitsAttr(
              builder.getArrayAttr({init.getUnitAttr()}));
        });
  }

  rewriter.replaceOp(init, reg.getResult());
  return success();
}

struct LowerInitRegisterPass
    : public impl::LowerInitRegisterPassBase<LowerInitRegisterPass> {
  using LowerInitRegisterPassBase::LowerInitRegisterPassBase;

  void runOnOperation() override {
    SmallVector<ktdf::InitRegisterOp> inits;
    getOperation()->walk(
        [&](ktdf::InitRegisterOp init) { inits.push_back(init); });

    IRRewriter rewriter(&getContext());
    bool failed_any = false;
    for (auto init : inits) {
      if (failed(lowerInitRegister(init, rewriter))) failed_any = true;
    }

    if (failed_any) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace scheduler
