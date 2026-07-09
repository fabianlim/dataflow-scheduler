//===-- PipelineScope.cpp ---------------------------------------*- c++ -*-===//
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
// This file implements the pipeline scope analysis.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/PipelineScope.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

/// True iff `loop`'s body contains no `ktdf.pipeline` op other than ops
/// transitively inside `chain_member` (the op on the path back to the
/// pipeline being scoped).
bool loopHasNoPeerPipeline(scf::ForOp loop, Operation* chain_member) {
  bool found = false;
  loop.getBody()->walk([&](PipelineOp pipeline) {
    if (found) return WalkResult::interrupt();
    // Skip pipelines that are inside chain_member's subtree.
    if (chain_member->isAncestor(pipeline.getOperation()) ||
        chain_member == pipeline.getOperation()) {
      return WalkResult::skip();
    }
    found = true;
    return WalkResult::interrupt();
  });
  return !found;
}

/// True iff `op` is a scope-terminating boundary for the enclosing-scope
/// walk: anything that is not an scf.for (stage, pipeline, parallel, func,
/// etc.) stops the walk.
bool isScopeBoundary(Operation* op) { return !isa<scf::ForOp>(op); }

}  // namespace

auto mlir::ktdf::getPipelineEnclosingScope(PipelineOp pipeline)
    -> PipelineEnclosingScope {
  PipelineEnclosingScope scope;
  Operation* current = pipeline.getOperation();
  Operation* parent = current->getParentOp();
  while (parent) {
    if (isScopeBoundary(parent)) {
      break;
    }
    auto for_op = cast<scf::ForOp>(parent);
    // A loop annotated parallel_loop replicates its entire body per instance,
    // so multiple enclosed pipelines legitimately share that scope.  Bypass the
    // peer-pipeline gate for such loops; keep the gate for all other loops.
    if (!isParallelLoop(for_op) && !loopHasNoPeerPipeline(for_op, current)) {
      break;
    }
    scope.loops.push_back(for_op);
    current = for_op.getOperation();
    parent = current->getParentOp();
  }
  return scope;
}
