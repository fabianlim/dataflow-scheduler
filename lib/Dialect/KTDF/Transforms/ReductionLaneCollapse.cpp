//===-- ReductionLaneCollapse.cpp -------------------------------*- c++ -*-===//
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
// ReductionLaneCollapse: take the in-register reduction out of a
// linalg.generic's reduction set and hand it to a ktdf.opaque intrinsic
// instead.
//
// linalg spells in-register and cross-register reduction identically, but the
// downstream passes implement only the cross-register kind: they turn a
// reduction axis into a loop and divide the input FIFO by its extent. Applied
// to the input's fastest-varying axis -- whose elements are contiguous and so
// arrive together in one vector -- that is correct but degenerate: the FIFO
// shrinks to one element and the loop steps a scalar at a time, discarding the
// vector the hardware was going to operate on.
//
// So this must run before anything downstream inspects the reduction set. The
// invariant it establishes is that every remaining reduction axis steps between
// registers, which is why none of the later machinery needs a special case; do
// not fold this back into them.
//
// The intrinsic yields the shape the original reduction produced, so its
// consumers never need adjusting.
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/OpaqueTemplates.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "reduction-lane-collapse"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONLANECOLLAPSEPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Reduction Lane Collapse pass"),
    llvm::cl::init(false));

namespace {

// The dimension whose lanes share a register, if this op reduces one.
//
// A reduction is in-register exactly when the summed axis is the input's own
// fastest-varying axis -- the last result of the input's indexing map. Any
// other reduction axis steps between whole registers, which the ordinary
// accumulate machinery already handles; std::nullopt for those.
std::optional<unsigned> findInRegisterReductionDim(
    linalg::GenericOp generic_op) {
  if (generic_op.getNumDpsInputs() < 1) return std::nullopt;
  AffineMap in_map =
      generic_op.getMatchingIndexingMap(generic_op.getDpsInputOperand(0));
  if (in_map.getNumResults() == 0) return std::nullopt;

  auto fastest =
      dyn_cast<AffineDimExpr>(in_map.getResult(in_map.getNumResults() - 1));
  if (!fastest) return std::nullopt;

  auto iter_types = generic_op.getIteratorTypesArray();
  unsigned dim = fastest.getPosition();
  if (iter_types[dim] != utils::IteratorType::reduction) return std::nullopt;
  return dim;
}

// Rewrite one linalg.generic that reduces the lanes of a register. Two shapes
// reach this, differing in whether the output already has a lane axis:
//
//   No output lane axis. The lane axis is appended to the output map and shape,
//   the reduced dim becomes parallel, and the intrinsic narrows the result back.
//
//   Output lane axis already present, as the fastest-varying parallel dim. The
//   iteration space names one register twice, so the two dims are merged: the
//   input's lane axis is remapped onto the output's and the redundant dim drops
//   out. The output shape is unchanged and the intrinsic keeps the width.
//
// Both leave a generic whose reduction set holds only cross-register axes, so
// nothing downstream needs to know which shape it was.
LogicalResult collapseInRegisterReduction(linalg::GenericOp generic_op,
                                          unsigned lane_dim) {
  // A multi-result generic would need a policy for which result carries the
  // lanes. Refuse rather than skip: skipping leaves the degenerate
  // scalar-at-a-time form for the later reduction machinery to build.
  if (generic_op.getNumDpsInits() != 1 || generic_op->getNumResults() != 1)
    return generic_op.emitError(PASS_NAME)
           << ": a register's lanes are reduced, but the operation has "
           << generic_op.getNumDpsInits() << " outputs and "
           << generic_op->getNumResults()
           << " results; exactly one of each is required to keep the lane "
              "dimension live";

  auto out_type =
      dyn_cast<RankedTensorType>(generic_op.getDpsInits().front().getType());
  if (!out_type)
    return generic_op.emitError(PASS_NAME)
           << ": a register's lanes are reduced, but the output operand is not "
              "a ranked tensor";

  const unsigned num_loops = generic_op.getNumLoops();
  SmallVector<int64_t> loop_ranges = generic_op.getStaticLoopRanges();
  if (ShapedType::isDynamic(loop_ranges[lane_dim]))
    return generic_op.emitError(PASS_NAME)
           << ": a register's lanes are reduced but the extent is dynamic; a "
              "register-width extent must be known here";
  const int64_t lane_extent = loop_ranges[lane_dim];

  auto iter_types = generic_op.getIteratorTypesArray();
  SmallVector<AffineMap> indexing_maps = generic_op.getIndexingMapsArray();
  AffineMap out_map = indexing_maps.back();

  // An output lane axis must also be the innermost loop dim, since that is the
  // axis the store walks contiguously; anywhere else it is not the register
  // this reduction lives in.
  std::optional<unsigned> out_lane_dim;
  if (out_map.getNumResults() > 0) {
    if (auto out_fastest = dyn_cast<AffineDimExpr>(
            out_map.getResult(out_map.getNumResults() - 1))) {
      unsigned dim = out_fastest.getPosition();
      if (dim != lane_dim && dim == num_loops - 1 &&
          iter_types[dim] == utils::IteratorType::parallel &&
          loop_ranges[dim] == lane_extent)
        out_lane_dim = dim;
    }
  }

  MLIRContext* ctx = generic_op.getContext();
  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  RankedTensorType carrier_type;  // what the intrinsic consumes
  SmallVector<utils::IteratorType> new_iter_types;

  if (out_lane_dim) {
    // Merge the two names for one register. Every reference to the output's
    // lane dim becomes a reference to the input's, and the iteration space
    // loses the dim that is now unused.
    SmallVector<AffineExpr> replacement;
    for (unsigned i = 0; i < num_loops; ++i)
      replacement.push_back(
          getAffineDimExpr(i == *out_lane_dim ? lane_dim : i, ctx));
    for (AffineMap& map : indexing_maps)
      map = map.replaceDimsAndSymbols(replacement, /*symReplacements=*/{},
                                      num_loops - 1, /*numResultSyms=*/0);

    for (unsigned i = 0; i < num_loops; ++i) {
      if (i == *out_lane_dim) continue;
      new_iter_types.push_back(i == lane_dim ? utils::IteratorType::parallel
                                             : iter_types[i]);
    }
    carrier_type = out_type;
  } else {
    // Give the output a register-wide axis of its own, last, matching the loop
    // order the reduced dim had.
    SmallVector<AffineExpr> out_results(out_map.getResults());
    out_results.push_back(getAffineDimExpr(lane_dim, ctx));
    indexing_maps.back() =
        AffineMap::get(num_loops, /*symbolCount=*/0, out_results, ctx);

    SmallVector<int64_t> widened_shape(out_type.getShape());
    widened_shape.push_back(lane_extent);
    carrier_type =
        RankedTensorType::get(widened_shape, out_type.getElementType());

    new_iter_types.assign(iter_types.begin(), iter_types.end());
    new_iter_types[lane_dim] = utils::IteratorType::parallel;
  }

  // The original init is a bare uninitialised destination, so it is replaced
  // wholesale rather than derived from.
  auto carrier_init = tensor::EmptyOp::create(
      builder, loc, carrier_type.getShape(), carrier_type.getElementType());

  auto new_generic = linalg::GenericOp::create(
      builder, loc, TypeRange{carrier_type}, generic_op.getInputs(),
      ValueRange{carrier_init.getResult()}, indexing_maps, new_iter_types);
  // The body is indifferent to which dimensions are reduced, so clone it as is.
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&new_generic.getRegion(), mapping);

  // The intrinsic yields the original output shape, so downstream sees the
  // element count it already expected. It is destination-passing: the
  // destination has its *result* type, not its input type.
  auto opaque_init = tensor::EmptyOp::create(builder, loc, out_type.getShape(),
                                             out_type.getElementType());
  auto opaque = ktdf::OpaqueOp::create(builder, loc, TypeRange{out_type},
                                       ktdf::kLaneReduceTemplateName,
                                       ValueRange{new_generic.getResult(0)},
                                       ValueRange{opaque_init.getResult()});

  generic_op->getResult(0).replaceAllUsesWith(opaque->getResult(0));

  Value old_init = generic_op.getDpsInits().front();
  generic_op->erase();
  if (Operation* def = old_init.getDefiningOp())
    if (def->use_empty()) def->erase();

  return success();
}

struct ReductionLaneCollapsePass
    : public ktdf::impl::ReductionLaneCollapsePassBase<
          ReductionLaneCollapsePass> {
  using ReductionLaneCollapsePassBase<
      ReductionLaneCollapsePass>::ReductionLaneCollapsePassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    // Collect before rewriting: the rewrite replaces the op being visited.
    SmallVector<std::pair<linalg::GenericOp, unsigned>> candidates;
    getOperation().walk([&](linalg::GenericOp generic) {
      if (std::optional<unsigned> lane_dim =
              findInRegisterReductionDim(generic))
        candidates.emplace_back(generic, *lane_dim);
    });

    for (auto [generic, lane_dim] : candidates)
      if (failed(collapseInRegisterReduction(generic, lane_dim))) {
        signalPassFailure();
        return;
      }
  }
};

}  // namespace

auto mlir::ktdf::createReductionLaneCollapsePass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionLaneCollapsePass>();
}
