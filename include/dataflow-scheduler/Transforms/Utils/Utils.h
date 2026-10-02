//===-------------------------------------------------------------*- c++ -*-==//
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
// Utility functions for transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_UTILS_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_UTILS_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace scheduler {

/// Returns the static trip count of `loop` iff lb, ub, step are all
/// arith.constant index ops. step must be positive.
std::optional<int64_t> getStaticTripCount(mlir::scf::ForOp loop);

/// Extracts and validate grid size from function attributes
/// Grid is a 1D array attribute containing a single integer
mlir::LogicalResult extractGridSize(mlir::func::FuncOp func, int& grid_size);

/// Clones an scf.for operation and add additional iter args (and corresponding
/// return values). The initial values of the new loop-carried arguments are
/// dummy constants (index 0). The return values are simply the loop-carried
/// arguments passed through. The reason for cloning is that MLIR infrastructure
/// doesn't allow adding new results to an operation without cloning it.
///
/// If \p delete_op is set, \p loop_op will be deleted after the new loop is
/// created.
///
/// After cloning the loop body of \p loop_op, \p ir_map will contain the IR
/// mapping of the loop body. If \p delete_op was false, this IRMapping can be
/// used outside of this function.
///
/// \param loop_op The loop to update.
/// \param n_values How many additional return values/iter_args should the new
/// loop contain.
/// \param ir_map The IRMapping object to store the cloned loop body in. Only
/// valid for use if \p delete_op was false.
/// \param delete_op If true, deletes \p loop_op after creation of the new loop.
/// \return A new loop created from \p loop_op but with \p n_values number of
/// additional return values.
mlir::scf::ForOp createForOpWithAdditionalIterArgs(mlir::scf::ForOp loop_op,
                                                   int n_values,
                                                   mlir::IRMapping& ir_map,
                                                   bool delete_op = true);

/// Erases the ops of \p roots that nothing reads, and whatever they were the
/// last reader of.
void eraseDeadAncestorOps(llvm::ArrayRef<mlir::Operation*> roots);

/// One dimension of the walk of one side of a transfer: which memref dimension
/// it advances, and by how many indices of that dimension per step.
struct TransferTimeStep {
  unsigned memref_dim;
  int64_t index_step;
};

/// How one side of a transfer is walked in vectors: the extent of each walked
/// dimension, slowest-varying first, and what each dimension advances. In
/// DFIR the walked dimensions are AGEN time dimensions: `extents` becomes
/// `time_set` and `offsets()` the results of that side's `*_time_addr_map`.
struct TransferTimeDims {
  llvm::SmallVector<int64_t> extents;
  llvm::SmallVector<TransferTimeStep> steps;  // parallel to `extents`
  size_t rank = 0;

  /// A traversal of a memref of `rank` dimensions that walks nothing.
  explicit TransferTimeDims(size_t rank) : rank(rank) {}

  /// The offset added to each memref index at time step (d0, ..., dn-1); zero
  /// at every index a time dimension does not advance. Time dimensions are
  /// numbered in the order they were added, which is also why an identity
  /// `time_order` is correct: d0 is the slowest-varying.
  llvm::SmallVector<mlir::AffineExpr> offsets(
      mlir::MLIRContext* context) const {
    llvm::SmallVector<mlir::AffineExpr> result(
        rank, mlir::getAffineConstantExpr(0, context));
    for (auto [time_dim, step] : llvm::enumerate(steps)) {
      result[step.memref_dim] =
          mlir::getAffineConstantExpr(step.index_step, context) *
          mlir::getAffineDimExpr(time_dim, context);
    }
    return result;
  }

  /// Drop time dimension `time_dim`. The remaining dimensions keep their
  /// relative order and are renumbered by `offsets()`.
  void eraseDim(unsigned time_dim) {
    extents.erase(extents.begin() + time_dim);
    steps.erase(steps.begin() + time_dim);
  }
};

/// Describe how `sizes` is traversed in vectors of `lanes` elements. Every
/// non-unit dimension except the innermost contributes a time dimension
/// stepping by one; the innermost contributes one stepping by a whole vector,
/// and only when it holds more than one. With a 64-lane vector:
///
///   sizes           extents      offsets           time_set
///   [1, 256, 64]    [256]        (0, d0, 0)        (d0) : 0 <= d0 <= 255
///   [1, 1, 128]     [2]          (0, 0, 64 * d0)   (d0) : 0 <= d0 <= 1
///   [2, 4, 8, 64]   [2, 4, 8]    (d0, d1, d2, 0)   3 dims of those extents
///   [1, 64]         []           (0, 0)            nothing walked
///
/// The last row is a transfer that fits in one vector: the offsets are already
/// the all-zero map, and the caller supplies the single pinned time step.
TransferTimeDims describeTransferTimeDims(llvm::ArrayRef<int64_t> sizes,
                                          int64_t lanes);

/// Extend `map` with one trailing dimension that advances `step.memref_dim` by
/// `step.index_step` indices per unit. Appending a loop induction variable to
/// the subscript operands then drives that dimension from the loop instead of
/// from the time axis.
mlir::AffineMap foldStepIntoSubscripts(mlir::MLIRContext* context,
                                       mlir::AffineMap map,
                                       TransferTimeStep step);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_UTILS_H_

// Made with Bob
