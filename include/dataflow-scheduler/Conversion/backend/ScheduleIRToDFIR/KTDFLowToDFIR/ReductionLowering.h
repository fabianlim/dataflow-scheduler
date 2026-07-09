//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_REDUCTIONLOWERING_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_REDUCTIONLOWERING_H_

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Peel the first iteration of each compute reduction loop (identified by the
/// ktdf.reduction_accumulator attribute + a vectorchain.binary body) out of
/// the loop and into an unconditional init store before the loop.
///
/// The unconditional init-store is required because a conditional store inside
/// scf.if is not honored by hardware. Only the compute loop (body has
/// vectorchain.binary) is peeled; transfer loops that share the
/// ktdf.reduction_accumulator attribute are left unchanged.
void peelReductionComputeLoop(mlir::func::FuncOp func);

/// Emit the lrfreg write-back for a reduction linalg op that has been lowered:
/// store `result` to `acc_view`, erase the materialize_in_destination bridge
/// op for the generic result, erase `generic_op`, and (if dead) `acc_to_tensor`.
void emitReductionWriteBack(mlir::PatternRewriter& rewriter,
                            mlir::linalg::GenericOp generic_op,
                            mlir::bufferization::ToTensorOp acc_to_tensor,
                            mlir::Value result,
                            mlir::Value acc_view);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_REDUCTIONLOWERING_H_
