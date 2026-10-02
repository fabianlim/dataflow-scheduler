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
//
/// Build dataflow.program_unit ops by cloning the work region into each, then
/// filter inner ktdf_lowering ops by component-type applicability.
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PROGRAMUNITBUILDER_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PROGRAMUNITBUILDER_H_

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/PreludeWorkPartition.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// For each entry in `components`, in iteration order, create a
/// dataflow.program_unit at the function entry block (right before the
/// terminator) and clone every op in `work_ops` into its body. Then filter
/// inner ktdf_lowering.execute_on / ktdf_lowering.signal ops by applicability
/// to the entry's resource type. A program unit whose body creates a
/// multicast group gets one uniform.uniformize_regions region per unit, each
/// holding a copy of the body, since the group's attributes differ per unit.
///
/// Erases the original `work_ops` from the function entry block on success.
///
/// Returns failure if any inner unit operand cannot be resolved to a
/// resource type.
mlir::LogicalResult buildProgramUnits(mlir::func::FuncOp func,
                                      llvm::ArrayRef<mlir::Operation*> work_ops,
                                      const ResourceToUnits& components);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PROGRAMUNITBUILDER_H_
