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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SCRATCHPADCONFLICTS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SCRATCHPADCONFLICTS_H_

#include <llvm/ADT/SmallVector.h>

#include <map>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/StageToUnitsMap.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"

namespace scheduler {

/// Step 3: Compute scratchpad conflicts between ordered stage pairs. A
/// producer stage and a consumer stage conflict if the producer writes a
/// memory that the consumer reads, as given by the memory spaces of the
/// memrefs their ops access. Fifos are not memory.
mlir::LogicalResult computeScratchpadConflicts(
    const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& dag,
    std::map<std::pair<mlir::Operation*, mlir::Operation*>,
             llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SCRATCHPADCONFLICTS_H_
