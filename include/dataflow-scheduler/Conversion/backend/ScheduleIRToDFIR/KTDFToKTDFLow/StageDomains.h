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
// Stage domains in the backend.
//
// Units are materialized per component, and each component's units run one
// program unit, so the units of a component are on the tiles of the domain of
// the stages that use it. A stage without a domain uses every tile of the
// grid.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_STAGEDOMAINS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_STAGEDOMAINS_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace scheduler {

/// Component -> the tiles its units are on, in increasing order.
using ComponentTiles = llvm::DenseMap<ResourceType, llvm::SmallVector<int64_t>>;

/// Reads and verifies the domain of each of @p stages, and gets the tiles of
/// each component they use.
///
/// A plain domain must be a set over the tile alone, non-empty and within the
/// grid [0, @p grid_size). A domain with one symbol, the group, is only valid
/// on a stage that writes or reads exactly one cross-core fifo, and the
/// stages that write and read a cross-core fifo must have one: the groups of
/// the fifo are resolved from its group domain and the two stage domains, and
/// the stages execute on the producers and on the consumers of all groups.
/// All stages of a component must execute on the same tiles.
mlir::LogicalResult resolveComponentTiles(
    llvm::ArrayRef<mlir::ktdf::StageOp> stages, int grid_size,
    ComponentTiles& component_tiles);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_STAGEDOMAINS_H_
