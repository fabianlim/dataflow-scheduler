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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNITMATERIALIZER_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNITMATERIALIZER_H_

#include <map>
#include <string>
#include <tuple>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Lowercase unit-type tag for a resource attribute, as the emitted DFIR
/// `name`/`type` strings spell it (code generation requires lowercase).
///
/// This is the single derivation of a backend-facing name from an architecture
/// resource: compute resource tokens are lowercased directly, Spyre
/// memory-space attributes become their lowercased kind ("l1"/"ddr"), and any
/// other attribute falls back to its lowercased printed form so nothing is left
/// uppercase.  Every consumer that has to name a unit — get_unit for an
/// addressable space, get_local_unit for a compute-unit-local one — must go
/// through here, so that a memory kind declared in the device description and
/// the name the backend receives never drift apart.
std::string unitTypeTag(ResourceType rt);

/// Storage for unit SSA values: (component, core) -> Value for non-parallel,
/// (parallel_op, component, corelet, core) -> Value for parallel
struct UnitSSAMap {
  llvm::DenseMap<std::pair<ResourceType, int>, mlir::Value>
      non_parallel;  // (component resource, core) -> Value
  llvm::DenseMap<std::tuple<mlir::Operation*, ResourceType, int, int>,
                 mlir::Value>
      parallel;  // (parallel_op, component resource, corelet, core) -> Value
};

/// Storage for memory-space unit SSA values.
/// Global spaces (DDR): keyed by (memory_space_attr, -1).
/// Per-core spaces (L1): keyed by (memory_space_attr, core).
using MemoryUnitSSAMap =
    llvm::DenseMap<std::pair<ResourceType, int>, mlir::Value>;

/// Materializes dataflow.get_unit operations for all components and cores
class UnitMaterializer {
 public:
  explicit UnitMaterializer(mlir::func::FuncOp func) : func_(func) {}

  /// Create all unit SSA values at function entry
  mlir::LogicalResult materialize(const ComponentClassification& components,
                                  int grid_size, UnitSSAMap& unit_ssa_map,
                                  mlir::OpBuilder& builder);

  /// Emit dataflow.get_unit ops for memory-space components at func entry.
  /// The instance multiplicity of a space follows its role in the device:
  /// - Global spaces (memory_tree.isGlobalMemory, depth 0): one unit, key
  ///   core = -1.
  /// - Per-core scratchpad spaces (memory_tree.isPerCoreScratchPadMemory,
  ///   depth 1): one unit per core 0..grid_size-1.
  /// - Compute-unit-local spaces (group_local_mem.isComputeUnitLocal): no unit
  ///   is materialized, and none is needed — such memory is accessed only from
  ///   the compute unit that owns it, which resolves its own handle via
  ///   dataflow.get_local_unit. These spaces are normally filtered out before
  ///   they reach here (discoverAndPrune); the branch is a guard against a
  ///   caller that collects them anyway.
  /// Any other space is an error.
  mlir::LogicalResult materializeMemoryUnits(
      const llvm::SetVector<ResourceType>& needed_spaces, int grid_size,
      const scheduler::arch_view::MemoryTree& memory_tree,
      const scheduler::arch_view::GroupLocalMemory& group_local_mem,
      MemoryUnitSSAMap& memory_unit_ssa, mlir::OpBuilder& builder);

 private:
  mlir::func::FuncOp func_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNITMATERIALIZER_H_
