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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNIFORMINFRA_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNIFORMINFRA_H_

#include <map>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Storage for queried unit SSA values
struct QueriedUnitsMap {
  llvm::DenseMap<ResourceType, mlir::Value>
      non_parallel;  // component resource -> Value
  llvm::DenseMap<std::pair<std::pair<mlir::Operation*, ResourceType>, int>,
                 mlir::Value>
      parallel;  // ((parallel_op, component resource), corelet) -> Value
};

/// Storage for uniform maps
struct UniformMapsStorage {
  llvm::DenseMap<ResourceType, mlir::Value>
      non_parallel;  // component resource -> Value
  llvm::DenseMap<std::pair<std::pair<mlir::Operation*, ResourceType>, int>,
                 mlir::Value>
      parallel;  // ((parallel_op, component resource), corelet) -> Value
};

/// Creates uniform maps and queries at function entry
class UniformInfra {
 public:
  explicit UniformInfra(mlir::func::FuncOp func) : func_(func) {}

  /// Create all maps and queries: for each component, a map from each tile of
  /// @p component_tiles to the component's unit on it
  mlir::LogicalResult createMapsAndQueries(
      const ComponentClassification& components,
      const ComponentTiles& component_tiles, const UnitSSAMap& unit_ssa_map,
      QueriedUnitsMap& queried_units, UniformMapsStorage& uniform_maps,
      mlir::OpBuilder& builder);

  /// Emit uniform map + query inside a program_unit body for each per-core
  /// memory space. Keys = program_unit operand values (matched to memory unit
  /// values by the `core` attr on their defining GetUnitOp). Query key = the
  /// program_unit's block argument %arg0 (iterator). Returns a map from
  /// memory space attribute to the queried Value for use in that body.
  mlir::LogicalResult buildMemoryUniformMaps(
      mlir::dataflow::ProgramUnitOp pu,
      const llvm::SetVector<ResourceType>& per_core_spaces,
      const MemoryUnitSSAMap& memory_unit_ssa,
      const scheduler::SchedulerExtContext& ext_ctx,
      llvm::DenseMap<ResourceType, mlir::Value>& resolved_units,
      mlir::OpBuilder& builder);

  /// Build a query map for a signal operand within a program_unit context.
  /// Takes a query_map (with get_compute_tile_id as key) and constructs a new
  /// query_map where:
  /// - Keys are the program_unit operands (current unit instances)
  /// - Values are the corresponding units from the original mapping
  /// - Query key is the program_unit's iter_arg (%arg0)
  /// This allows signaling between specific unit instances across cores.
  static llvm::FailureOr<mlir::Value> buildSignalQueryMap(
      mlir::Value signal_query_map, mlir::dataflow::ProgramUnitOp program_unit,
      mlir::OpBuilder& builder, mlir::Location loc);

 private:
  mlir::func::FuncOp func_;

  std::string getComponentName(ResourceType component);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNIFORMINFRA_H_
