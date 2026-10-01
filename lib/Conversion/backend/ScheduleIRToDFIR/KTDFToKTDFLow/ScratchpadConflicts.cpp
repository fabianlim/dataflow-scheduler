//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ScratchpadConflicts.h"

#include <optional>

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "phase2-analysis"

using namespace scheduler;

namespace {

/// Gets the memories that the ops of @p stage read and write: the memory
/// spaces of the memrefs they access, from their memory effects. An op with
/// unknown effects may read and write each of its memref operands. Fifos are
/// not memory.
void getStageAccesses(mlir::Operation* stage,
                      llvm::SmallDenseSet<mlir::Attribute>& read_memories,
                      llvm::SmallDenseSet<mlir::Attribute>& write_memories) {
  const auto getMemory =
      [](mlir::Value value) -> std::optional<mlir::Attribute> {
    auto type =
        value ? llvm::dyn_cast<mlir::MemRefType>(value.getType()) : nullptr;
    if (!type) {
      return std::nullopt;
    }
    return type.getMemorySpace();
  };
  stage->walk([&](mlir::Operation* op) {
    auto effects_op = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    if (!effects_op) {
      for (mlir::Value operand : op->getOperands()) {
        if (auto memory = getMemory(operand)) {
          read_memories.insert(*memory);
          write_memories.insert(*memory);
        }
      }
      return;
    }
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    effects_op.getEffects(effects);
    for (const auto& effect : effects) {
      auto memory = getMemory(effect.getValue());
      if (!memory) {
        continue;
      }
      if (llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
        read_memories.insert(*memory);
      }
      if (llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
        write_memories.insert(*memory);
      }
    }
  });
}

}  // namespace

mlir::LogicalResult scheduler::computeScratchpadConflicts(
    const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& dag,
    std::map<std::pair<mlir::Operation*, mlir::Operation*>,
             llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts) {
  LDBG(1) << "Step 3: Compute scratchpad conflicts";

  for (const auto& [producer_op, successors] : dag.successors) {
    assert(llvm::isa<mlir::ktdf::StageOp>(producer_op) &&
           "producer op has to be a stage operation");

    auto producer_it = stage_to_units.mapping.find(producer_op);
    assert(producer_it != stage_to_units.mapping.end() &&
           "producer op should be found in stage_to_units mapping");

    const auto& producer_units = producer_it->second;

    llvm::SmallDenseSet<mlir::Attribute> producer_reads_unused;
    llvm::SmallDenseSet<mlir::Attribute> producer_writes;
    getStageAccesses(producer_op, producer_reads_unused, producer_writes);

    for (auto consumer_op : successors) {
      assert(llvm::isa<mlir::ktdf::StageOp>(consumer_op) &&
             "consumer op has to be a stage operation");

      auto consumer_it = stage_to_units.mapping.find(consumer_op);
      assert(consumer_it != stage_to_units.mapping.end() &&
             "consumer op should be found in stage_to_units mapping");

      const auto& consumer_units = consumer_it->second;

      llvm::SmallDenseSet<mlir::Attribute> consumer_reads;
      llvm::SmallDenseSet<mlir::Attribute> consumer_writes_unused;
      getStageAccesses(consumer_op, consumer_reads, consumer_writes_unused);

      // The stages conflict if the producer writes a memory the consumer
      // reads.
      const auto written = llvm::find_if(producer_writes, [&](auto memory) {
        return consumer_reads.contains(memory);
      });
      if (written == producer_writes.end()) {
        continue;
      }

      llvm::SmallVector<scheduler::ResourceType, 2> conflicting_units;

      for (auto producer_unit_val : producer_units) {
        auto producer_comp_opt =
            scheduler::getUnitResourceType(producer_unit_val);
        if (!producer_comp_opt.has_value()) continue;
        scheduler::ResourceType producer_comp = producer_comp_opt.value();

        for (auto consumer_unit_val : consumer_units) {
          auto consumer_comp_opt =
              scheduler::getUnitResourceType(consumer_unit_val);
          assert(consumer_comp_opt.has_value() &&
                 "Consumer unit should have a component");

          scheduler::ResourceType consumer_comp = consumer_comp_opt.value();
          LDBG(1) << "  Found conflict between " << producer_comp << " and "
                  << consumer_comp << " in " << *written;
          conflicting_units.push_back(producer_comp);
          conflicting_units.push_back(consumer_comp);
        }
      }

      if (!conflicting_units.empty()) {
        conflicts[{producer_op, consumer_op}] = conflicting_units;
        LDBG(1) << "  Conflict between stages: " << producer_op << " -> "
                << consumer_op;
      }
    }
  }

  return mlir::success();
}
