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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/SignalInsertion.h"

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "phase2-signal-insertion"

using namespace scheduler;

mlir::LogicalResult scheduler::insertSignalsInPipeline(
    mlir::ktdf::PipelineOp pipeline,
    const llvm::SmallVector<mlir::ktdf::StageOp, 8>& sorted_stages,
    const StageToUnitsMap& stage_to_units,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    mlir::OpBuilder& builder) {
  LLVM_DEBUG(llvm::dbgs() << "Step 9: Insert signal operations\n");

  auto loc = pipeline.getLoc();

  for (size_t i = 0; i + 1 < sorted_stages.size(); ++i) {
    auto producer_stage = sorted_stages[i];
    auto consumer_stage = sorted_stages[i + 1];

    bool has_conflict =
        conflicts.count(std::make_pair(producer_stage.getOperation(),
                                       consumer_stage.getOperation())) > 0;
    if (!has_conflict) continue;

    LLVM_DEBUG(llvm::dbgs() << "  Inserting signal between stages\n");
    builder.setInsertionPointAfter(producer_stage);

    // Narrow to immediate leaf/root stages across any nested pipeline boundary.
    auto signal_producers = mlir::ktdf::getLeafStages(producer_stage);
    auto signal_consumers = mlir::ktdf::getRootStages(consumer_stage);

    llvm::SmallVector<mlir::Value, 8> signal_units;
    auto collect_units = [&](mlir::ktdf::StageOp stage) {
      auto it = stage_to_units.mapping.find(stage.getOperation());
      if (it != stage_to_units.mapping.end()) {
        for (auto unit : it->second) {
          signal_units.push_back(unit);
        }
      }
    };

    for (auto s : signal_producers) collect_units(s);
    for (auto s : signal_consumers) collect_units(s);

    // The token-dependency DAG used by getLeafStages/getRootStages can be
    // oriented opposite to physical dataflow inside a coarsened nested
    // pipeline. When that happens the "leaf" of the producer resolves to a
    // compute sub-stage instead of the DMA sub-stage that actually writes the
    // shared scratchpad buffer, producing a mis-paired signal that the backend
    // rejects. Detect this by checking the leaf/root selection against the
    // conflicting resource types the conflict analysis already computed, and
    // fall back to a buffer-granular selection when it disagrees.
    const auto& payload = conflicts.at(std::make_pair(
        producer_stage.getOperation(), consumer_stage.getOperation()));
    llvm::SmallDenseSet<mlir::Attribute> conflicting_producer_rts;
    llvm::SmallDenseSet<mlir::Attribute> conflicting_consumer_rts;
    for (size_t k = 0; k + 1 < payload.size(); k += 2) {
      conflicting_producer_rts.insert(payload[k]);
      conflicting_consumer_rts.insert(payload[k + 1]);
    }

    auto unitRtInSet = [&](mlir::Value unit,
                           const llvm::SmallDenseSet<mlir::Attribute>& s) {
      auto rt = scheduler::getUnitResourceType(unit);
      return rt.has_value() && s.contains(rt.value());
    };

    bool selection_ok = !signal_units.empty();
    for (auto unit : signal_units) {
      if (!unitRtInSet(unit, conflicting_producer_rts) &&
          !unitRtInSet(unit, conflicting_consumer_rts)) {
        selection_ok = false;
        break;
      }
    }

    if (!selection_ok) {
      // Buffer-granular reselection. The consumer reads a set of scratchpad
      // memrefs (its data_transfer sources); the correct producer unit is the
      // one whose data_transfer destination writes one of those exact memrefs.
      // This distinguishes the drain writer (e.g. LXSU) from an unrelated
      // load-staging writer (e.g. L3LU) that shares the same memory-space kind.
      llvm::SmallDenseSet<mlir::Value> consumer_read_memrefs;
      consumer_stage.walk([&](mlir::ktdf::DataTransferOp dt) {
        if (dt.isSourceMemRef()) consumer_read_memrefs.insert(dt.getSource());
      });

      llvm::SmallVector<mlir::Value, 8> reselected;
      llvm::SmallDenseSet<mlir::Operation*> seen_producer_stages;
      producer_stage.walk([&](mlir::ktdf::DataTransferOp dt) {
        if (!dt.isDestMemRef()) return;
        if (!consumer_read_memrefs.contains(dt.getDestination())) return;
        auto owning_stage = dt->getParentOfType<mlir::ktdf::StageOp>();
        if (!owning_stage || !seen_producer_stages.insert(owning_stage).second)
          return;
        auto it = stage_to_units.mapping.find(owning_stage.getOperation());
        if (it == stage_to_units.mapping.end()) return;
        for (auto unit : it->second) {
          if (unitRtInSet(unit, conflicting_producer_rts))
            reselected.push_back(unit);
        }
      });

      // Consumer side: keep only units whose resource type actually conflicts.
      for (auto s : signal_consumers) {
        auto it = stage_to_units.mapping.find(s.getOperation());
        if (it == stage_to_units.mapping.end()) continue;
        for (auto unit : it->second) {
          if (unitRtInSet(unit, conflicting_consumer_rts))
            reselected.push_back(unit);
        }
      }

      if (!reselected.empty()) signal_units = std::move(reselected);
    }

    if (!signal_units.empty()) {
      mlir::ktdf_lowering::SignalOp::create(builder, loc,
                                            mlir::ValueRange(signal_units));
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Signal insertion complete\n");
  return mlir::success();
}
