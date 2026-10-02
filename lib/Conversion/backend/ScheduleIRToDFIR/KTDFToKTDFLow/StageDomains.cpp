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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/StageDomains.h"

#include <string>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/InterleavedRange.h>
#include <mlir/IR/Diagnostics.h>

#include "dataflow-scheduler/Analysis/CrossCoreChannels.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "ktdf-to-operand-lowering"

using namespace scheduler;

namespace {

/// The stages that write and the stages that read each fifo slot. A stage is
/// the innermost one around the fifo access.
struct FifoAccesses {
  llvm::MapVector<mlir::Value, llvm::SmallVector<mlir::ktdf::StageOp, 1>>
      writers;
  llvm::MapVector<mlir::Value, llvm::SmallVector<mlir::ktdf::StageOp, 1>>
      readers;
};

FifoAccesses collectFifoAccesses(llvm::ArrayRef<mlir::ktdf::StageOp> stages) {
  FifoAccesses accesses;
  for (mlir::ktdf::StageOp stage : stages) {
    stage.walk([&](mlir::Operation* op) {
      if (op->getParentOfType<mlir::ktdf::StageOp>() != stage) {
        return;
      }
      if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
        if (transfer.isDestFifo()) {
          accesses.writers[transfer.getDestination()].push_back(stage);
        }
        if (transfer.isSourceFifo()) {
          accesses.readers[transfer.getSource()].push_back(stage);
        }
      } else if (auto write = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op)) {
        accesses.writers[write.getFifoSlot()].push_back(stage);
      } else if (auto read = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op)) {
        accesses.readers[read.getFifoSlot()].push_back(stage);
      }
    });
  }
  return accesses;
}

/// Gets the cross-core fifos among those in @p fifos, in order.
llvm::SetVector<mlir::Value> getCrossCoreFifos(
    const llvm::MapVector<mlir::Value,
                          llvm::SmallVector<mlir::ktdf::StageOp, 1>>& fifos) {
  llvm::SetVector<mlir::Value> cross_core;
  for (const auto& [fifo, unused_stages] : fifos) {
    if (getFifoGroups(fifo)) {
      cross_core.insert(fifo);
    }
  }
  return cross_core;
}

/// Gets the domain of @p stage, verified to be an integer set over the tile
/// with at most one symbol, the group, or nullptr if it has none.
mlir::FailureOr<mlir::IntegerSetAttr> getStageDomain(
    mlir::ktdf::StageOp stage) {
  const mlir::Attribute attr = stage->getAttr(kStageDomainAttrName);
  if (!attr) {
    return mlir::IntegerSetAttr();
  }

  auto domain_attr = mlir::dyn_cast<mlir::IntegerSetAttr>(attr);
  if (!domain_attr) {
    stage.emitError() << "'" << kStageDomainAttrName
                      << "' must be an integer set";
    return mlir::failure();
  }
  const mlir::IntegerSet domain = domain_attr.getValue();
  if (domain.getNumDims() != 1 || domain.getNumSymbols() > 1) {
    stage.emitError() << "domain must be a set over the compute tile, with "
                         "one dimension and at most one symbol, the group, "
                         "but it has "
                      << domain.getNumDims() << " dimensions and "
                      << domain.getNumSymbols() << " symbols";
    return mlir::failure();
  }
  return domain_attr;
}

/// Gets the tiles of the plain domain @p domain_attr of @p stage, verified to
/// be non-empty and within the grid.
mlir::FailureOr<llvm::SmallVector<int64_t>> getPlainDomainTiles(
    mlir::ktdf::StageOp stage, mlir::IntegerSetAttr domain_attr,
    int grid_size) {
  const mlir::IntegerSet domain = domain_attr.getValue();
  if (!isDomainWithinGrid(domain, grid_size)) {
    stage.emitError() << "domain " << domain_attr
                      << " has tiles outside the grid [0, " << grid_size
                      << ")";
    return mlir::failure();
  }

  auto tiles = getDomainTiles(domain, grid_size);
  if (tiles.empty()) {
    stage.emitError() << "domain " << domain_attr << " is empty";
    return mlir::failure();
  }
  return tiles;
}

/// Prints the groups of the cross-core fifo @p fifo for debugging, with what
/// is derived from them.
void debugPrintGroups(mlir::Value fifo,
                      llvm::ArrayRef<ChannelGroup> channel_groups) {
  LDBG(1) << "Groups of the cross-core fifo " << getFifoGroups(fifo);
  for (const ChannelGroup& group : channel_groups) {
    LDBG(1) << "  group " << group.id << ": producers {"
            << llvm::interleaved(group.producers) << "}, consumers {"
            << llvm::interleaved(group.consumers) << "}";
  }
  LDBG(1) << "  producer tiles {"
          << llvm::interleaved(getProducerTiles(channel_groups))
          << "}, consumer tiles {"
          << llvm::interleaved(getConsumerTiles(channel_groups))
          << "}, ring consumer tiles {"
          << llvm::interleaved(getRingConsumerTiles(channel_groups)) << "}";
  LDBG(1) << "  producer of each consumer: "
          << llvm::interleaved(llvm::map_range(
                 getConsumerTiles(channel_groups), [&](int64_t consumer) {
                   return std::to_string(consumer) + " <- " +
                          std::to_string(
                              *getProducerOf(channel_groups, consumer));
                 }));
  LDBG(1) << "  self-deliveries {"
          << llvm::interleaved(getSelfDeliveries(channel_groups)) << "}";
}

/// Resolves the channel of each cross-core fifo in @p accesses: the groups
/// from its group domain and the domains of the stages that write and read
/// it. Sets the tiles of those stages in @p stage_tiles: the producers of all
/// groups for the writing stage, and the ring consumers of all groups for the
/// reading stage. The ring cannot deliver a tile's data to the tile itself;
/// the self-deliveries are carried by the local copies of
/// HandleCrossCoreStages, which runs before.
mlir::LogicalResult resolveCrossCoreFifos(
    const FifoAccesses& accesses,
    const llvm::DenseMap<mlir::Operation*, mlir::IntegerSetAttr>& domains,
    int grid_size,
    llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>>& stage_tiles,
    CrossCoreChannelMap& channels) {
  llvm::SetVector<mlir::Value> fifos = getCrossCoreFifos(accesses.writers);
  fifos.insert_range(getCrossCoreFifos(accesses.readers));

  // The domain of a stage that writes or reads a cross-core fifo selects its
  // tiles by group.
  const auto getParametricDomain =
      [&](mlir::ktdf::StageOp stage,
          llvm::StringRef access) -> mlir::FailureOr<mlir::IntegerSetAttr> {
    const mlir::IntegerSetAttr domain = domains.lookup(stage.getOperation());
    if (!domain || domain.getValue().getNumSymbols() != 1) {
      stage.emitError() << "stage " << access
                        << " a cross-core fifo, so its domain must have one "
                           "symbol, the group, that selects its tiles in "
                           "each group";
      return mlir::failure();
    }
    return domain;
  };

  for (mlir::Value fifo : fifos) {
    auto writers = accesses.writers.lookup(fifo);
    auto readers = accesses.readers.lookup(fifo);
    if (writers.empty()) {
      return readers.front().emitError()
             << "stage reads a cross-core fifo that no stage writes";
    }
    if (readers.empty()) {
      return writers.front().emitError()
             << "stage writes a cross-core fifo that no stage reads";
    }
    // The groups belong to the channel, one producer stage and one consumer
    // stage, and both create the same multicast groups from them.
    if (writers.size() != 1 || readers.size() != 1) {
      return writers.front().emitError()
             << "a cross-core fifo is written by one stage and read by one "
                "stage for now, but the fifo this stage writes is written by "
             << writers.size() << " and read by " << readers.size();
    }
    mlir::ktdf::StageOp writer = writers.front();
    mlir::ktdf::StageOp reader = readers.front();

    auto producer_domain = getParametricDomain(writer, "writes");
    if (mlir::failed(producer_domain)) {
      return mlir::failure();
    }
    auto consumer_domain = getParametricDomain(reader, "reads");
    if (mlir::failed(consumer_domain)) {
      return mlir::failure();
    }

    auto channel_groups = resolveChannelGroups(
        getFifoGroups(fifo), *producer_domain, *consumer_domain, grid_size,
        [&] {
          auto diag = writer.emitError()
                      << "invalid groups of the cross-core fifo this "
                         "stage writes: ";
          diag.attachNote(reader.getLoc()) << "the stage reading the fifo";
          return diag;
        });
    if (mlir::failed(channel_groups)) {
      return mlir::failure();
    }
    LLVM_DEBUG(debugPrintGroups(fifo, *channel_groups));

    stage_tiles[writer.getOperation()] = getProducerTiles(*channel_groups);
    stage_tiles[reader.getOperation()] = getRingConsumerTiles(*channel_groups);
    channels[fifo] = {writer, reader, std::move(*channel_groups)};
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::resolveComponentTiles(
    llvm::ArrayRef<mlir::ktdf::StageOp> stages, int grid_size,
    ComponentTiles& component_tiles, CrossCoreChannelMap& channels) {
  LDBG(1) << "Step 2b: Resolve stage domains";

  const FifoAccesses accesses = collectFifoAccesses(stages);

  // A stage without a domain executes on the whole grid and a stage with a
  // plain domain on its tiles. A domain with the group symbol selects the
  // tiles of the groups of the one cross-core fifo the stage writes or reads,
  // which are resolved below.
  llvm::DenseMap<mlir::Operation*, mlir::IntegerSetAttr> domains;
  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>> stage_tiles;
  for (mlir::ktdf::StageOp stage : stages) {
    auto domain = getStageDomain(stage);
    if (mlir::failed(domain)) {
      return mlir::failure();
    }
    domains[stage.getOperation()] = *domain;
    if (!*domain) {
      stage_tiles[stage.getOperation()] =
          llvm::to_vector(llvm::seq<int64_t>(0, grid_size));
      continue;
    }
    if (domain->getValue().getNumSymbols() == 0) {
      auto tiles = getPlainDomainTiles(stage, *domain, grid_size);
      if (mlir::failed(tiles)) {
        return mlir::failure();
      }
      stage_tiles[stage.getOperation()] = std::move(*tiles);
      continue;
    }

    llvm::SmallPtrSet<mlir::Value, 2> stage_fifos;
    for (const auto* fifos : {&accesses.writers, &accesses.readers}) {
      for (const auto& [fifo, fifo_stages] : *fifos) {
        if (getFifoGroups(fifo) && llvm::is_contained(fifo_stages, stage)) {
          stage_fifos.insert(fifo);
        }
      }
    }
    if (stage_fifos.size() != 1) {
      stage.emitError() << "a domain with the group symbol is only valid on a "
                           "stage that writes or reads exactly one cross-core "
                           "fifo, but this stage writes or reads "
                        << stage_fifos.size();
      return mlir::failure();
    }
  }

  if (mlir::failed(resolveCrossCoreFifos(accesses, domains, grid_size,
                                         stage_tiles, channels))) {
    return mlir::failure();
  }

  // Each component's units run one program unit, so all leaf stages of a
  // component must execute on the same tiles. A stage that wraps a pipeline
  // does no work itself: its units are those of the stages inside.
  llvm::DenseMap<ResourceType, mlir::ktdf::StageOp> first_stages;
  for (mlir::ktdf::StageOp stage : stages) {
    if (!mlir::ktdf::isLeafStage(stage)) {
      continue;
    }
    auto units_attr = stage.getApplicableUnitsAttr();
    if (!units_attr) {
      continue;
    }
    const auto& tiles = stage_tiles.at(stage.getOperation());
    for (mlir::Attribute component : units_attr.getValue()) {
      const auto resource = llvm::dyn_cast<ResourceType>(component);
      if (!resource) {
        continue;
      }
      auto [it, inserted] = component_tiles.try_emplace(resource, tiles);
      if (inserted) {
        first_stages[resource] = stage;
        continue;
      }
      if (it->second != tiles) {
        auto diag = stage.emitError()
                    << "stages of kind " << resource
                    << " have different domains, which is not supported yet";
        diag.attachNote(first_stages.at(resource).getLoc())
            << "another stage of kind " << resource;
        return diag;
      }
    }
  }

  LDBG(1) << "Stage domains resolved for " << component_tiles.size()
          << " components";
  return mlir::success();
}
