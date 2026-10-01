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

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <mlir/IR/Diagnostics.h>

#include "dataflow-scheduler/Analysis/CrossCoreChannels.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "ktdf-to-operand-lowering"

using namespace scheduler;

namespace {

/// Gets the tiles @p stage executes on: those of its domain, verified to be a
/// non-empty set over the tile within the grid, or every tile of the grid if
/// it has none.
mlir::FailureOr<llvm::SmallVector<int64_t>> getStageTiles(
    mlir::ktdf::StageOp stage, int grid_size) {
  const mlir::Attribute attr = stage->getAttr(kStageDomainAttrName);
  if (!attr) {
    return llvm::to_vector(llvm::seq<int64_t>(0, grid_size));
  }

  auto domain_attr = mlir::dyn_cast<mlir::IntegerSetAttr>(attr);
  if (!domain_attr) {
    stage.emitError() << "'" << kStageDomainAttrName
                      << "' must be an integer set";
    return mlir::failure();
  }
  const mlir::IntegerSet domain = domain_attr.getValue();
  if (domain.getNumDims() != 1 || domain.getNumSymbols() != 0) {
    stage.emitError() << "domain must be a set over the compute tile alone, "
                         "with one dimension and no symbols, but it has "
                      << domain.getNumDims() << " dimensions and "
                      << domain.getNumSymbols() << " symbols";
    return mlir::failure();
  }
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

/// Prints @p tiles as a set, e.g. `{0, 1, 2}`.
std::string printTiles(llvm::ArrayRef<int64_t> tiles) {
  std::string printed;
  llvm::raw_string_ostream os(printed);
  os << "{";
  llvm::interleaveComma(tiles, os);
  os << "}";
  return printed;
}

/// Verifies that the stage writing each peer fifo executes on the tiles that
/// feed the stage reading it: the image of the peer map over the reader's
/// tiles.
mlir::LogicalResult verifyPeerFifos(
    llvm::ArrayRef<mlir::ktdf::StageOp> stages,
    const llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>>&
        stage_tiles) {
  // Fifo slot -> the stages that write it and the stages that read it. A
  // stage is the innermost one around the fifo access.
  llvm::MapVector<mlir::Value, llvm::SmallVector<mlir::ktdf::StageOp, 1>>
      writers;
  llvm::MapVector<mlir::Value, llvm::SmallVector<mlir::ktdf::StageOp, 1>>
      readers;
  for (mlir::ktdf::StageOp stage : stages) {
    stage.walk([&](mlir::Operation* op) {
      if (op->getParentOfType<mlir::ktdf::StageOp>() != stage) {
        return;
      }
      if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
        if (transfer.isDestFifo()) {
          writers[transfer.getDestination()].push_back(stage);
        }
        if (transfer.isSourceFifo()) {
          readers[transfer.getSource()].push_back(stage);
        }
      } else if (auto write = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op)) {
        writers[write.getFifoSlot()].push_back(stage);
      } else if (auto read = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op)) {
        readers[read.getFifoSlot()].push_back(stage);
      }
    });
  }

  for (auto& [fifo, fifo_writers] : writers) {
    const mlir::AffineMapAttr peer = getFifoPeer(fifo);
    if (!peer) {
      continue;
    }
    const mlir::AffineMap peer_map = peer.getValue();
    if (peer_map.getNumDims() != 1 || peer_map.getNumSymbols() != 0 ||
        peer_map.getNumResults() != 1) {
      return fifo_writers.front().emitError()
             << "the peer relation " << peer
             << " of the fifo this stage writes must map one tile to one "
                "tile, with no symbols";
    }

    for (mlir::ktdf::StageOp reader : readers.lookup(fifo)) {
      const auto feeding_tiles =
          getPeerImage(peer_map, stage_tiles.at(reader.getOperation()));
      for (mlir::ktdf::StageOp writer : fifo_writers) {
        const auto& writer_tiles = stage_tiles.at(writer.getOperation());
        if (writer_tiles == feeding_tiles) {
          continue;
        }
        auto diag = writer.emitError()
                    << "stage writes a peer fifo on the tiles "
                    << printTiles(writer_tiles)
                    << ", but the tiles that feed the stage reading it are "
                    << printTiles(feeding_tiles)
                    << ": its domain must be the image of the peer relation "
                    << peer << " over the domain of the reading stage";
        diag.attachNote(reader.getLoc()) << "the stage reading it";
        return diag;
      }
    }
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::resolveComponentTiles(
    llvm::ArrayRef<mlir::ktdf::StageOp> stages, int grid_size,
    ComponentTiles& component_tiles) {
  LDBG(1) << "Step 2b: Resolve stage domains";

  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>> stage_tiles;
  for (mlir::ktdf::StageOp stage : stages) {
    auto tiles = getStageTiles(stage, grid_size);
    if (mlir::failed(tiles)) {
      return mlir::failure();
    }
    stage_tiles[stage.getOperation()] = std::move(*tiles);
  }

  if (mlir::failed(verifyPeerFifos(stages, stage_tiles))) {
    return mlir::failure();
  }

  // Each component's units run one program unit, so all stages of a
  // component must execute on the same tiles.
  llvm::DenseMap<ResourceType, mlir::ktdf::StageOp> first_stages;
  for (mlir::ktdf::StageOp stage : stages) {
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
