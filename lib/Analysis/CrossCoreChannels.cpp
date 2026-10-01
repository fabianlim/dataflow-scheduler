//===-- CrossCoreChannels.cpp -----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Analysis/CrossCoreChannels.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

auto scheduler::getFifoPeer(mlir::Value fifo) -> mlir::AffineMapAttr {
  auto private_result = mlir::dyn_cast<mlir::OpResult>(fifo);
  auto private_op =
      private_result
          ? mlir::dyn_cast<mlir::ktdf::PrivateOp>(private_result.getOwner())
          : nullptr;
  if (!private_op) {
    return nullptr;
  }

  const mlir::Value slot =
      private_op.getYieldOp().getOperands()[private_result.getResultNumber()];
  auto alloc_op = slot.getDefiningOp<mlir::ktdf::FifoAllocateOp>();
  if (!alloc_op) {
    return nullptr;
  }

  return alloc_op->getAttrOfType<mlir::AffineMapAttr>(kFifoPeerAttrName);
}

auto scheduler::getPeerImage(mlir::AffineMap peer,
                             llvm::ArrayRef<int64_t> consumer_tiles)
    -> llvm::SmallVector<int64_t> {
  assert(peer.getNumDims() == 1 && peer.getNumSymbols() == 0 &&
         peer.getNumResults() == 1 && "expected a map from tile to tile");
  llvm::SmallVector<int64_t> producer_tiles;
  for (const int64_t consumer : consumer_tiles) {
    producer_tiles.push_back(peer.compose({consumer}).front());
  }
  llvm::sort(producer_tiles);
  producer_tiles.erase(llvm::unique(producer_tiles), producer_tiles.end());
  return producer_tiles;
}

auto scheduler::getDomainTiles(mlir::IntegerSet domain, int64_t grid_size)
    -> llvm::SmallVector<int64_t> {
  assert(domain.getNumDims() == 1 && domain.getNumSymbols() == 0 &&
         "expected a set over the tile");
  // Evaluating the constraints at a tile gives one value per constraint: the
  // tile is in the domain if the equalities give zero and the inequalities
  // give no negative value.
  const auto constraints = mlir::AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0, domain.getConstraints(),
      domain.getContext());
  llvm::SmallVector<int64_t> tiles;
  for (int64_t tile = 0; tile < grid_size; ++tile) {
    const auto values = constraints.compose({tile});
    const bool contains =
        llvm::all_of(llvm::enumerate(values), [&](const auto& constraint) {
          return domain.isEq(constraint.index()) ? constraint.value() == 0
                                                 : constraint.value() >= 0;
        });
    if (contains) {
      tiles.push_back(tile);
    }
  }
  return tiles;
}

auto scheduler::isDomainWithinGrid(mlir::IntegerSet domain, int64_t grid_size)
    -> bool {
  assert(domain.getNumDims() == 1 && domain.getNumSymbols() == 0 &&
         "expected a set over the tile");
  // Enumerating the grid cannot see the tiles outside it, so this asks
  // whether the domain has any tile below 0 or any tile from grid_size on.
  const auto hasTileWhere = [&](mlir::presburger::BoundType type,
                                int64_t bound) {
    mlir::affine::FlatAffineValueConstraints constraints(domain);
    constraints.addBound(type, /*pos=*/0, bound);
    return !constraints.isEmpty();
  };
  return !hasTileWhere(mlir::presburger::BoundType::UB, -1) &&
         !hasTileWhere(mlir::presburger::BoundType::LB, grid_size);
}
