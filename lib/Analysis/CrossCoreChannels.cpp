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

#include <string>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace {

/// Gets the values in [@p begin, @p end) that the set @p set, with one
/// dimension and no symbols, contains, in increasing order.
auto getPointsInRange(mlir::IntegerSet set, int64_t begin, int64_t end)
    -> llvm::SmallVector<int64_t> {
  assert(set.getNumDims() == 1 && set.getNumSymbols() == 0 &&
         "expected a set over one dimension");
  // Evaluating the constraints at a value gives one result per constraint:
  // the value is in the set if the equalities give zero and the inequalities
  // give no negative result.
  const auto constraints =
      mlir::AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                           set.getConstraints(), set.getContext());
  llvm::SmallVector<int64_t> points;
  for (int64_t point = begin; point < end; ++point) {
    const auto values = constraints.compose({point});
    const bool contains =
        llvm::all_of(llvm::enumerate(values), [&](const auto& constraint) {
          return set.isEq(constraint.index()) ? constraint.value() == 0
                                              : constraint.value() >= 0;
        });
    if (contains) {
      points.push_back(point);
    }
  }
  return points;
}

/// Gets the domain over the tile that the stage domain @p domain, with one
/// dimension and one symbol, the group, gives for the group @p group.
auto getGroupDomain(mlir::IntegerSet domain, int64_t group)
    -> mlir::IntegerSet {
  mlir::MLIRContext* context = domain.getContext();
  return domain.replaceDimsAndSymbols(
      {mlir::getAffineDimExpr(0, context)},
      {mlir::getAffineConstantExpr(group, context)}, /*numResultDims=*/1,
      /*numResultSyms=*/0);
}

/// Prints @p tiles as a set, e.g. `{0, 1, 2}`.
auto printTiles(llvm::ArrayRef<int64_t> tiles) -> std::string {
  std::string printed;
  llvm::raw_string_ostream os(printed);
  os << "{";
  llvm::interleaveComma(tiles, os);
  os << "}";
  return printed;
}

/// Gets the union of the tiles that @p get_tiles gives for each of @p groups,
/// in increasing order.
template <typename GetTiles>
auto getUnion(llvm::ArrayRef<scheduler::ChannelGroup> groups,
              GetTiles get_tiles) -> llvm::SmallVector<int64_t> {
  llvm::SmallVector<int64_t> tiles;
  for (const scheduler::ChannelGroup& group : groups) {
    llvm::append_range(tiles, get_tiles(group));
  }
  llvm::sort(tiles);
  tiles.erase(llvm::unique(tiles), tiles.end());
  return tiles;
}

}  // namespace

auto scheduler::getFifoGroups(mlir::Value fifo) -> mlir::IntegerSetAttr {
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

  return alloc_op->getAttrOfType<mlir::IntegerSetAttr>(kFifoGroupsAttrName);
}

auto scheduler::getDomainTiles(mlir::IntegerSet domain, int64_t grid_size)
    -> llvm::SmallVector<int64_t> {
  assert(domain.getNumDims() == 1 && domain.getNumSymbols() == 0 &&
         "expected a set over the tile");
  return getPointsInRange(domain, 0, grid_size);
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

auto scheduler::resolveChannelGroups(
    mlir::IntegerSetAttr groups, mlir::IntegerSetAttr producer_domain,
    mlir::IntegerSetAttr consumer_domain, int64_t grid_size,
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error)
    -> mlir::FailureOr<llvm::SmallVector<ChannelGroup>> {
  // The group indices: the constant bounds of the group domain, then each
  // value in between that it contains.
  const mlir::IntegerSet group_set = groups.getValue();
  if (group_set.getNumDims() != 1 || group_set.getNumSymbols() != 0) {
    emit_error() << "group domain " << groups
                 << " must be a set over the group alone, with one dimension "
                    "and no symbols, but it has "
                 << group_set.getNumDims() << " dimensions and "
                 << group_set.getNumSymbols() << " symbols";
    return mlir::failure();
  }
  const mlir::affine::FlatAffineValueConstraints group_constraints(group_set);
  if (group_constraints.isEmpty()) {
    emit_error() << "group domain " << groups << " is empty";
    return mlir::failure();
  }
  const auto lower =
      group_constraints.getConstantBound64(mlir::presburger::BoundType::LB, 0);
  const auto upper =
      group_constraints.getConstantBound64(mlir::presburger::BoundType::UB, 0);
  if (!lower || !upper) {
    emit_error() << "group domain " << groups
                 << " must be bounded, with a constant lower and upper bound";
    return mlir::failure();
  }
  const auto group_ids = getPointsInRange(group_set, *lower, *upper + 1);
  if (group_ids.empty()) {
    emit_error() << "group domain " << groups << " is empty";
    return mlir::failure();
  }

  // The tiles each stage domain selects for each group.
  const auto getTilesPerGroup = [&](llvm::StringRef side,
                                    mlir::IntegerSetAttr domain_attr)
      -> mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<int64_t>>> {
    const mlir::IntegerSet domain = domain_attr.getValue();
    if (domain.getNumDims() != 1 || domain.getNumSymbols() != 1) {
      emit_error() << side << " domain " << domain_attr
                   << " must be a set over the compute tile with one symbol, "
                      "the group, but it has "
                   << domain.getNumDims() << " dimensions and "
                   << domain.getNumSymbols() << " symbols";
      return mlir::failure();
    }
    llvm::SmallVector<llvm::SmallVector<int64_t>> tiles_per_group;
    for (const int64_t group : group_ids) {
      const mlir::IntegerSet group_domain = getGroupDomain(domain, group);
      if (!isDomainWithinGrid(group_domain, grid_size)) {
        emit_error() << side << " domain " << domain_attr
                     << " has tiles outside the grid [0, " << grid_size
                     << ") in group " << group;
        return mlir::failure();
      }
      auto tiles = getDomainTiles(group_domain, grid_size);
      if (tiles.empty()) {
        emit_error() << side << " domain " << domain_attr
                     << " has no tile in group " << group;
        return mlir::failure();
      }
      tiles_per_group.push_back(std::move(tiles));
    }
    return tiles_per_group;
  };
  auto producers = getTilesPerGroup("producer", producer_domain);
  if (mlir::failed(producers)) {
    return mlir::failure();
  }
  auto consumers = getTilesPerGroup("consumer", consumer_domain);
  if (mlir::failed(consumers)) {
    return mlir::failure();
  }

  llvm::SmallVector<ChannelGroup> channel_groups;
  for (auto [group, group_producers, group_consumers] :
       llvm::zip_equal(group_ids, *producers, *consumers)) {
    channel_groups.push_back(
        {group, std::move(group_producers), std::move(group_consumers)});
  }

  // What a multicast group of the DFIR expresses: one producer, and tiles
  // that are a producer or a consumer of one group at most.
  const auto checkOneGroupPerTile =
      [&](llvm::StringRef role, const ChannelGroup& group,
          llvm::ArrayRef<int64_t> tiles,
          llvm::DenseMap<int64_t, int64_t>& group_of) -> mlir::LogicalResult {
    for (const int64_t tile : tiles) {
      auto [it, inserted] = group_of.try_emplace(tile, group.id);
      if (!inserted) {
        emit_error() << role << " tile " << tile << " is in the groups "
                     << it->second << " and " << group.id
                     << ", but a tile is a " << role
                     << " of at most one group for now";
        return mlir::failure();
      }
    }
    return mlir::success();
  };
  llvm::DenseMap<int64_t, int64_t> producer_group;
  llvm::DenseMap<int64_t, int64_t> consumer_group;
  for (const ChannelGroup& group : channel_groups) {
    if (group.producers.size() != 1) {
      emit_error() << "producer domain " << producer_domain << " has "
                   << group.producers.size() << " tiles "
                   << printTiles(group.producers) << " in group " << group.id
                   << ", but a group has exactly one producer for now";
      return mlir::failure();
    }
    if (mlir::failed(checkOneGroupPerTile("producer", group, group.producers,
                                          producer_group)) ||
        mlir::failed(checkOneGroupPerTile("consumer", group, group.consumers,
                                          consumer_group))) {
      return mlir::failure();
    }
  }
  return channel_groups;
}

auto scheduler::getProducerTiles(llvm::ArrayRef<ChannelGroup> groups)
    -> llvm::SmallVector<int64_t> {
  return getUnion(groups,
                  [](const ChannelGroup& group) { return group.producers; });
}

auto scheduler::getConsumerTiles(llvm::ArrayRef<ChannelGroup> groups)
    -> llvm::SmallVector<int64_t> {
  return getUnion(groups,
                  [](const ChannelGroup& group) { return group.consumers; });
}

auto scheduler::getProducerOf(llvm::ArrayRef<ChannelGroup> groups,
                              int64_t consumer) -> std::optional<int64_t> {
  for (const ChannelGroup& group : groups) {
    if (llvm::is_contained(group.consumers, consumer)) {
      assert(group.producers.size() == 1 && "expected one producer per group");
      return group.producers.front();
    }
  }
  return std::nullopt;
}

auto scheduler::getSelfFeedingConsumers(llvm::ArrayRef<ChannelGroup> groups)
    -> llvm::SmallVector<int64_t> {
  return getUnion(groups, [](const ChannelGroup& group) {
    llvm::SmallVector<int64_t> self_feeding;
    for (const int64_t consumer : group.consumers) {
      if (llvm::is_contained(group.producers, consumer)) {
        self_feeding.push_back(consumer);
      }
    }
    return self_feeding;
  });
}
