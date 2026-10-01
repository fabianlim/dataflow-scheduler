//===-- CrossCoreChannels.h -------------------------------------*- c++ -*-===//
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
// Cross-core channels.
//
// A cross-core channel is a ktdf fifo whose `ktdf.fifo.allocate` carries a
// group domain G: an integer set over the group index g. The producer stage
// writes the fifo and the consumer stage reads it, on different cores, so the
// data crosses cores through switches instead of staying on one core. A fifo
// without a group domain is an ordinary same-core fifo.
//
// The tiles a stage executes on are given by its domain, an integer set over
// the compute tile, the index into the function's 1-D `grid`, carried by the
// `ktdf.stage`. The domain of a stage that writes or reads a cross-core fifo
// has one symbol, the group: substituting g gives the tiles that take part in
// group g. For the producer stage that is the gather side P(g), the producers
// that feed group g; for the consumer stage the scatter side C(g), the
// consumers group g delivers to. For each g in G, P(g) sends to C(g). Any
// other domain has no symbol and is a plain set of tiles.
//
// Groups and domains are small, so they are evaluated by enumeration: the
// constant bounds of G, then each value in between, then each tile of the
// grid. The analysis resolves the groups of a channel from its group domain
// and its two stage domains, and derives from them the tiles of each stage,
// the producer of each consumer and the consumers that feed themselves. It
// will grow to derive what lowering needs beyond these, such as the ring
// direction each producer sends in.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_

#include <optional>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace scheduler {

/// Name of the discardable attribute of a `ktdf.fifo.allocate` that makes its
/// fifos cross cores: the group domain, an integer set with one dimension,
/// the group, and no symbols.
inline constexpr llvm::StringLiteral kFifoGroupsAttrName =
    "dataflow_scheduler.groups";

/// Gets the group domain of the fifo slot @p fifo, which a `ktdf.private`
/// yields from a `ktdf.fifo.allocate`.
///
/// @retval nullptr        @p fifo does not cross cores.
/// @retval IntegerSetAttr The group domain.
[[nodiscard]]
auto getFifoGroups(mlir::Value fifo) -> mlir::IntegerSetAttr;

/// Name of the discardable attribute of a `ktdf.stage` that restricts the
/// compute tiles it executes on: an integer set with one dimension, the
/// compute tile, and either no symbols or one symbol, the group of the
/// cross-core fifo the stage writes or reads. A stage without it executes on
/// every tile of the grid.
inline constexpr llvm::StringLiteral kStageDomainAttrName =
    "dataflow_scheduler.domain";

/// Gets the tiles in [0, @p grid_size) that the stage domain @p domain
/// contains, in increasing order. @p domain must have one dimension and no
/// symbols.
[[nodiscard]]
auto getDomainTiles(mlir::IntegerSet domain, int64_t grid_size)
    -> llvm::SmallVector<int64_t>;

/// Checks whether the stage domain @p domain contains no tile outside
/// [0, @p grid_size). @p domain must have one dimension and no symbols.
[[nodiscard]]
auto isDomainWithinGrid(mlir::IntegerSet domain, int64_t grid_size) -> bool;

/// One group of a cross-core channel: its producers send to its consumers.
struct ChannelGroup {
  /// The group index g.
  int64_t id;
  /// P(g), the producer tiles, in increasing order.
  llvm::SmallVector<int64_t> producers;
  /// C(g), the consumer tiles, in increasing order.
  llvm::SmallVector<int64_t> consumers;
};

/// Resolves the groups of a cross-core channel with the group domain
/// @p groups, whose producer stage has the domain @p producer_domain and
/// whose consumer stage has the domain @p consumer_domain, on a grid of
/// @p grid_size tiles: for each g in the group domain, in increasing order,
/// its producers P(g) and consumers C(g).
///
/// The group domain must have one dimension and no symbols, and be bounded
/// and non-empty. The stage domains must have one dimension and exactly one
/// symbol, the group. For each g, P(g) and C(g) must be non-empty and within
/// the grid. For now, as a multicast group of the DFIR expresses, P(g) must
/// have exactly one tile and each tile can be a consumer of at most one group
/// and a producer of at most one group. A violation is reported through
/// @p emit_error.
[[nodiscard]]
auto resolveChannelGroups(
    mlir::IntegerSetAttr groups, mlir::IntegerSetAttr producer_domain,
    mlir::IntegerSetAttr consumer_domain, int64_t grid_size,
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error)
    -> mlir::FailureOr<llvm::SmallVector<ChannelGroup>>;

/// Gets the tiles the producer stage of a channel with the groups @p groups
/// executes on: the union of P(g), in increasing order.
[[nodiscard]]
auto getProducerTiles(llvm::ArrayRef<ChannelGroup> groups)
    -> llvm::SmallVector<int64_t>;

/// Gets the tiles the consumer stage of a channel with the groups @p groups
/// executes on: the union of C(g), in increasing order.
[[nodiscard]]
auto getConsumerTiles(llvm::ArrayRef<ChannelGroup> groups)
    -> llvm::SmallVector<int64_t>;

/// Gets the producer that feeds the consumer tile @p consumer in a channel
/// with the groups @p groups: the one tile of P(g) for the g with @p consumer
/// in C(g).
///
/// @retval std::nullopt @p consumer is in no group.
[[nodiscard]]
auto getProducerOf(llvm::ArrayRef<ChannelGroup> groups, int64_t consumer)
    -> std::optional<int64_t>;

/// Gets the consumers of a channel with the groups @p groups that feed
/// themselves: the tiles in both C(g) and P(g) for some g, in increasing
/// order.
[[nodiscard]]
auto getSelfFeedingConsumers(llvm::ArrayRef<ChannelGroup> groups)
    -> llvm::SmallVector<int64_t>;

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_
