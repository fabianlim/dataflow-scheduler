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
// peer relation: an affine map from each consumer core to the producer core
// that feeds it. The producer stage writes the fifo on the producer core and
// the consumer stage reads it on the consumer core, so the data crosses cores
// through switches instead of staying on one core. A fifo without the
// relation is an ordinary same-core fifo.
//
// The cores a stage executes on are its domain: an integer set over the
// compute tile, the index into the function's 1-D `grid`, carried by the
// `ktdf.stage`. The producer stage of a cross-core channel must execute on
// exactly the cores that feed its consumer stage, the image of the peer map
// over the consumer domain. Domains are small, so they are evaluated by
// enumerating the tiles of the grid.
//
// For now the analysis reads the peer relation of a fifo and the domain of a
// stage, and computes the producer set. It will grow to derive the rest of
// the channel from them: the consumers of each producer, the group a consumer
// belongs to, and the ring direction each producer sends in.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Value.h>

namespace scheduler {

/// Name of the discardable attribute of a `ktdf.fifo.allocate` that makes its
/// fifos cross cores: an affine map from each consumer core to the producer
/// core that feeds it.
inline constexpr llvm::StringLiteral kFifoPeerAttrName =
    "dataflow_scheduler.peer";

/// Gets the peer relation of the fifo slot @p fifo, which a `ktdf.private`
/// yields from a `ktdf.fifo.allocate`.
///
/// @retval nullptr      @p fifo does not cross cores.
/// @retval AffineMapAttr  Consumer core to producer core.
[[nodiscard]]
auto getFifoPeer(mlir::Value fifo) -> mlir::AffineMapAttr;

/// Gets the producer tiles that feed @p consumer_tiles through the peer
/// relation @p peer: the image of the peer map over them, in increasing order
/// and without duplicates. @p peer must map one dimension to one result and
/// have no symbols.
[[nodiscard]]
auto getPeerImage(mlir::AffineMap peer, llvm::ArrayRef<int64_t> consumer_tiles)
    -> llvm::SmallVector<int64_t>;

/// Name of the discardable attribute of a `ktdf.stage` that restricts the
/// compute tiles it executes on: an integer set with one dimension, the
/// compute tile, and no symbols. A stage without it executes on every tile of
/// the grid.
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

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_
