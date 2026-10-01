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
// For now the analysis only reads the peer relation of a fifo. It will grow
// to derive the rest of the channel from it: the producer set (the image of
// the peer map over the consumer domain), the consumers of each producer, the
// group a consumer belongs to, and the ring direction each producer sends in.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
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

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_CROSSCORECHANNELS_H_
