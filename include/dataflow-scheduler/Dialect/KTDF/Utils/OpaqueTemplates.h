//===-- OpaqueTemplates.h ---------------------------------------*- c++ -*-===//
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
// Template names carried by `ktdf.opaque`: the contract between the transform
// that classifies an op as needing a target intrinsic and the lowering arm that
// expands it. The two live in different libraries, hence the shared header.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_OPAQUETEMPLATES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_OPAQUETEMPLATES_H_

#include <llvm/ADT/StringRef.h>

namespace mlir::ktdf {

/// Marks an on-stick (lane-axis) reduction: the innermost, stride-1 SIMD lane
/// axis is reduced within a single vector register rather than accumulated
/// across registers over time.
inline constexpr llvm::StringLiteral kLaneReduceTemplateName = "LANE_REDUCE";

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_OPAQUETEMPLATES_H_
