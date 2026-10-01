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
