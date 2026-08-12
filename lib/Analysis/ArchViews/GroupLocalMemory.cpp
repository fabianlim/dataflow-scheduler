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

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"

#include <llvm/ADT/SmallPtrSet.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace scheduler::arch_view;

namespace {

/// Returns the kind of the memory that @p endpoint denotes, or nullptr if the
/// endpoint is not a memory.  Groups are IsolatedFromAbove, so a memory shared
/// into a group appears as a block argument and has to be traced back through
/// the capture list of each group it was handed down through.
mlir::Attribute resolveMemoryKind(mlir::Value endpoint) {
  while (auto arg = mlir::dyn_cast<mlir::BlockArgument>(endpoint)) {
    auto group =
        mlir::dyn_cast<mlir::ktdf_arch::GroupOp>(arg.getOwner()->getParentOp());
    if (!group || arg.getArgNumber() >= group.getSharedMemory().size())
      return {};
    endpoint = group.getSharedMemory()[arg.getArgNumber()];
  }
  auto mem_op = endpoint.getDefiningOp<mlir::ktdf_arch::MemoryOp>();
  if (!mem_op) return {};
  return mem_op.getKind();
}

}  // namespace

GroupLocalMemory::GroupLocalMemory(const mlir::ktdf_arch::Device& device)
    : DeviceView(device) {
  initialize();
}

void GroupLocalMemory::initialize() {
  // Walk every group in the device definition.  For each group, scan its
  // immediate children for exec_unit ops and memory ops.  For each exec_unit
  // kind the set of local memory kinds is intersected across all groups that
  // contain it — only memory kinds present in every such group are retained.
  // Ambiguity (intersection set size > 1) is not diagnosed here; it
  // surfaces as a nullptr result in getLocalMemoryKind if queried.
  //
  // The same walk records the inverse mapping, plus every memory kind that is
  // named as a datapath endpoint anywhere.  Co-location with an exec_unit does
  // not on its own make a memory private to it — a core-level scratchpad also
  // sits alongside its load/store units — so a memory that anything can
  // transfer through is not compute-unit-local, whatever its position in the
  // hierarchy.  Datapaths are collected across all groups because a shared
  // memory is reached through a capture, from a group below the one declaring
  // it.
  llvm::SmallPtrSet<mlir::Attribute, 4> transfer_endpoints;

  getDevice().getBodyRegion().walk([&](mlir::ktdf_arch::GroupOp group) {
    llvm::SmallVector<mlir::Attribute> exec_kinds;
    llvm::SmallPtrSet<mlir::Attribute, 1> mem_kinds;

    for (auto& op : group.getRegion().front().getOperations()) {
      if (auto exec_op =
              mlir::dyn_cast<mlir::ktdf_arch::ExecutionUnitOp>(&op)) {
        if (mlir::Attribute k = exec_op.getKind()) exec_kinds.push_back(k);
      } else if (auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(&op)) {
        if (mlir::Attribute k = mem_op.getKind()) mem_kinds.insert(k);
      } else if (auto path = mlir::dyn_cast<mlir::ktdf_arch::DatapathOp>(&op)) {
        for (mlir::Value endpoint : {path.getSource(), path.getTarget()})
          if (mlir::Attribute k = resolveMemoryKind(endpoint))
            transfer_endpoints.insert(k);
      }
    }

    // The forward direction records co-location only. TODO: restrict it to
    // exec_units with an explicit ktdf_arch.datapath to/from the memory —
    // co-location alone does not imply access. The inverse direction below
    // already applies that test, via the transfer_endpoints erase at the end.
    for (mlir::Attribute ek : exec_kinds) {
      auto [it, inserted] = exec_to_mem_kinds_.try_emplace(ek, mem_kinds);
      if (!inserted)
        it->second.remove_if(
            [&](mlir::Attribute mk) { return !mem_kinds.count(mk); });
      for (mlir::Attribute mk : mem_kinds) mem_to_exec_kinds_[mk].insert(ek);
    }

    return mlir::WalkResult::advance();
  });

  for (mlir::Attribute mk : transfer_endpoints) mem_to_exec_kinds_.erase(mk);
}

mlir::Attribute GroupLocalMemory::getLocalMemoryKind(
    mlir::Attribute exec_unit_kind) const {
  auto it = exec_to_mem_kinds_.find(exec_unit_kind);
  if (it == exec_to_mem_kinds_.end() || it->second.size() != 1) return nullptr;
  return *it->second.begin();
}

bool GroupLocalMemory::isComputeUnitLocal(mlir::Attribute memory_kind) const {
  return mem_to_exec_kinds_.contains(memory_kind);
}
