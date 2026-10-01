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
//
// Handle Cross-Core Stages Pass
//
// A cross-core channel delivers the data of the producers P(g) of each group
// to its consumers C(g). The ring cannot deliver a tile's data to the tile
// itself, so lowering carries only C(g) without P(g) over it. This pass
// carries the rest, the self-deliveries, by a local copy on the tiles that
// deliver to themselves, which runs concurrently with the channel.
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <tuple>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/CrossCoreChannels.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/ApplicableUnits.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Utils/Utils.h"

#define DEBUG_TYPE "handle-cross-core-stages"

namespace scheduler {
#define GEN_PASS_DEF_HANDLECROSSCORESTAGESPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace mlir;
using namespace scheduler;
using RoutingGraph = arch_view::RoutingGraph;

namespace {

/// The units a local copy passes through on one core, from memory to memory.
struct LocalRoute {
  ResourceType load;
  ResourceType compute;
  ResourceType store;
};

/// The self-deliveries of a cross-core channel, and how they are copied.
struct SelfDelivery {
  /// The transfer into the channel, on the producer stage.
  ktdf::DataTransferOp write;
  /// The transfer out of the channel, on the consumer stage.
  ktdf::DataTransferOp read;
  /// The self-delivering tiles, as a plain stage domain.
  IntegerSetAttr domain;
  /// The route of the local copy.
  LocalRoute route;
};

/// Gets the memory @p value is in, or nullptr if it is not a memref in one.
auto getMemory(Value value) -> ResourceType {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type ? dyn_cast_or_null<ResourceType>(type.getMemorySpace()) : nullptr;
}

/// Finds the route of a local copy from the memory @p source to the memory
/// @p target in @p graph: through a load unit, a compute unit and a store
/// unit, which keeps it on one core. A failure to find exactly one such route
/// is reported on @p op.
auto findLocalRoute(ResourceType source, ResourceType target,
                    const RoutingGraph& graph, Operation* op)
    -> FailureOr<LocalRoute> {
  using RK = RoutingGraph::ResourceNode::ResourceKind;
  const auto getNeighbors = [&](RoutingGraph::NodeId node, RK kind) {
    return llvm::to_vector(llvm::make_filter_range(
        graph.getNeighbors(node), [&](RoutingGraph::NodeId neighbor) {
          return graph.getNode(neighbor)->kind == kind;
        }));
  };
  const auto getResource = [&](RoutingGraph::NodeId node) {
    return graph.getNode(node)->resource;
  };

  // Units, not nodes: the graph may hold several nodes of one unit kind.
  llvm::SmallSetVector<std::tuple<ResourceType, ResourceType, ResourceType>, 1>
      routes;
  for (RoutingGraph::NodeId memory : graph.getNodeIdsForResource(source)) {
    for (RoutingGraph::NodeId load : getNeighbors(memory, RK::LoadStoreUnit)) {
      for (RoutingGraph::NodeId compute : getNeighbors(load, RK::Compute)) {
        for (RoutingGraph::NodeId store :
             getNeighbors(compute, RK::LoadStoreUnit)) {
          if (llvm::is_contained(
                  llvm::map_range(graph.getNeighbors(store), getResource),
                  target)) {
            routes.insert(
                {getResource(load), getResource(compute), getResource(store)});
          }
        }
      }
    }
  }
  if (routes.size() != 1) {
    op->emitError() << "expected one same-core route from " << source << " to "
                    << target
                    << " through a load unit, a compute unit and a store unit "
                       "for the local copy of the self-deliveries, but found "
                    << routes.size();
    return failure();
  }
  auto [load, compute, store] = routes.front();
  return LocalRoute{load, compute, store};
}

/// Gets the plain stage domain of the tiles @p tiles, in increasing order, if
/// they are one interval.
auto getIntervalDomain(ArrayRef<int64_t> tiles, MLIRContext* context)
    -> std::optional<IntegerSet> {
  const int64_t first = tiles.front();
  const int64_t last = tiles.back();
  if (last - first + 1 != static_cast<int64_t>(tiles.size())) {
    return std::nullopt;
  }
  const AffineExpr tile = getAffineDimExpr(0, context);
  if (first == last) {
    return IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0, {tile - first},
                           {true});
  }
  return IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0,
                         {tile - first, last - tile}, {false, false});
}

/// Finds the cross-core channels of @p pipeline that have self-deliveries,
/// on a grid of @p grid_size tiles, and how to copy them along @p graph.
auto planSelfDeliveries(ktdf::PipelineOp pipeline, int grid_size,
                        const RoutingGraph& graph,
                        SmallVectorImpl<SelfDelivery>& deliveries)
    -> LogicalResult {
  for (Value fifo : pipeline.getPrivateOp().getResults()) {
    const IntegerSetAttr groups = getFifoGroups(fifo);
    if (!groups) {
      continue;
    }

    // The channel: one transfer into the fifo and one out of it.
    SmallVector<ktdf::DataTransferOp> writes;
    SmallVector<ktdf::DataTransferOp> reads;
    for (Operation* user : fifo.getUsers()) {
      auto transfer = dyn_cast<ktdf::DataTransferOp>(user);
      if (!transfer) {
        return user->emitError()
               << "only data transfers can access a cross-core fifo for now";
      }
      (transfer.getDestination() == fifo ? writes : reads).push_back(transfer);
    }
    if (writes.size() != 1 || reads.size() != 1) {
      return pipeline.emitError()
             << "a cross-core fifo must be written by one data transfer and "
                "read by one for now, but it is written by "
             << writes.size() << " and read by " << reads.size();
    }
    ktdf::DataTransferOp write = writes.front();
    ktdf::DataTransferOp read = reads.front();

    auto producer = write->getParentOfType<ktdf::StageOp>();
    auto consumer = read->getParentOfType<ktdf::StageOp>();
    const auto producer_domain =
        producer->getAttrOfType<IntegerSetAttr>(kStageDomainAttrName);
    const auto consumer_domain =
        consumer->getAttrOfType<IntegerSetAttr>(kStageDomainAttrName);
    if (!producer_domain || !consumer_domain) {
      return (producer_domain ? consumer : producer).emitError()
             << "a stage that writes or reads a cross-core fifo must have a "
                "domain with the group symbol";
    }
    auto channel_groups = resolveChannelGroups(
        groups, producer_domain, consumer_domain, grid_size, [&] {
          auto diag = producer.emitError()
                      << "invalid groups of the cross-core fifo this stage "
                         "writes: ";
          diag.attachNote(consumer.getLoc()) << "the stage reading the fifo";
          return diag;
        });
    if (failed(channel_groups)) {
      return failure();
    }

    const auto tiles = getSelfDeliveries(*channel_groups);
    if (tiles.empty()) {
      continue;
    }
    const auto domain = getIntervalDomain(tiles, pipeline.getContext());
    if (!domain) {
      return producer.emitError()
             << "self-deliveries on tiles that are not one interval are not "
                "supported yet";
    }
    if (cast<ktdf::FifoSlotType>(fifo.getType()).isDynamicNumElements()) {
      return pipeline.emitError() << "self-deliveries of a cross-core fifo of "
                                     "dynamic size are not supported yet";
    }
    auto route =
        findLocalRoute(getMemory(write.getSource()),
                       getMemory(read.getDestination()), graph, producer);
    if (failed(route)) {
      return failure();
    }
    LDBG(1) << "Self-deliveries on " << IntegerSetAttr::get(*domain)
            << " of the cross-core fifo " << groups;
    deliveries.push_back({write, read, IntegerSetAttr::get(*domain), *route});
  }
  return success();
}

/// Builds the local copy of @p delivery: a pipeline that loads the producer's
/// data, passes it through an identity on the compute unit and stores it
/// where the consumer stores it, on the self-delivering tiles.
auto buildLocalCopy(OpBuilder& builder, Location loc, SelfDelivery delivery)
    -> ktdf::PipelineOp {
  MLIRContext* context = builder.getContext();
  const Value channel = delivery.write.getDestination();
  const auto channel_type = cast<ktdf::FifoSlotType>(channel.getType());
  const LocalRoute& route = delivery.route;
  const auto getFifoType = [&](ResourceType source, ResourceType target) {
    return ktdf::FifoSlotType::get(context, source, target,
                                   channel_type.getNumElements(),
                                   channel_type.getElementType());
  };
  const auto to_compute = getFifoType(route.load, route.compute);
  const auto from_compute = getFifoType(route.compute, route.store);
  const auto token = ktdf::TokenType::get(context);

  auto pipeline = ktdf::PipelineOp::create(builder, loc);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(pipeline.getBody());
  auto private_op = ktdf::PrivateOp::create(
      builder, loc, TypeRange{to_compute, from_compute, token, token},
      [&](OpBuilder& builder, Location loc) {
        auto into = ktdf::FifoAllocateOp::create(
            builder, loc, TypeRange{to_compute}, ValueRange{});
        auto out_of = ktdf::FifoAllocateOp::create(
            builder, loc, TypeRange{from_compute}, ValueRange{});
        auto loaded = ktdf::CreateTokenOp::create(builder, loc);
        auto computed = ktdf::CreateTokenOp::create(builder, loc);
        ktdf::PrivateYieldOp::create(
            builder, loc,
            ValueRange{into.getResult(0), out_of.getResult(0), loaded,
                       computed});
      });
  const Value into_compute = private_op.getResult(0);
  const Value out_of_compute = private_op.getResult(1);
  const Value loaded = private_op.getResult(2);
  const Value computed = private_op.getResult(3);

  const auto createStage = [&](ValueRange depends_in, ValueRange depends_out,
                               ResourceType unit,
                               function_ref<void(OpBuilder&, Location)> body) {
    auto stage =
        ktdf::StageOp::create(builder, loc, depends_in, depends_out, body);
    stage.setApplicableUnitsAttr(builder.getArrayAttr({unit}));
    stage->setAttr(kStageDomainAttrName, delivery.domain);
  };
  // The producer's transfer, into the compute unit instead of the channel.
  createStage({}, loaded, route.load, [&](OpBuilder& builder, Location) {
    IRMapping mapping;
    mapping.map(channel, into_compute);
    builder.clone(*delivery.write, mapping);
  });
  createStage(
      loaded, computed, route.compute, [&](OpBuilder& builder, Location loc) {
        const auto type = RankedTensorType::get({channel_type.getNumElements()},
                                                channel_type.getElementType());
        Value value =
            ktdf::ReadFromFifoOp::create(builder, loc, type, into_compute);
        Value init = tensor::EmptyOp::create(builder, loc, type.getShape(),
                                             type.getElementType());
        const AffineMap identity = builder.getMultiDimIdentityMap(1);
        auto copy = linalg::GenericOp::create(
            builder, loc, TypeRange{type}, ValueRange{value}, ValueRange{init},
            ArrayRef<AffineMap>{identity, identity},
            ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
            [](OpBuilder& builder, Location loc, ValueRange args) {
              linalg::YieldOp::create(builder, loc, args.front());
            });
        ktdf::WriteToFifoOp::create(builder, loc, copy.getResult(0),
                                    out_of_compute);
      });
  // The consumer's transfer, out of the compute unit instead of the channel.
  createStage(computed, {}, route.store, [&](OpBuilder& builder, Location) {
    IRMapping mapping;
    mapping.map(channel, out_of_compute);
    builder.clone(*delivery.read, mapping);
  });
  return pipeline;
}

/// Replaces @p channel by an outer pipeline whose stages run concurrently,
/// with no token between them: one wraps @p channel, and one wraps the local
/// copy of each of @p deliveries.
void wrapWithLocalCopies(ktdf::PipelineOp channel,
                         ArrayRef<SelfDelivery> deliveries) {
  const Location loc = channel.getLoc();
  OpBuilder builder(channel);
  auto outer = ktdf::PipelineOp::create(builder, loc);
  builder.setInsertionPointToStart(outer.getBody());

  // As in stage coarsening, a stage that wraps a pipeline carries the units
  // of the stages inside.
  const auto createWrapper = [&](function_ref<ktdf::PipelineOp()> wrap) {
    auto stage =
        ktdf::StageOp::create(builder, loc, ValueRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(stage.getBody());
    const auto units = ktdf::collectPipelineApplicableUnits(wrap());
    stage.setApplicableUnitsAttr(
        builder.getArrayAttr(llvm::to_vector_of<Attribute>(units)));
  };
  createWrapper([&] {
    channel->moveBefore(builder.getInsertionBlock(),
                        builder.getInsertionPoint());
    return channel;
  });
  for (const SelfDelivery& delivery : deliveries) {
    createWrapper([&] { return buildLocalCopy(builder, loc, delivery); });
  }
}

struct HandleCrossCoreStagesPass
    : public scheduler::impl::HandleCrossCoreStagesPassBase<
          HandleCrossCoreStagesPass> {
  using HandleCrossCoreStagesPassBase::HandleCrossCoreStagesPassBase;

  void runOnOperation() override {
    ModuleOp module_op = getOperation();

    SmallVector<ktdf::PipelineOp> channels;
    module_op.walk([&](ktdf::PipelineOp pipeline) {
      ktdf::PrivateOp private_op = pipeline.getPrivateOp();
      if (private_op && llvm::any_of(private_op.getResults(), getFifoGroups)) {
        channels.push_back(pipeline);
      }
    });
    if (channels.empty()) {
      return;
    }

    auto& device_manager = getAnalysis<ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification for handling cross-core "
          "stages. This could happen if the device spec file is empty or "
          "contains multiple devices");
      signalPassFailure();
      return;
    }
    const auto& graph = device_manager.getOrCreateView<RoutingGraph>(*device);

    for (ktdf::PipelineOp channel : channels) {
      int grid_size = 0;
      SmallVector<SelfDelivery> deliveries;
      if (failed(extractGridSize(channel->getParentOfType<func::FuncOp>(),
                                 grid_size)) ||
          failed(planSelfDeliveries(channel, grid_size, graph, deliveries))) {
        signalPassFailure();
        return;
      }
      if (!deliveries.empty()) {
        wrapWithLocalCopies(channel, deliveries);
      }
    }
  }
};

}  // namespace
