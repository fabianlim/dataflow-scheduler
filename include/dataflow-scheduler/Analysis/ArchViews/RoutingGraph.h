//===-- RoutingGraph.h ------------------------------------------*- c++ -*-===//
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
// Routing Graph
//
// This file defines a reusable architecture-routing analysis abstraction used
// by path expansion. The graph models legal resource-to-resource routing hops.
// Load/store units are represented as first-class nodes (LoadStoreUnit kind).
//
// The graph is kind-level: symmetric groups are deduplicated by kind, and so
// are switches. A switch becomes a single Switch node per kind, and datapaths
// between switch ports are resolved to that node:
//
//   - The switch's port connectivity is honoured by resolving datapaths on a
//     port-level graph first: an edge into or out of a switch node, or between
//     two switch nodes, is only added if it lies on some route from a
//     non-switch resource through the switch fabric to another non-switch
//     resource. What the kind-level graph cannot express is which entry
//     reaches which exit: any edge into a switch node composes with any edge
//     out of it.
//   - Links between two switches of the same kind (e.g. between the stops of
//     a ring) collapse into a self edge on that kind's node. It records that
//     the fabric forwards between instances; shortest paths never take it, so
//     a path crosses each switch kind once, whatever the instance-level hop
//     count.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_ROUTINGGRAPH_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_ROUTINGGRAPH_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/AnalysisManager.h>

#include <optional>
#include <string>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

namespace scheduler {

namespace arch_view {

class RoutingGraph : public mlir::ktdf_arch::DeviceView {
 public:
  using NodeId = unsigned;

  struct ResourceNode {
    enum class ResourceKind {
      Memory,
      Compute,
      LoadStoreUnit,
      Switch,
    };

    NodeId id;
    ResourceType resource;
    ResourceKind kind;
    std::optional<size_t>
        total_capacity_in_bytes;  // Memory capacity (for memory resources)
    std::optional<size_t> total_reserved_in_bytes;  // Reserved/system memory
  };

  struct EdgeInfo {
    NodeId source;
    NodeId target;
    unsigned cost = 1;
  };

  using Path = llvm::SmallVector<NodeId>;
  using NeighborList = llvm::SmallVector<NodeId>;
  using EdgeList = llvm::SmallVector<EdgeInfo>;

  /// Construct a RoutingGraph for @p device .
  explicit RoutingGraph(const mlir::ktdf_arch::Device& device);

  NodeId addNode(ResourceType resource, ResourceNode::ResourceKind kind);
  void addEdge(NodeId source, NodeId target, unsigned cost = 1);

  bool hasNode(NodeId node_id) const;
  std::optional<ResourceNode> getNode(NodeId node_id) const;
  std::optional<ResourceNode> getNode(ResourceType resource) const;

  std::optional<EdgeInfo> getEdgeInfo(NodeId source, NodeId target) const;

  /// Get edge info between two resources (convenience wrapper)
  /// This is a convenience method that looks up node IDs and calls getEdgeInfo
  std::optional<EdgeInfo> getEdgeInfoForResources(ResourceType source,
                                                  ResourceType dest) const;

  std::optional<Path> findShortestPath(NodeId source, NodeId target) const;

  /// Get the node ID for a given resource attribute.
  /// Precondition: exactly one node carries \p resource. Use
  /// getNodeIdsForResource when a resource may map to multiple nodes
  /// (e.g. deduplicated group kinds).
  NodeId getNodeIdForResource(ResourceType resource) const;

  /// All node IDs whose resource attribute equals \p resource.
  /// Returns an empty span when the resource is not in the graph.
  llvm::ArrayRef<NodeId> getNodeIdsForResource(ResourceType resource) const;

  /// Get the total capacity in bytes for a resource (if applicable)
  std::optional<size_t> getResourceCapacity(ResourceType resource) const;

  /// Get the reserved capacity in bytes for a resource (if applicable)
  std::optional<size_t> getResourceReserved(ResourceType resource) const;

  /// Get the available capacity in bytes for a resource (capacity - reserved)
  std::optional<size_t> getResourceAvailable(ResourceType resource) const;

  /// Get all node IDs in the graph
  llvm::SmallVector<NodeId> getAllNodeIds() const;

  /// Get neighbors of a node
  NeighborList getNeighbors(NodeId node_id) const;

  void print(llvm::raw_ostream& os) const;
  void dump() const;

  static llvm::StringRef stringifyResourceKind(ResourceNode::ResourceKind kind);

 private:
  void initialize();

  NodeId next_node_id_ = 0;
  llvm::MapVector<NodeId, ResourceNode> nodes_;
  llvm::DenseMap<NodeId, EdgeList> adjacency_;
  llvm::DenseMap<ResourceType, llvm::SmallVector<NodeId>> resource_index_;
};

}  // namespace arch_view
}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_ROUTINGGRAPH_H_

// Made with Bob