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

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"

#include <cassert>
#include <queue>
#include <sstream>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace scheduler::arch_view;

namespace {

// Helper struct to track state during device initialization
struct DeviceInitContext {
  RoutingGraph& graph;

  // Maps SSA values (from memory/exec_unit ops) to their corresponding NodeIds
  llvm::DenseMap<mlir::Value, RoutingGraph::NodeId> value_to_node_id;

  // Tracks which group kinds we've already processed (for flattening)
  llvm::DenseSet<mlir::Attribute> processed_group_kinds;

  // Collects memory node sizes to apply after traversal
  // Key: NodeId of memory node
  // Value: Size in bytes
  llvm::DenseMap<RoutingGraph::NodeId, size_t> node_sizes;

  // Maps each switch kind to the node that represents all its instances.
  llvm::DenseMap<mlir::Attribute, RoutingGraph::NodeId> switch_kind_nodes;

  // Every switch instance seen, with the node it was resolved to.
  llvm::SmallVector<std::pair<mlir::ktdf_arch::SwitchOp, RoutingGraph::NodeId>>
      switches;

  // Datapaths with a switch port on either end. They are resolved once the
  // whole switch fabric is known (see resolveSwitchDatapaths).
  llvm::SmallVector<mlir::ktdf_arch::DatapathOp> port_datapaths;

  explicit DeviceInitContext(RoutingGraph& g) : graph(g) {}
};

// Process a memory operation
void processMemoryOp(mlir::ktdf_arch::MemoryOp mem_op, DeviceInitContext& ctx) {
  auto kind = mem_op.getKind();
  assert(kind && "Memory operation must have a kind attribute");

  auto node_id =
      ctx.graph.addNode(kind, RoutingGraph::ResourceNode::ResourceKind::Memory);
  ctx.value_to_node_id[mem_op.getResult()] = node_id;

  // Extract size property if present - will be set in initializeFromDeviceOp
  if (auto size_attr = mem_op.getSizeAttr()) {
    if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(size_attr)) {
      ctx.node_sizes[node_id] = int_attr.getValue().getSExtValue();
    }
  }
}

// Process an execution unit operation
void processExecutionUnitOp(mlir::ktdf_arch::ExecutionUnitOp exec_op,
                            DeviceInitContext& ctx) {
  auto kind = exec_op.getKind();
  // Use a default kind if not specified
  if (!kind) {
    kind = mlir::StringAttr::get(exec_op.getContext(), "exec_unit");
  }

  // Load/store units become LoadStoreUnit nodes; other exec units become
  // Compute nodes.
  bool is_ls =
      exec_op.getFeature<mlir::ktdf_arch::feature::Load>() != nullptr ||
      exec_op.getFeature<mlir::ktdf_arch::feature::Store>() != nullptr;
  auto node_kind = is_ls
                       ? RoutingGraph::ResourceNode::ResourceKind::LoadStoreUnit
                       : RoutingGraph::ResourceNode::ResourceKind::Compute;
  auto node_id = ctx.graph.addNode(kind, node_kind);
  ctx.value_to_node_id[exec_op.getResult()] = node_id;
}

// Process a switch operation. Like groups, switches are deduplicated by kind:
// all instances of a kind share one node. A switch without a kind is its own
// class and gets its own node.
void processSwitchOp(mlir::ktdf_arch::SwitchOp switch_op,
                     DeviceInitContext& ctx) {
  constexpr auto node_kind = RoutingGraph::ResourceNode::ResourceKind::Switch;
  auto kind = switch_op.getKind();
  if (!kind) {
    auto node_id = ctx.graph.addNode(
        mlir::StringAttr::get(switch_op.getContext(), "switch"), node_kind);
    ctx.switches.emplace_back(switch_op, node_id);
    return;
  }

  auto it = ctx.switch_kind_nodes.find(kind);
  if (it == ctx.switch_kind_nodes.end()) {
    it = ctx.switch_kind_nodes
             .try_emplace(kind, ctx.graph.addNode(kind, node_kind))
             .first;
  }
  ctx.switches.emplace_back(switch_op, it->second);
}

bool isSwitchPort(mlir::Value value) {
  return mlir::isa<mlir::ktdf_arch::PortType>(value.getType());
}

// Forward declaration for mutual recursion
void processRegion(mlir::Region& region, DeviceInitContext& ctx);

// Process a group operation
void processGroupOp(mlir::ktdf_arch::GroupOp group_op, DeviceInitContext& ctx) {
  auto group_kind = group_op.getKind();

  // Flatten groups: only process first instance of each kind.
  // For skipped (deduplicated) groups, still add nodes for any yielded
  // exec_unit results so that datapaths referencing them can be resolved.
  if (group_kind && ctx.processed_group_kinds.contains(group_kind)) {
    for (auto result : group_op.getResults()) {
      auto node_id = ctx.graph.addNode(
          group_kind, RoutingGraph::ResourceNode::ResourceKind::Compute);
      ctx.value_to_node_id[result] = node_id;
    }
    return;
  }

  if (group_kind) {
    ctx.processed_group_kinds.insert(group_kind);
  }

  // Map shared memory block arguments to their operands
  // This allows datapaths inside the group to reference shared resources
  auto& block = group_op.getRegion().front();
  for (auto [operand, block_arg] :
       llvm::zip(group_op.getSharedMemory(), block.getArguments())) {
    auto it = ctx.value_to_node_id.find(operand);
    assert(it != ctx.value_to_node_id.end() &&
           "Shared memory operand must be defined before group");
    ctx.value_to_node_id[block_arg] = it->second;
  }

  // Recursively process the group's body
  processRegion(group_op.getRegion(), ctx);

  // Map yielded values to their definitions inside the group
  if (auto yield_op = mlir::dyn_cast<mlir::ktdf_arch::YieldOp>(
          group_op.getRegion().front().getTerminator())) {
    for (auto [result, operand] :
         llvm::zip(group_op.getResults(), yield_op.getOperands())) {
      // The yielded operand should already be mapped
      auto it = ctx.value_to_node_id.find(operand);
      assert(it != ctx.value_to_node_id.end() &&
             "Yielded value must be defined in group body");
      ctx.value_to_node_id[result] = it->second;
    }
  }
}

// Process a datapath operation — all resources (including LS units) are now
// first-class nodes, so every datapath creates a direct edge.
void processDatapathOp(mlir::ktdf_arch::DatapathOp datapath_op,
                       DeviceInitContext& ctx) {
  auto source_val = datapath_op.getSource();
  auto target_val = datapath_op.getTarget();

  if (isSwitchPort(source_val) || isSwitchPort(target_val)) {
    ctx.port_datapaths.push_back(datapath_op);
    return;
  }

  auto source_it = ctx.value_to_node_id.find(source_val);
  auto target_it = ctx.value_to_node_id.find(target_val);

  assert(source_it != ctx.value_to_node_id.end() &&
         "Datapath source must be a defined resource");
  assert(target_it != ctx.value_to_node_id.end() &&
         "Datapath target must be a defined resource");

  auto source_id = source_it->second;
  auto target_id = target_it->second;

  // Check for self-loops
  assert(source_id != target_id &&
         "Datapath cannot connect a resource to itself");

  ctx.graph.addEdge(source_id, target_id, 1);
}

// Process a region recursively
void processRegion(mlir::Region& region, DeviceInitContext& ctx) {
  for (mlir::Operation& op : region.front()) {
    if (auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(&op)) {
      processMemoryOp(mem_op, ctx);
    } else if (auto exec_op =
                   mlir::dyn_cast<mlir::ktdf_arch::ExecutionUnitOp>(&op)) {
      processExecutionUnitOp(exec_op, ctx);
    } else if (auto group_op = mlir::dyn_cast<mlir::ktdf_arch::GroupOp>(&op)) {
      processGroupOp(group_op, ctx);
    } else if (auto switch_op =
                   mlir::dyn_cast<mlir::ktdf_arch::SwitchOp>(&op)) {
      processSwitchOp(switch_op, ctx);
    } else if (auto datapath_op =
                   mlir::dyn_cast<mlir::ktdf_arch::DatapathOp>(&op)) {
      processDatapathOp(datapath_op, ctx);
    }
    // Ignore other operations (like YieldOp, which is handled in
    // processGroupOp)
  }
}

// Resolve the datapaths that touch switch ports into kind-level edges.
//
// Routing is first worked out on a port-level graph, whose nodes are the
// switch ports of every instance: a port reaches another port of the same
// switch if the switch's connectivity allows it, and a port of a different
// switch if a datapath links the two. Entries are datapaths from a resource
// into a port, exits are datapaths from a port to a resource. A kind-level
// edge is added only for datapaths that lie on some entry-to-exit route:
//
//   - resource -> port:  the port reaches some exit;
//   - port -> resource:  the port is reached from some entry;
//   - port -> port:      the source is reached from some entry and the target
//                        reaches some exit.
//
// Endpoints are mapped to the node of their switch's kind. A port -> port
// link between two instances of the same kind therefore becomes a self edge
// on that node. Duplicate edges (one per instance) are added once.
void resolveSwitchDatapaths(DeviceInitContext& ctx) {
  if (ctx.port_datapaths.empty()) return;

  llvm::DenseMap<mlir::Value, RoutingGraph::NodeId> port_to_node_id;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> successors;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> predecessors;
  auto addPortLink = [&](mlir::Value from, mlir::Value to) {
    successors[from].push_back(to);
    predecessors[to].push_back(from);
  };

  for (auto [switch_op, node_id] : ctx.switches) {
    auto connectivity = switch_op.getConnectivity();
    auto ports = switch_op.getResults();
    for (auto [source_index, source] : llvm::enumerate(ports)) {
      port_to_node_id[source] = node_id;
      for (auto [target_index, target] : llvm::enumerate(ports)) {
        if (connectivity.contains(source_index, target_index)) {
          addPortLink(source, target);
        }
      }
    }
  }

  llvm::SmallVector<mlir::Value> entry_ports, exit_ports;
  for (auto datapath_op : ctx.port_datapaths) {
    auto source = datapath_op.getSource();
    auto target = datapath_op.getTarget();
    bool source_is_port = isSwitchPort(source);
    bool target_is_port = isSwitchPort(target);
    if (source_is_port && target_is_port) {
      addPortLink(source, target);
    } else if (target_is_port) {
      entry_ports.push_back(target);
    } else {
      exit_ports.push_back(source);
    }
  }

  // Ports reachable from @p roots along @p links (roots included).
  auto reachable =
      [](llvm::ArrayRef<mlir::Value> roots,
         const llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>&
             links) {
        llvm::DenseSet<mlir::Value> seen(roots.begin(), roots.end());
        llvm::SmallVector<mlir::Value> worklist(roots.begin(), roots.end());
        while (!worklist.empty()) {
          mlir::Value port = worklist.pop_back_val();
          auto it = links.find(port);
          if (it == links.end()) continue;
          for (mlir::Value next : it->second) {
            if (seen.insert(next).second) worklist.push_back(next);
          }
        }
        return seen;
      };
  llvm::DenseSet<mlir::Value> from_entry = reachable(entry_ports, successors);
  llvm::DenseSet<mlir::Value> to_exit = reachable(exit_ports, predecessors);

  auto lookupNode = [&](mlir::Value value) {
    if (isSwitchPort(value)) {
      auto it = port_to_node_id.find(value);
      assert(it != port_to_node_id.end() &&
             "Datapath port must belong to a defined switch");
      return it->second;
    }
    auto it = ctx.value_to_node_id.find(value);
    assert(it != ctx.value_to_node_id.end() &&
           "Datapath endpoint must be a defined resource");
    return it->second;
  };

  llvm::SetVector<std::pair<RoutingGraph::NodeId, RoutingGraph::NodeId>> edges;
  for (auto datapath_op : ctx.port_datapaths) {
    auto source = datapath_op.getSource();
    auto target = datapath_op.getTarget();
    bool routed = (!isSwitchPort(source) || from_entry.contains(source)) &&
                  (!isSwitchPort(target) || to_exit.contains(target));
    if (routed) edges.insert({lookupNode(source), lookupNode(target)});
  }
  for (auto [source_id, target_id] : edges) {
    ctx.graph.addEdge(source_id, target_id, 1);
  }
}

}  // namespace

RoutingGraph::RoutingGraph(const mlir::ktdf_arch::Device& device)
    : DeviceView(device) {
  initialize();
}

// Initialize a RoutingGraph from the device held by this view.
//
// Walks the device IR and constructs the equivalent RoutingGraph
// representation. The graph must be empty before calling this function.
//
// Group operations are flattened by processing only the first instance of each
// group kind. This creates a representative view of the hierarchical structure
// where symmetric groups are deduplicated.
//
// Switches are deduplicated the same way, by kind; see
// resolveSwitchDatapaths for how datapaths on their ports become edges.
//
// Properties currently extracted:
//   - Memory nodes: kind (memory space), size (if specified)
//   - Execution unit nodes: kind (compute unit type)
//   - Switch nodes: kind; connectivity restricts which port datapaths become
//     edges
//   - Datapath edges: kind (transfer unit), source, target
//
// Properties currently skipped:
//   - features, bandwidth, overlaps (not yet modeled in RoutingGraph)
//
void RoutingGraph::initialize() {
  // Precondition: graph must be empty
  assert(nodes_.empty() && "Graph must be empty before initialization");
  assert(adjacency_.empty() && "Graph must be empty before initialization");

  // Create context for tracking state during initialization
  DeviceInitContext ctx(*this);

  // Process the device body
  processRegion(getDevice().getBodyRegion(), ctx);
  resolveSwitchDatapaths(ctx);

  // Apply node sizes collected during traversal
  for (const auto& [node_id, size] : ctx.node_sizes) {
    nodes_[node_id].total_capacity_in_bytes = size;
    nodes_[node_id].total_reserved_in_bytes = 0;
  }
}

RoutingGraph::NodeId RoutingGraph::addNode(ResourceType resource,
                                           ResourceNode::ResourceKind kind) {
  NodeId node_id = next_node_id_++;

  nodes_.insert(
      {node_id, {node_id, resource, kind, std::nullopt, std::nullopt}});
  resource_index_[resource].push_back(node_id);
  return node_id;
}

void RoutingGraph::addEdge(NodeId source, NodeId target, unsigned cost) {
  assert(hasNode(source) && "source node must exist before adding edge");
  assert(hasNode(target) && "target node must exist before adding edge");

  adjacency_[source].push_back({source, target, cost});
}

bool RoutingGraph::hasNode(NodeId node_id) const {
  return nodes_.count(node_id) != 0;
}

std::optional<RoutingGraph::ResourceNode> RoutingGraph::getNode(
    NodeId node_id) const {
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) return std::nullopt;
  return it->second;
}

RoutingGraph::NeighborList RoutingGraph::getNeighbors(NodeId node_id) const {
  NeighborList neighbors;
  auto it = adjacency_.find(node_id);
  if (it == adjacency_.end()) return neighbors;

  for (const EdgeInfo& edge : it->second) {
    neighbors.push_back(edge.target);
  }
  return neighbors;
}

std::optional<RoutingGraph::EdgeInfo> RoutingGraph::getEdgeInfo(
    NodeId source, NodeId target) const {
  auto it = adjacency_.find(source);
  if (it == adjacency_.end()) return std::nullopt;

  for (const EdgeInfo& edge : it->second) {
    if (edge.target == target) return edge;
  }
  return std::nullopt;
}

std::optional<RoutingGraph::Path> RoutingGraph::findShortestPath(
    NodeId source, NodeId target) const {
  if (!hasNode(source) || !hasNode(target)) return std::nullopt;

  std::queue<NodeId> worklist;
  llvm::DenseMap<NodeId, NodeId> predecessor;
  llvm::DenseMap<NodeId, bool> visited;

  worklist.push(source);
  visited[source] = true;

  while (!worklist.empty()) {
    NodeId current = worklist.front();
    worklist.pop();

    if (current == target) break;

    auto neighbors = getNeighbors(current);
    for (NodeId neighbor : neighbors) {
      if (visited.lookup(neighbor)) continue;
      visited[neighbor] = true;
      predecessor[neighbor] = current;
      worklist.push(neighbor);
    }
  }

  if (!visited.lookup(target)) return std::nullopt;

  Path reversed_path;
  NodeId current = target;
  reversed_path.push_back(current);

  while (current != source) {
    auto pred_it = predecessor.find(current);
    if (pred_it == predecessor.end()) return std::nullopt;
    current = pred_it->second;
    reversed_path.push_back(current);
  }

  Path path;
  path.reserve(reversed_path.size());
  for (auto it = reversed_path.rbegin(); it != reversed_path.rend(); ++it) {
    path.push_back(*it);
  }
  return path;
}

RoutingGraph::NodeId RoutingGraph::getNodeIdForResource(
    ResourceType resource) const {
  auto it = resource_index_.find(resource);
  assert(it != resource_index_.end() && "resource not found in graph");
  assert(it->second.size() == 1 &&
         "resource maps to multiple nodes; use getNodeIdsForResource");
  return it->second.front();
}

llvm::ArrayRef<RoutingGraph::NodeId> RoutingGraph::getNodeIdsForResource(
    ResourceType resource) const {
  auto it = resource_index_.find(resource);
  if (it == resource_index_.end()) return {};
  return it->second;
}

std::optional<RoutingGraph::EdgeInfo> RoutingGraph::getEdgeInfoForResources(
    ResourceType source, ResourceType dest) const {
  NodeId source_node = getNodeIdForResource(source);
  NodeId dest_node = getNodeIdForResource(dest);
  return getEdgeInfo(source_node, dest_node);
}

llvm::SmallVector<RoutingGraph::NodeId> RoutingGraph::getAllNodeIds() const {
  llvm::SmallVector<NodeId> node_ids;
  node_ids.reserve(nodes_.size());
  for (const auto& [node_id, _] : nodes_) {
    node_ids.push_back(node_id);
  }
  return node_ids;
}

void RoutingGraph::dump() const { print(llvm::dbgs()); }

void RoutingGraph::print(llvm::raw_ostream& os) const {
  os << "RoutingGraph {\n";
  os << "  Nodes:\n";
  for (const auto& [node_id, node] : nodes_) {
    os << "    - " << node_id << ": ";
    if (node.resource) {
      node.resource.print(os);
    } else {
      os << "<null>";
    }
    os << " [" << stringifyResourceKind(node.kind) << "]";

    // Print capacity and reserved info for Memory nodes
    if (node.kind == ResourceNode::ResourceKind::Memory) {
      bool has_info = false;
      if (node.total_capacity_in_bytes || node.total_reserved_in_bytes) {
        os << " (";
        if (node.total_capacity_in_bytes) {
          os << "capacity=" << *node.total_capacity_in_bytes << " bytes";
          has_info = true;
        }
        if (node.total_reserved_in_bytes) {
          if (has_info) os << ", ";
          os << "reserved=" << *node.total_reserved_in_bytes << " bytes";
        }
        os << ")";
      }
    }
    os << "\n";
  }

  os << "  Edges:\n";
  for (const auto& [node_id, node] : nodes_) {
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end()) continue;
    for (const EdgeInfo& edge : it->second) {
      auto source_node = getNode(edge.source);
      auto target_node = getNode(edge.target);
      assert(source_node && target_node &&
             "edge endpoint must reference a node");

      os << "    - " << edge.source << ": ";
      if (source_node->resource) {
        source_node->resource.print(os);
      } else {
        os << "<null>";
      }
      os << " -> " << edge.target << ": ";
      if (target_node->resource) {
        target_node->resource.print(os);
      } else {
        os << "<null>";
      }
      os << " [cost=" << edge.cost << "]\n";
    }
  }
  os << "}\n";
}

llvm::StringRef RoutingGraph::stringifyResourceKind(
    ResourceNode::ResourceKind kind) {
  switch (kind) {
    case ResourceNode::ResourceKind::Memory:
      return "Memory";
    case ResourceNode::ResourceKind::Compute:
      return "Compute";
    case ResourceNode::ResourceKind::LoadStoreUnit:
      return "LoadStoreUnit";
    case ResourceNode::ResourceKind::Switch:
      return "Switch";
  }
  llvm_unreachable("unknown RoutingGraph::ResourceNode::ResourceKind");
}

std::optional<RoutingGraph::ResourceNode> RoutingGraph::getNode(
    ResourceType resource) const {
  NodeId node_id = getNodeIdForResource(resource);
  return getNode(node_id);
}

std::optional<size_t> RoutingGraph::getResourceCapacity(
    ResourceType resource) const {
  auto node = getNode(resource);
  if (!node) return std::nullopt;
  return node->total_capacity_in_bytes;
}

std::optional<size_t> RoutingGraph::getResourceReserved(
    ResourceType resource) const {
  auto node = getNode(resource);
  if (!node) return std::nullopt;
  return node->total_reserved_in_bytes;
}

std::optional<size_t> RoutingGraph::getResourceAvailable(
    ResourceType resource) const {
  auto capacity = getResourceCapacity(resource);
  auto reserved = getResourceReserved(resource);

  if (!capacity || !reserved) return std::nullopt;

  return *capacity - *reserved;
}

// Made with Bob