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

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"

#include <queue>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace scheduler;
using namespace scheduler::arch_view;

MemoryTree::MemoryTree(const mlir::ktdf_arch::Device& device)
    : DeviceView(device) {
  initialize();
}

void MemoryTree::initialize() {
  // Precondition: tree must be empty
  assert(nodes_.empty() && "Tree must be empty before initialization");
  assert(resource_to_node_.empty() &&
         "Tree must be empty before initialization");

  // Build the tree by walking the device structure
  // Track memory values and their parent relationships
  llvm::DenseMap<mlir::Value, NodeId> value_to_node;
  llvm::DenseMap<mlir::Value, mlir::Value> child_to_parent;

  // Helper to process a region and track parent-child relationships
  std::function<void(mlir::Region&, std::optional<mlir::Value>)> processRegion;
  processRegion = [&](mlir::Region& region,
                      std::optional<mlir::Value> parent_mem) {
    for (auto& op : region.front().getOperations()) {
      if (auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(&op)) {
        // Create a node for this memory
        auto kind = mem_op.getKind();
        assert(kind && "Memory operation must have a kind attribute");

        std::optional<size_t> capacity;
        if (auto size = mem_op.getSize()) {
          capacity = size->getSExtValue();
        }
        // Bytes at the start of the memory the scheduler must not use.
        std::optional<size_t> reserved = mem_op.getFirstAvailableAddress();

        NodeId node_id = addNode(kind, capacity, reserved);
        value_to_node[mem_op.getResult()] = node_id;

        // Track parent relationship
        if (parent_mem) {
          child_to_parent[mem_op.getResult()] = *parent_mem;
        }
      } else if (auto group_op =
                     mlir::dyn_cast<mlir::ktdf_arch::GroupOp>(&op)) {
        // Determine which memory is shared in this group
        std::optional<mlir::Value> group_parent_mem = parent_mem;

        // Check if group shares any memory
        if (!group_op.getSharedMemory().empty()) {
          // Use the first shared memory as the parent for this group's scope
          group_parent_mem = group_op.getSharedMemory().front();
        }

        // Recursively process the group
        processRegion(group_op.getRegion(), group_parent_mem);
      }
    }
  };

  // Start processing from the device body
  processRegion(getDevice().getBodyRegion(), std::nullopt);

  // Now establish parent-child relationships in the tree
  for (auto [child_val, parent_val] : child_to_parent) {
    auto child_it = value_to_node.find(child_val);
    auto parent_it = value_to_node.find(parent_val);

    if (child_it != value_to_node.end() && parent_it != value_to_node.end()) {
      setParent(child_it->second, parent_it->second);
    }
  }

  // Populate aliases from import declaration if available
  if (auto mem_space_mapping = getDevice().getAttr("mem_space_mapping")) {
    populateAliases(mem_space_mapping);
  }
}

MemoryTree::NodeId MemoryTree::addNode(
    ResourceType memory_resource, std::optional<size_t> capacity_in_bytes,
    std::optional<size_t> reserved_in_bytes) {
  NodeId node_id = next_node_id_++;

  MemoryNode node;
  node.id = node_id;
  node.memory_resource = memory_resource;
  node.parent = std::nullopt;
  node.capacity_in_bytes = capacity_in_bytes;
  node.reserved_in_bytes = reserved_in_bytes;

  nodes_[node_id] = node;
  resource_to_node_[memory_resource] = node_id;

  return node_id;
}

void MemoryTree::setParent(NodeId child_id, NodeId parent_id) {
  assert(nodes_.count(child_id) && "Child node must exist");
  assert(nodes_.count(parent_id) && "Parent node must exist");

  nodes_[child_id].parent = parent_id;
  nodes_[parent_id].children.push_back(child_id);
}

std::string MemoryTree::stringifyResource(ResourceType resource) {
  std::string result;
  llvm::raw_string_ostream os(result);
  resource.print(os);
  return result;
}

std::optional<MemoryTree::MemoryNode> MemoryTree::getNode(
    NodeId node_id) const {
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) return std::nullopt;
  return it->second;
}

std::optional<MemoryTree::MemoryNode> MemoryTree::getNode(
    ResourceType memory_resource) const {
  auto it = resource_to_node_.find(memory_resource);
  if (it == resource_to_node_.end()) return std::nullopt;
  return getNode(it->second);
}

MemoryTree::MemoryNode MemoryTree::getRootOf(const MemoryNode& node) const {
  MemoryNode current = node;
  while (current.parent.has_value()) {
    auto parent_node = getNode(*current.parent);
    assert(parent_node.has_value() &&
           "MemoryTree node has parent ID with no corresponding MemoryNode");
    current = *parent_node;
  }
  return current;
}

std::optional<MemoryTree::NodeId> MemoryTree::getNodeIdForMemory(
    ResourceType memory_resource) const {
  // First check direct mapping (canonical resource)
  auto it = resource_to_node_.find(memory_resource);
  if (it != resource_to_node_.end()) {
    return it->second;
  }

  // Check alias mapping (only if aliases were configured)
  auto alias_it = alias_to_node_.find(memory_resource);
  if (alias_it != alias_to_node_.end()) {
    return alias_it->second;
  }

  // Not found
  return std::nullopt;
}

llvm::SmallVector<MemoryTree::NodeId> MemoryTree::getRootNodes() const {
  llvm::SmallVector<NodeId> roots;
  for (const auto& [node_id, node] : nodes_) {
    if (!node.parent.has_value()) {
      roots.push_back(node_id);
    }
  }
  return roots;
}

llvm::SmallVector<MemoryTree::NodeId> MemoryTree::getLeafNodes() const {
  llvm::SmallVector<NodeId> leaves;
  for (const auto& [node_id, node] : nodes_) {
    if (node.children.empty()) {
      leaves.push_back(node_id);
    }
  }
  return leaves;
}

llvm::SmallVector<MemoryTree::NodeId> MemoryTree::getPath(NodeId source,
                                                          NodeId target) const {
  llvm::SmallVector<NodeId> path;

  if (!nodes_.count(source) || !nodes_.count(target)) {
    return path;
  }

  // Find path from source up to common ancestor, then down to target
  // First, collect ancestors of both nodes
  llvm::DenseMap<NodeId, unsigned> source_ancestors;
  NodeId current = source;
  unsigned depth = 0;
  source_ancestors[current] = depth++;

  while (nodes_.at(current).parent.has_value()) {
    current = *nodes_.at(current).parent;
    source_ancestors[current] = depth++;
  }

  // Find common ancestor by walking up from target
  llvm::SmallVector<NodeId> target_to_ancestor;
  current = target;
  target_to_ancestor.push_back(current);

  while (nodes_.at(current).parent.has_value()) {
    current = *nodes_.at(current).parent;
    target_to_ancestor.push_back(current);

    if (source_ancestors.count(current)) {
      // Found common ancestor
      break;
    }
  }

  if (!source_ancestors.count(current)) {
    // No common ancestor found
    return path;
  }

  NodeId common_ancestor = current;

  // Build path: source -> ... -> common_ancestor -> ... -> target
  // First part: source to common ancestor
  current = source;
  path.push_back(current);
  while (current != common_ancestor) {
    current = *nodes_.at(current).parent;
    path.push_back(current);
  }

  // Second part: common ancestor to target (reverse the target_to_ancestor
  // path)
  for (auto it = target_to_ancestor.rbegin(); it != target_to_ancestor.rend();
       ++it) {
    if (*it == common_ancestor) continue;
    path.push_back(*it);
  }

  return path;
}

llvm::SmallVector<MemoryTree::NodeId> MemoryTree::getPath(
    ResourceType source, ResourceType target) const {
  auto source_id = getNodeIdForMemory(source);
  auto target_id = getNodeIdForMemory(target);

  if (!source_id || !target_id) {
    return llvm::SmallVector<NodeId>();
  }

  return getPath(*source_id, *target_id);
}

llvm::SmallVector<MemoryTree::NodeId> MemoryTree::getNodesAtDepth(
    unsigned depth) const {
  llvm::SmallVector<NodeId> result;

  for (const auto& [node_id, node] : nodes_) {
    auto node_depth = getNodeDepth(node_id);
    if (node_depth && *node_depth == depth) {
      result.push_back(node_id);
    }
  }

  return result;
}

std::optional<unsigned> MemoryTree::getNodeDepth(NodeId node_id) const {
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) return std::nullopt;

  unsigned depth = 0;
  NodeId current = node_id;

  while (nodes_.at(current).parent.has_value()) {
    current = *nodes_.at(current).parent;
    depth++;
  }

  return depth;
}

llvm::SmallVector<MemoryTree::NodeId> MemoryTree::getAllNodeIds() const {
  llvm::SmallVector<NodeId> result;
  for (const auto& [node_id, _] : nodes_) {
    result.push_back(node_id);
  }
  return result;
}

bool MemoryTree::isDescendantOf(NodeId descendant, NodeId ancestor) const {
  if (!nodes_.count(descendant) || !nodes_.count(ancestor)) {
    return false;
  }

  // Walk up from descendant to see if we reach ancestor
  auto current_node = getNode(descendant);
  while (current_node && current_node->parent.has_value()) {
    if (*current_node->parent == ancestor) {
      return true;
    }
    current_node = getNode(*current_node->parent);
  }

  return false;
}

bool MemoryTree::isDescendantOf(ResourceType descendant,
                                ResourceType ancestor) const {
  auto descendant_id = getNodeIdForMemory(descendant);
  auto ancestor_id = getNodeIdForMemory(ancestor);

  if (!descendant_id || !ancestor_id) {
    return false;
  }

  return isDescendantOf(*descendant_id, *ancestor_id);
}

void MemoryTree::printSubtree(llvm::raw_ostream& os, NodeId node_id,
                              unsigned indent_level) const {
  auto node = getNode(node_id);
  if (!node) return;

  // Print indentation
  for (unsigned i = 0; i < indent_level; ++i) {
    os << "  ";
  }

  // Print node info
  os << "- " << stringifyResource(node->memory_resource);

  // Print alias if present
  if (node->alias.has_value()) {
    os << " [alias: " << stringifyResource(*node->alias) << "]";
  }

  if (node->capacity_in_bytes) {
    os << " (capacity: " << *node->capacity_in_bytes << " bytes";
    if (node->reserved_in_bytes) {
      os << ", reserved: " << *node->reserved_in_bytes << " bytes";
    }
    os << ")";
  }

  os << "\n";

  // Print children recursively
  for (NodeId child_id : node->children) {
    printSubtree(os, child_id, indent_level + 1);
  }
}

void MemoryTree::print(llvm::raw_ostream& os) const {
  os << "Memory Hierarchy Tree:\n";

  auto roots = getRootNodes();
  if (roots.empty()) {
    os << "  (empty)\n";
    return;
  }

  for (NodeId root_id : roots) {
    printSubtree(os, root_id, 0);
  }
}

void MemoryTree::dump() const { print(llvm::dbgs()); }

std::optional<size_t> MemoryTree::getCapacity(
    ResourceType memory_resource) const {
  auto node = getNode(memory_resource);
  if (!node) return std::nullopt;
  return node->capacity_in_bytes;
}

std::optional<size_t> MemoryTree::getReserved(
    ResourceType memory_resource) const {
  auto node = getNode(memory_resource);
  if (!node) return std::nullopt;
  return node->reserved_in_bytes;
}

std::optional<size_t> MemoryTree::getAvailable(
    ResourceType memory_resource) const {
  auto node = getNode(memory_resource);
  if (!node || !node->capacity_in_bytes) return std::nullopt;

  size_t reserved = node->reserved_in_bytes.value_or(0);
  return *node->capacity_in_bytes - reserved;
}

llvm::SmallVector<ResourceType> MemoryTree::getMemoryResourcesWithCapacity()
    const {
  llvm::SmallVector<ResourceType> resources;
  for (const auto& [node_id, node] : nodes_) {
    if (node.capacity_in_bytes.has_value()) {
      resources.push_back(node.memory_resource);
    }
  }
  return resources;
}

// Made with Bob
void MemoryTree::populateAliases(mlir::Attribute mem_space_mapping_attr) {
  auto mem_space_mapping =
      mlir::dyn_cast<mlir::ktdf_arch::TypedMapAttr<ResourceType, ResourceType>>(
          mem_space_mapping_attr);
  if (!mem_space_mapping) {
    return;
  }

  // In the mapping: key = value
  // - value is the canonical resource (e.g., "global") - this is what's in the
  // tree
  // - key is the alias (e.g., #ktdp.memory_space<global>)
  for (auto [key, value] : mem_space_mapping.getEntries()) {
    // Find the node with the VALUE resource (the canonical one from the device
    // file)
    auto value_node_id = resource_to_node_.find(value);
    if (value_node_id != resource_to_node_.end()) {
      // Store key as the alias for this node
      nodes_[value_node_id->second].alias = key;

      // Add alias mapping so we can look up by alias
      alias_to_node_[key] = value_node_id->second;
    }
  }
}

bool MemoryTree::areAliases(ResourceType resource1,
                            ResourceType resource2) const {
  // Same resource is always an alias of itself
  if (resource1 == resource2) {
    return true;
  }

  // Check if both resources map to the same node
  auto node1_id = getNodeIdForMemory(resource1);
  auto node2_id = getNodeIdForMemory(resource2);

  if (!node1_id || !node2_id) {
    return false;
  }

  return *node1_id == *node2_id;
}

llvm::SmallVector<ResourceType> MemoryTree::getAliases(
    ResourceType resource) const {
  llvm::SmallVector<ResourceType> aliases;

  auto node_id = getNodeIdForMemory(resource);
  if (!node_id) {
    // Resource not found, return just the resource itself
    aliases.push_back(resource);
    return aliases;
  }

  auto node = getNode(*node_id);
  if (!node) {
    aliases.push_back(resource);
    return aliases;
  }

  // Add the canonical resource
  aliases.push_back(node->memory_resource);

  // Add the alias if present
  if (node->alias.has_value()) {
    aliases.push_back(*node->alias);
  }

  return aliases;
}

bool MemoryTree::isGlobalMemory(ResourceType memory_resource) const {
  auto roots = getRootNodes();
  if (roots.empty()) {
    return false;
  }

  // For each root node
  for (auto root_id : roots) {
    auto root_node = getNode(root_id);
    if (!root_node) {
      continue;
    }

    // Check if resource matches root's canonical resource
    if (root_node->memory_resource == memory_resource) {
      return true;
    }

    // Check if resource matches root's alias
    if (root_node->alias.has_value() && *root_node->alias == memory_resource) {
      return true;
    }
  }

  return false;
}

bool MemoryTree::isPerCoreScratchPadMemory(ResourceType memory_resource) const {
  auto roots = getRootNodes();
  if (roots.empty()) {
    return false;
  }

  // Get nodes at depth 1 (immediate children of roots)
  auto depth1_nodes = getNodesAtDepth(1);

  for (auto node_id : depth1_nodes) {
    auto node = getNode(node_id);
    if (!node) {
      continue;
    }

    // Check canonical resource
    if (node->memory_resource == memory_resource) {
      return true;
    }

    // Check alias
    if (node->alias.has_value() && *node->alias == memory_resource) {
      return true;
    }
  }

  return false;
}

bool MemoryTree::isBelowScratchPad(ResourceType memory_resource) const {
  // First find the scratchpad node(s) at depth 1
  auto depth1_nodes = getNodesAtDepth(1);

  // Get the node for the query resource
  auto resource_node_id = getNodeIdForMemory(memory_resource);
  if (!resource_node_id) {
    return false;
  }

  // Check if resource is a descendant of any scratchpad
  for (auto scratchpad_id : depth1_nodes) {
    if (isDescendantOf(*resource_node_id, scratchpad_id)) {
      return true;
    }
  }

  return false;
}
