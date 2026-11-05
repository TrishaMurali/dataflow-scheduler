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

#include "dataflow-scheduler/Analysis/MemoryTracker.h"

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "memory-tracker"

using namespace scheduler;

MemoryTracker::MemoryTracker(const arch_view::MemoryTree& memory_tree) {
  for (auto node_id : memory_tree.getAllNodeIds()) {
    auto node = memory_tree.getNode(node_id);
    if (node && node->capacity_in_bytes) {
      size_t reserved = node->reserved_in_bytes.value_or(0);
      capacities_[node->memory_resource] = *node->capacity_in_bytes;
      reserved_[node->memory_resource] = reserved;
      next_address_[node->memory_resource] = reserved;
    }
  }
}

std::string MemoryTracker::resourceToString(ResourceType memory_resource) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  if (memory_resource) {
    memory_resource.print(rso);
  } else {
    rso << "<null>";
  }
  return str;
}

size_t MemoryTracker::alignAddress(size_t address, size_t alignment) {
  if (alignment <= 1) return address;

  // Round up to next multiple of alignment
  size_t remainder = address % alignment;
  if (remainder == 0) return address;

  return address + (alignment - remainder);
}

llvm::Expected<size_t> MemoryTracker::allocate(ResourceType memory_resource,
                                               size_t size_in_bytes,
                                               size_t alignment) {
  assert(next_address_.find(memory_resource) != next_address_.end() &&
         "Memory resource not found in capacity map");
  // Get current address for this memory resource
  size_t current_address = next_address_[memory_resource];

  // Align the address
  size_t aligned_address = alignAddress(current_address, alignment);

  // Calculate the new next address after this allocation
  size_t new_next_address = aligned_address + size_in_bytes;

  // Query the available capacity
  auto capacity_it = capacities_.find(memory_resource);
  if (capacity_it == capacities_.end()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Memory resource %s does not have capacity information",
        resourceToString(memory_resource).c_str());
  }

  // Check if allocation fits
  if (new_next_address > capacity_it->second) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Allocation of %zu bytes (aligned to %zu) exceeds %s capacity "
        "(%zu bytes of capacity, %zu bytes already allocated)",
        size_in_bytes, alignment, resourceToString(memory_resource).c_str(),
        capacity_it->second, current_address - reserved_[memory_resource]);
  }

  // Update the next address
  next_address_[memory_resource] = new_next_address;

  size_t capacity = capacity_it->second;
  LDBG(1) << "[" DEBUG_TYPE "] allocate: resource="
          << resourceToString(memory_resource) << " requested=" << size_in_bytes
          << "B"
          << " aligned_addr=" << aligned_address << " used=" << new_next_address
          << "B"
          << " free=" << (capacity - new_next_address) << "B"
          << " capacity=" << capacity << "B";

  return aligned_address;
}

size_t MemoryTracker::getNextAvailableAddress(
    ResourceType memory_resource) const {
  auto it = next_address_.find(memory_resource);
  assert(it != next_address_.end() &&
         "Memory resource not found in next_address_ map");
  return it->second;
}

size_t MemoryTracker::getTotalAllocated(ResourceType memory_resource) const {
  auto reserved_it = reserved_.find(memory_resource);
  assert(reserved_it != reserved_.end() &&
         "Memory resource not found in reserved_ map");
  return getNextAvailableAddress(memory_resource) - reserved_it->second;
}

void MemoryTracker::reset(llvm::ArrayRef<ResourceType> memory_resources) {
  for (ResourceType resource : memory_resources) {
    auto it = next_address_.find(resource);
    if (it != next_address_.end()) {
      auto reserved_it = reserved_.find(resource);
      it->second = reserved_it != reserved_.end() ? reserved_it->second : 0;
    }
  }
}

// Made with Bob
