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
/// This pass identifies compute groups (sequences of operations between LoadOps
/// and StoreOps with the same access tile equivalence class) and extracts them
/// into separate functions in new modules.
///
/// Example transformation:
///
/// BEFORE:
/// -------
/// module {
///   func.func @main() {
///     scf.for %i = %c0 to %N step %c1 {
///       %tile1 = ktdp.construct_access_tile %A[%i, %c0]
///       %tile2 = ktdp.construct_access_tile %B[%i, %c0]
///       %tile3 = ktdp.construct_access_tile %C[%i, %c0]
///
///       %a = ktdp.load %tile1 : tensor<1x64xf16>
///       %b = ktdp.load %tile2 : tensor<1x64xf16>
///       %c_empty = tensor.empty() : tensor<1x64xf16>
///       %c = linalg.add ins(%a, %b : tensor<1x64xf16>, tensor<1x64xf16>)
///                       outs(%c_empty : tensor<1x64xf16>)
///       ktdp.store %c, %tile3
///     }
///   }
/// }
///
/// AFTER:
/// ------
/// module {
///   module {  // Original module
///     func.func @main() {
///       scf.for %i = %c0 to %N step %c1 {
///         func.call @local-schedule-0(%i) : (index) -> ()
///       }
///     }
///     func.func public @local-schedule-0(index)
///   }
///   module {  // Extracted function module
///     func.func public @local-schedule-0(%arg0: index) {
///       %tile1 = ktdp.construct_access_tile %A[%arg0, %c0]
///       %tile2 = ktdp.construct_access_tile %B[%arg0, %c0]
///       %tile3 = ktdp.construct_access_tile %C[%arg0, %c0]
///
///       %a = ktdp.load %tile1 : tensor<1x64xf16>
///       %b = ktdp.load %tile2 : tensor<1x64xf16>
///       %c_empty = tensor.empty() : tensor<1x64xf16>
///       %c = linalg.add ins(%a, %b : tensor<1x64xf16>, tensor<1x64xf16>)
///                       outs(%c_empty : tensor<1x64xf16>)
///       ktdp.store %c, %tile3
///       return
///     }
///   }
/// }
///
/// Key features:
/// - Identifies compute groups by access tile equivalence classes
/// - Extracts operations from first LoadOp to last StoreOp of each class
/// - Handles iter args (block arguments) by passing them as function parameters
/// - Recursively materializes dependencies (constants, memory views, etc.)
/// - Creates nested module structure with forward declarations
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#define PASS_NAME "compute-group-extraction"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_COMPUTEGROUPEXTRACTIONPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Compute Group Extraction pass"),
    llvm::cl::init(false));

}  // unnamed namespace

namespace {
struct ComputeGroupExtractionPass
    : public impl::ComputeGroupExtractionPassBase<ComputeGroupExtractionPass> {
  void runOnOperation() final;

 private:
  /// A compute group is defined as the set of operations between
  /// first_load and last_store, in addition to any ancestor operations.
  struct ComputeGroup {
    mlir::Operation* first_load;
    mlir::Operation* last_store;
  };

  void runOn(mlir::ModuleOp module_op);

  /// Walks the ancestor tree of a StoreOp's data operand and unions access
  /// tiles from any LoadOps found with the StoreOp's access tile.
  /// @param store_op The StoreOp whose ancestors to walk
  /// @param access_tile_eq_classes Equivalence classes to update
  void unionAccessTilesFromAncestors(
      mlir::ktdp::StoreOp store_op,
      llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes);

  /// Print equivalence classes for debugging.
  /// @param access_tile_eq_classes The equivalence classes to print
  void printEqClasses(
      llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes);

  /// Verifiy that a compute operation has valid usage patterns.
  /// @param op The operation to verify
  /// @return true if the operation has valid uses, false otherwise
  bool verifyComputeOpUses(mlir::Operation* op);

  /// Build equivalence classes of access tiles by processing StoreOps in a
  /// block.
  /// @param block The block to process
  /// @param access_tile_eq_classes Equivalence classes to populate
  void buildAccessTileEqClasses(
      mlir::Block* block,
      llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes);

  /// Extract a compute group into a new function in a child module.
  /// @param top_level_module The top-level module containing all child modules
  /// @param original_module The module containing the original function
  /// @param group The compute group to extract
  void extractComputeGroup(mlir::ModuleOp top_level_module,
                           mlir::ModuleOp original_module,
                           const ComputeGroup& group);

  /// Process a block to identify and collect compute groups for extraction.
  /// @param block The block to process
  /// @param access_tile_eq_classes Rquivalence classes of access tiles
  void processBlockForExtraction(
      mlir::Block* block,
      llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes);

  /// Collected compute groups
  llvm::SmallVector<ComputeGroup> groups_to_extract_;
};

void ComputeGroupExtractionPass::unionAccessTilesFromAncestors(
    mlir::ktdp::StoreOp store_op,
    llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes) {
  mlir::Value store_access_tile = store_op.getAccessTile();
  mlir::Block* store_block = store_op->getBlock();

  llvm::SmallVector<mlir::Operation*> worklist = {store_op};
  llvm::DenseSet<mlir::Operation*> visited;

  // Starting at the store, walk all the ancestors, searching for loads
  // and union the store's access tile equivalence class with the class
  // of the load's access tile. This captures the dependencies between
  // loads and stores.
  while (!worklist.empty()) {
    mlir::Operation* current_op = worklist.back();
    worklist.pop_back();

    if (visited.count(current_op)) continue;
    visited.insert(current_op);

    if (auto load_op = mlir::dyn_cast<mlir::ktdp::LoadOp>(current_op)) {
      mlir::Value load_access_tile = load_op.getAccessTile();
      assert(load_op->getBlock() == store_block &&
             "LoadOp expected to be in same block as StoreOp");
      access_tile_eq_classes.unionSets(store_access_tile, load_access_tile);
      continue;
    }

    for (mlir::Value operand : current_op->getOperands()) {
      // Skip if operand is a basic block argument.
      if (mlir::isa<mlir::BlockArgument>(operand)) continue;

      mlir::Operation* operand_def_op = operand.getDefiningOp();
      worklist.push_back(operand_def_op);
    }
  }
}

bool ComputeGroupExtractionPass::verifyComputeOpUses(mlir::Operation* op) {
  bool has_store_use = false;
  bool has_compute_use = false;

  for (mlir::Value result : op->getResults()) {
    for (mlir::Operation* user : result.getUsers()) {
      if (mlir::isa<mlir::ktdp::StoreOp>(user)) {
        has_store_use = true;
      } else if (llvm::isa<mlir::linalg::LinalgDialect>(user->getDialect()) ||
                 llvm::isa<mlir::math::MathDialect>(user->getDialect()) ||
                 llvm::isa<mlir::arith::ArithDialect>(user->getDialect())) {
        has_compute_use = true;
      }
    }
  }

  if (has_store_use && has_compute_use) return false;

  return true;
}

void ComputeGroupExtractionPass::printEqClasses(
    llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes) {
  for (llvm::EquivalenceClasses<mlir::Value>::iterator
           I = access_tile_eq_classes.begin(),
           E = access_tile_eq_classes.end();
       I != E; ++I) {
    if (!(*I)->isLeader()) continue;

    llvm::dbgs() << "  Class leader: " << (*I)->getData() << "\n";
    llvm::dbgs() << "    Members:\n";

    for (llvm::EquivalenceClasses<mlir::Value>::member_iterator MI =
             access_tile_eq_classes.member_begin(**I);
         MI != access_tile_eq_classes.member_end(); ++MI) {
      llvm::dbgs() << "    " << *MI << "\n";
    }
    llvm::dbgs() << "\n-------- END OF CLASS MEMBERS -------\n";
  }
}

void ComputeGroupExtractionPass::buildAccessTileEqClasses(
    mlir::Block* block,
    llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes) {
  for (mlir::Operation& block_op : block->getOperations()) {
    if (auto store_op = mlir::dyn_cast<mlir::ktdp::StoreOp>(&block_op)) {
      // For every store in block, union its access tile's class with the
      // classes of the access tiles of its ancestor loads.
      access_tile_eq_classes.insert(store_op.getAccessTile());
      unionAccessTilesFromAncestors(store_op, access_tile_eq_classes);
    } else if (llvm::isa<mlir::linalg::LinalgDialect>(block_op.getDialect()) ||
               llvm::isa<mlir::math::MathDialect>(block_op.getDialect()) ||
               llvm::isa<mlir::arith::ArithDialect>(block_op.getDialect())) {
      assert(verifyComputeOpUses(&block_op) &&
             "Compute operation must have valid usage pattern");
    }
  }
}

// Gets the values \p op reads and does not define: its operands, and whatever
// its regions read from outside them. A compute op can read a scalar in its
// body rather than through its operands, as the base and the stride of an
// address computation are read, and the group depends on that as much as on an
// operand.
static auto valuesReadFromAbove(mlir::Operation* op)
    -> llvm::SmallVector<mlir::Value> {
  // Ordered, because the order they are found in is the order they become
  // parameters of the extracted function in.
  llvm::SmallVector<mlir::Value> used;
  llvm::DenseSet<mlir::Value> seen;

  op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* child) {
    for (mlir::Value operand : child->getOperands()) {
      if (!operand.getParentRegion()->isAncestor(op->getParentRegion())) {
        continue;
      }
      if (seen.insert(operand).second) used.push_back(operand);
    }

    // Nothing inside an op isolated from above reads a value from out here.
    if (child->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });

  return used;
}

// Helper function to recursively collect iter args from what an op reads
static void collectArgsUsed(mlir::Value val,
                            llvm::DenseSet<mlir::Operation*>& visited,
                            llvm::SmallVectorImpl<mlir::Value>& args,
                            llvm::DenseSet<mlir::Value>& args_visited) {
  // If this is a block argument (e.g., iter arg from scf.for), add it to the
  // list of arguments to pass to the extracted function
  if (mlir::isa<mlir::BlockArgument>(val)) {
    // Avoid duplicates
    if (args_visited.insert(val).second) {
      args.push_back(val);
    }
    return;
  }

  mlir::Operation* op = val.getDefiningOp();

  // Avoid revisiting operations
  if (visited.count(op)) return;
  visited.insert(op);

  // Recursively collect from what it reads
  for (mlir::Value read : valuesReadFromAbove(op)) {
    collectArgsUsed(read, visited, args, args_visited);
  }
}

// Helper function to recursively materialize dependencies
static void materializeDependency(mlir::Value val, mlir::IRMapping& mapper,
                                  mlir::OpBuilder& builder) {
  // If this is a block argument, it should already be mapped
  if (mlir::isa<mlir::BlockArgument>(val)) {
    assert(mapper.contains(val) && "Block argument should be mapped");
    return;
  }

  // If we already cloned this ancestor, we're done
  if (mapper.contains(val)) return;

  mlir::Operation* op = val.getDefiningOp();

  // Ensure everything the ancestor reads is cloned first
  for (mlir::Value read : valuesReadFromAbove(op)) {
    materializeDependency(read, mapper, builder);
  }

  // Now clone the ancestor itself into the new block
  builder.clone(*op, mapper);
}

void ComputeGroupExtractionPass::extractComputeGroup(
    mlir::ModuleOp top_level_module, mlir::ModuleOp original_module,
    const ComputeGroup& group) {
  assert(group.first_load && group.last_store);

  mlir::OpBuilder builder(top_level_module);

  // Get the original function to copy its attributes
  auto original_func = group.first_load->getParentOfType<mlir::func::FuncOp>();
  assert(original_func && "Compute group must be inside a function");

  // Collect operations to move (from first_load to last_store inclusive)
  llvm::SmallVector<mlir::Operation*> ops_to_move;
  bool collecting = false;
  mlir::Block* group_block = group.first_load->getBlock();
  for (mlir::Operation& op : group_block->getOperations()) {
    if (&op == group.first_load) {
      collecting = true;
    }
    if (collecting) {
      ops_to_move.push_back(&op);
    }
    if (&op == group.last_store) {
      break;
    }
  }

  // Collect args (block arguments) needed by the extracted function
  llvm::SmallVector<mlir::Value> args;
  llvm::DenseSet<mlir::Operation*> visited;
  llvm::DenseSet<mlir::Value> args_visited;
  for (mlir::Operation* op : ops_to_move) {
    for (mlir::Value used : valuesReadFromAbove(op)) {
      collectArgsUsed(used, visited, args, args_visited);
    }
  }

  // Build function type with args as parameters
  llvm::SmallVector<mlir::Type> arg_types;
  for (mlir::Value arg : args) {
    arg_types.push_back(arg.getType());
  }
  auto func_type = builder.getFunctionType(arg_types, {});

  // Create a new child module in the top-level module, assigning it a symbol
  // name unique within the top-level module. The same name is reused for the
  // function inside, keeping module and function in a 1:1 correspondence.
  builder.setInsertionPointToEnd(top_level_module.getBody());
  auto extracted_module =
      mlir::ModuleOp::create(top_level_module.getLoc(), "local_schedule");
  builder.insert(extracted_module);
  mlir::SymbolTable top_level_symbol_table(top_level_module);
  (void)top_level_symbol_table.renameToUnique(extracted_module, {});
  llvm::StringRef func_name = extracted_module.getSymName().value();

  // Create forward declaration in the original module
  builder.setInsertionPointToEnd(original_module.getBody());
  auto forward_decl = mlir::func::FuncOp::create(top_level_module.getLoc(),
                                                 func_name, func_type);
  forward_decl.setPrivate();
  builder.insert(forward_decl);

  // Create the function definition in the extracted module
  builder.setInsertionPointToStart(extracted_module.getBody());
  auto func_op = mlir::func::FuncOp::create(top_level_module.getLoc(),
                                            func_name, func_type);
  func_op.setPublic();
  // Copy grid attribute from original function if it exists
  if (auto grid_attr = original_func->getAttr("grid")) {
    func_op->setAttr("grid", grid_attr);
  }
  extracted_module.push_back(func_op);

  // Create entry block with arguments
  mlir::Block* func_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(func_block);

  // Map args to function arguments so they can be used when cloning
  mlir::IRMapping mapper;
  for (size_t i = 0; i < args.size(); ++i) {
    mapper.map(args[i], func_block->getArgument(i));
  }

  // Step 1: Materialize all dependencies for the entire range.
  // (e.g. arith.constants, construct_memory_views, construct_access_tiles)
  for (mlir::Operation* op : ops_to_move) {
    for (mlir::Value used : valuesReadFromAbove(op)) {
      materializeDependency(used, mapper, builder);
    }
  }

  // Step 2: Clone the actual range
  for (mlir::Operation* op : ops_to_move) {
    // Skip if already cloned as dependency
    bool already_cloned = false;
    for (mlir::Value result : op->getResults()) {
      if (mapper.contains(result)) {
        already_cloned = true;
        break;
      }
    }

    if (!already_cloned) {
      builder.clone(*op, mapper);
    }
  }

  // Step 3: Replace the original operations with a call to the extracted
  // function, then erase them in reverse order to avoid dangling uses.
  // Insert the call where the first op was.
  builder.setInsertionPoint(ops_to_move.front());
  mlir::func::CallOp::create(builder, top_level_module.getLoc(), forward_decl,
                             args);

  for (auto it = ops_to_move.rbegin(); it != ops_to_move.rend(); ++it) {
    mlir::Operation* op = *it;
    assert(op->use_empty() && "Operation should have no uses left");
    op->erase();
  }

  // Add return operation to the new function
  builder.setInsertionPointToEnd(func_block);
  mlir::func::ReturnOp::create(builder, top_level_module.getLoc());

  LDBG(1) << "Extracted compute group into function: " << func_name;
  LDBG(1) << "Operations: " << ops_to_move.size();
  LDBG(1) << "Block arguments: " << args.size();
}

void ComputeGroupExtractionPass::processBlockForExtraction(
    mlir::Block* block,
    llvm::EquivalenceClasses<mlir::Value>& access_tile_eq_classes) {
  // Keep track of visited equivalence classes of access tiles for the current
  // block to assert they are not visited multiple times.
  llvm::DenseSet<mlir::Value> visited_class_leaders;
  mlir::Value current_class_leader = nullptr;
  bool last_mem_op_was_load = false;
  mlir::Operation* first_load_op = nullptr;
  mlir::Operation* last_store_op = nullptr;

  // Traverse operations in the block, extracting each compute group. Boundaries
  // are determined by changes in current equivalence class of load/store access
  // tiles and/or control flow operations.
  for (mlir::Operation& block_op : block->getOperations()) {
    // Check for control flow operations - they terminate the current compute
    // group.
    // Reason: Otherwise, if control flow operations were to be extracted,
    // we might end up with nested func calls.
    if (block_op.getNumRegions() > 0) {
      // Save current compute group if any
      if (current_class_leader && first_load_op && last_store_op) {
        groups_to_extract_.push_back({first_load_op, last_store_op});

        // Mark current class as visited
        visited_class_leaders.insert(current_class_leader);

        // Reset tracking
        current_class_leader = nullptr;
        first_load_op = nullptr;
        last_store_op = nullptr;
        last_mem_op_was_load = false;
      }
      continue;
    }

    if (auto load_op = mlir::dyn_cast<mlir::ktdp::LoadOp>(&block_op)) {
      mlir::Value load_access_tile = load_op.getAccessTile();
      assert(access_tile_eq_classes.findLeader(load_access_tile) !=
                 access_tile_eq_classes.member_end() &&
             "expecting access_tile to have equivalence class");
      mlir::Value load_class_leader =
          access_tile_eq_classes.getLeaderValue(load_access_tile);
      assert(visited_class_leaders.find(load_class_leader) ==
                 visited_class_leaders.end() &&
             "Equivalence class visited twice");

      // If this is a different class than current, save previous group and
      // switch
      if (current_class_leader && current_class_leader != load_class_leader) {
        assert(!last_mem_op_was_load &&
               "found consecutive loads in different equivalence classes");

        // Save the previous compute group
        if (first_load_op && last_store_op) {
          groups_to_extract_.push_back({first_load_op, last_store_op});
        }

        // Mark previous class as visited before switching
        visited_class_leaders.insert(current_class_leader);

        current_class_leader = load_class_leader;
        first_load_op = nullptr;
        last_store_op = nullptr;
      } else if (!current_class_leader) {
        // First load - initialize current class
        current_class_leader = load_class_leader;
      }

      // Track first load of current class
      if (!first_load_op) {
        first_load_op = &block_op;
      }

      last_mem_op_was_load = true;
    } else if (auto store_op = mlir::dyn_cast<mlir::ktdp::StoreOp>(&block_op)) {
      mlir::Value store_access_tile = store_op.getAccessTile();

      assert(access_tile_eq_classes.findLeader(store_access_tile) !=
                 access_tile_eq_classes.member_end() &&
             "expecting access_tile to have equivalence class");

      mlir::Value store_class_leader =
          access_tile_eq_classes.getLeaderValue(store_access_tile);

      assert(current_class_leader && "StoreOp found before any LoadOp");

      assert(current_class_leader == store_class_leader &&
             "StoreOp must belong to the same class as previous LoadOp");

      // Track last store of current class
      last_store_op = &block_op;
      last_mem_op_was_load = false;
    }
  }

  // Save the last compute group
  if (first_load_op && last_store_op) {
    groups_to_extract_.push_back({first_load_op, last_store_op});
  }
}

void ComputeGroupExtractionPass::runOn(mlir::ModuleOp module_op) {
  // Walk all blocks and collect compute groups to extract
  module_op.walk([&](mlir::Block* block) {
    llvm::EquivalenceClasses<mlir::Value> access_tile_eq_classes;

    buildAccessTileEqClasses(block, access_tile_eq_classes);

    if (access_tile_eq_classes.empty()) {
      return mlir::WalkResult::advance();
    }

    LDBG(1) << "Access tile equivalence classes:";
    LLVM_DEBUG(printEqClasses(access_tile_eq_classes));

    // Process block to collect compute groups
    processBlockForExtraction(block, access_tile_eq_classes);

    return mlir::WalkResult::advance();
  });

  // If no groups to extract, nothing to do
  if (groups_to_extract_.empty()) {
    return;
  }

  // Move original content to a child module and create child
  // modules for extracted functions. New structure:
  // module {
  //   module { // original
  //     func.func @orig {
  //       <original ops with extracted code replaced by func call>
  //     }
  //   }
  //   module { // extracted module #1
  //     ...
  //   }
  //   ... // other extracted modules
  // }
  mlir::OpBuilder builder(module_op);

  // Create a child module for the original content
  builder.setInsertionPointToStart(module_op.getBody());
  auto original_module = mlir::ModuleOp::create(builder, module_op.getLoc());

  // Move all operations from the original module to the new child module,
  // EXCEPT ktdf_arch.device declarations, which must stay as direct children of
  // the top-level module: device-dependent passes run once on the top module
  // and resolve the device via DeviceManager (which only sees devices directly
  // nested in the module it is built on), then walk into the nested modules to
  // do their work. Moving the device into a nested module hides it from those
  // passes.
  auto& original_ops = module_op.getBody()->getOperations();
  auto& child_ops = original_module.getBody()->getOperations();
  for (mlir::Operation& op : llvm::make_early_inc_range(llvm::make_range(
           std::next(original_module->getIterator()), original_ops.end()))) {
    if (mlir::isa<mlir::ktdf_arch::DeviceOp>(op)) continue;
    op.moveBefore(original_module.getBody(), child_ops.end());
  }

  // Now extract all collected compute groups, each into its own child module
  for (const auto& group : groups_to_extract_) {
    extractComputeGroup(module_op, original_module, group);
  }

  // Clean up side-effect-free ops left dead after extraction (e.g. ops that
  // were cloned into child modules but whose originals are now unused in the
  // top-level module). runRegionDCE recurses into nested regions, so passing
  // the module's regions covers child modules created above.
  mlir::IRRewriter rewriter(module_op.getContext());
  (void)mlir::runRegionDCE(rewriter, module_op->getRegions());
}

void ComputeGroupExtractionPass::runOnOperation() {
  if (DisableThisPass) return;
  LDBG(1) << "========= " PASS_NAME " =========";
  mlir::ModuleOp module_op = getOperation();
  runOn(module_op);
}

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createComputeGroupExtractionPass() {
  return std::make_unique<ComputeGroupExtractionPass>();
}
