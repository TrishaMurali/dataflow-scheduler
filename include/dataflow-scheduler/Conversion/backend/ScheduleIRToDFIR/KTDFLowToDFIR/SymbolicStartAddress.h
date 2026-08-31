//===------------------------------------------------------------*- c++ -*-===//
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
/// A tensor's start address can arrive as an input to the kernel rather than as
/// a constant: the host decides where the tensor lives, and the compiler is
/// told only that there is an address, not which one. In DataflowIR such an
/// address is a *symbol* -- `symbol.create_symbol` carrying the id that stands
/// for it -- and whatever runs the program writes the real value into every
/// instruction the symbol reached.
///
/// The symbol stands alone as the start address, so that what has to be written
/// over is a whole operand and code generation needs no form for an expression
/// over a placeholder. An address that is an expression over an input therefore
/// becomes a symbol of its own, and what it is computed from is said
/// separately, as arithmetic the IR carries alongside:
///
/// ```mlir
///   // -1 is the input; -3 is -1 + 24576.
///   symbol.create_id %arg0 {symbol_id = -1 : si64} : index
///   %0 = arith.addi %arg0, %c24576 : index
///   symbol.create_id %0 {symbol_id = -3 : si64} : index
///   %sym = symbol.create_symbol {SymbolId = -3 : i64} : index
///   %v = dataflow.get_logical_memory_view %unit, %sym {layout_map = ...}
/// ```
///
/// Nothing reads `%0`. It is there so that whatever resolves the symbols can
/// read what each one is from the IR rather than being told separately -- the
/// declarations *are* the table.
///
/// An address that varies with the grid element -- each one working on its own
/// slab of a tensor -- is one symbol per grid element rather than one overall.
/// Which slab that is can be said in either of two ways, and they come to the
/// same thing here: as arithmetic over the input at the view, so the address
/// the view is built on already differs; or as one view of the whole tensor
/// with an access tile whose offset into it is the grid element's, which is a
/// displacement of that view's address by the time it is lowered. Either way
/// the symbols are gathered into a uniform map keyed on the unit, so the start
/// address a program reads is the query, and each grid element's own symbol is
/// what gets written:
///
/// ```mlir
///   %m = uniform.def_immutable_mapping([%u0 -> %sym0], [%u1 -> %sym1]):index
///   %q = uniform.query_map(map:%m, key:%iter_arg) : index
///   %v = dataflow.get_logical_memory_view %unit, %q {layout_map = ...}
/// ```
///
/// In KTIR the address reaches `ktdp.construct_memory_view` as an expression
/// over a block argument: the kernel takes the argument in its own signature,
/// and splitting the kernel into per-schedule programs threads it down as an
/// argument of each program function that reads it.
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_

#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Hands out the symbol ids of one run and says which id each of its inputs is.
///
/// The run's inputs come first, one per argument of the functions nothing calls
/// -- the kernels -- and each is `-(position + 1)`. Everything derived from
/// them is numbered after, continuing downward in the order it is asked for.
///
/// Symbol ids are negative and are drawn from a single sequence: nothing sets a
/// range aside, so an id used for one thing cannot be used for another, and the
/// numbering has to be arrived at once for the whole run rather than per
/// function. That is why this is owned by the pass rather than by a walk.
///
/// Whatever resolves the symbols downstream has to arrive at the same numbering
/// for the inputs, and to leave everything this allocator handed out alone. The
/// declarations this writes into the IR are what say so: a consumer reads the
/// ids rather than working them out again.
/// One step of the expression a derived address is built by: which operation it
/// is, and which of its operands are constants.
///
/// Not the operation itself. Two program units running one tensor's address are
/// two copies of the same arithmetic, so an `Operation*` would make them two
/// addresses and hand out a symbol each -- and the copies are erased once a
/// symbol stands for them, leaving the key pointing at freed memory.
/// `OperationName` is interned by the context, so it compares and hashes as a
/// pointer and outlives the IR the key was built from.
struct ExpressionStep {
  mlir::OperationName op;
  /// One entry per operand, in operand order: the constant it carries, or
  /// std::nullopt where it carries none. In order because the order is part of
  /// the address -- `root - 5` and `5 - root` are two different ones.
  llvm::SmallVector<std::optional<int64_t>, 2> operands;

  bool operator==(const ExpressionStep& other) const {
    return op == other.op && operands == other.operands;
  }
};

/// So that a DerivedKey's expression can be hashed with the rest of it.
inline llvm::hash_code hash_value(const ExpressionStep& step) {
  llvm::hash_code hash = mlir::hash_value(step.op);
  for (const std::optional<int64_t>& operand : step.operands)
    hash = llvm::hash_combine(hash, operand.has_value(), operand.value_or(0));
  return hash;
}

/// What identifies a derived address: the input it is rooted at, the expression
/// over it, the offset added to it, the constants its per-unit operands take,
/// and the word size it is divided by. The word size is part of it because one
/// tensor read through units that address in different word sizes is two
/// different numbers.
struct DerivedKey {
  mlir::Value root;
  int64_t offset = 0;
  llvm::SmallVector<int64_t> terms;
  int64_t word_size = 1;
  /// The steps the address is built from the input by, outermost last. What
  /// tells two addresses apart that agree in root, offset, terms and word size.
  ///
  /// `(1024 + (tile * 6 + row) * 4096) * 2`, for a row the run hands in taken
  /// to a grid element's slab of a tensor the compiler placed at 1024, is:
  ///
  /// ```
  ///   arith.muli  [_, 6]      the grid element's share, in rows
  ///   arith.addi  [_, _]      displaced by the row the run hands in
  ///   arith.muli  [_, 4096]   rows to elements
  ///   arith.addi  [1024, _]   the address the compiler picked
  ///   arith.muli  [_, 2]      elements to bytes
  /// ```
  ///
  /// A second tensor read the same way differs in that 1024 and nothing else.
  llvm::SmallVector<ExpressionStep> expression;

  bool operator==(const DerivedKey& other) const {
    return root == other.root && offset == other.offset &&
           terms == other.terms && word_size == other.word_size &&
           expression == other.expression;
  }
};

class SymbolAllocator {
 public:
  /// Numbers the inputs of every kernel under \p top and declares each on the
  /// argument it arrives as, so that what an id means is said in the IR.
  /// Derived ids start after the last input's.
  ///
  /// A kernel is a function nothing under \p top calls. Its index-typed
  /// arguments are the run's inputs; an argument of any other type is not
  /// something a symbol can stand for and is passed over.
  explicit SymbolAllocator(mlir::ModuleOp top);

  /// Declares on \p func's own arguments the input each one arrives as.
  ///
  /// Splitting a kernel into programs hands each of them the subset of the
  /// inputs it reads, and a program has to say what its own arguments are for:
  /// what corrects it is written against what it declares, not against the
  /// kernel's signature. A kernel needs no call of this -- its arguments were
  /// declared when this was built.
  ///
  /// Call it once the program units exist. Before that a declaration counts as
  /// work to be done on a unit and is copied into every one of them.
  void declareInputsIn(mlir::func::FuncOp func, mlir::OpBuilder& builder) const;

  /// Returns the symbol id standing for \p address if it is a run input,
  /// std::nullopt if it is not.
  ///
  /// An input is a block argument of a function's entry block, traced back
  /// through the calls that pass it along until the kernel is reached. Its
  /// position in the *kernel's* signature is what identifies it: a program
  /// function takes only the inputs it reads, so two programs' first arguments
  /// are in general two different tensors.
  std::optional<int64_t> inputSymbolIdFor(mlir::Value address) const;

  /// Returns the id for the symbol \p key describes, and whether that id is
  /// newly handed out.
  ///
  /// The same address asked for twice gets the same id. One
  /// `ktdp.construct_memory_view` is copied into every unit the program runs
  /// on, so without this one tensor's address would become as many symbols as
  /// there are units, all meaning the same thing and each resolved separately.
  std::pair<int64_t, bool> derivedIdFor(const DerivedKey& key);

  /// How many inputs the run takes, i.e. how many ids are the inputs' own.
  int64_t numInputs() const { return -next_input_ - 1; }

 private:
  /// One past the last input id handed out, so `-(count + 1)`.
  int64_t next_input_ = -1;
  int64_t next_derived_ = -1;
  /// The id handed out for each derived address, keyed by the three things that
  /// identify one: the input it is rooted at, the offset added to it, and the
  /// constants its per-unit operands take. Nothing iterates it, so the order
  /// does not matter.
  llvm::DenseMap<DerivedKey, int64_t> derived_;
};

/// A start address that is an expression over a run input, taken apart: which
/// input it is rooted at, and what each grid element's own address is.
struct SymbolicAddress {
  /// The symbol id of the input the address is computed from.
  int64_t input = 0;
  /// The block argument that input arrives as. Every per-unit expression is
  /// rebuilt over this value.
  mlir::Value root;
  /// One entry per unit the address is read on: the unit, and the constant each
  /// of `per_unit_leaves` takes there, in the same order. Empty when the
  /// address does not vary with the grid element, which is the common case.
  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t>>>
      per_unit_terms;
  /// The operations of the expression, operands before results, so rebuilding
  /// them in order works. Empty when the address is the input itself.
  llvm::SmallVector<mlir::Operation*> expression;
  /// The operands of the expression whose value differs per unit -- each a
  /// `uniform.query_map` over per-unit constants. `per_unit_terms` gives what
  /// each is worth on each unit.
  llvm::SmallVector<mlir::Value> per_unit_leaves;
};

/// Takes \p address apart as an expression over a run input, or returns
/// std::nullopt if it is not one -- a constant, or rooted at something that is
/// not an input.
///
/// Fails, rather than returning std::nullopt, when the address *is* rooted at
/// an input but is not something a symbol can be written for: more than one
/// input in it, an operator whose definition could not be read back, or a
/// per-grid term that is not a query over constants. Reporting that here is
/// deliberate -- the alternative is a program that resolves to the wrong
/// address at runtime.
mlir::FailureOr<std::optional<SymbolicAddress>> takeSymbolicAddressApart(
    mlir::Value address, mlir::dataflow::ProgramUnitOp pu,
    const SymbolAllocator& symbols);

/// The offset of the tile a program reads into the view it is taken from. It is
/// part of the address the symbol stands for, so it has to be a number here.
///
/// It is one number when every grid element reads the same tile of the view,
/// and one number per unit when the view is the whole tensor and each element
/// reads its own slab of it.
struct Displacement {
  /// The offset every unit shares. When `per_unit` is empty this is the offset.
  int64_t common_offset = 0;
  /// The offset on each unit, in the order the program unit lists them. Empty
  /// when the offset does not vary with the grid element, which is the common
  /// case.
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> per_unit;

  /// Whether the offset differs between units.
  bool varies() const { return !per_unit.empty(); }

  /// The offset on \p unit.
  int64_t at(mlir::Value unit) const;

  /// Multiplies by \p factor, covering both `common_offset` and `per_unit`, to
  /// convert the unit the offsets are counted in.
  void scaleBy(int64_t factor);
};

/// Evaluates \p offset -- the offset of the tile a program reads into the view
/// it is taken from -- on each unit \p pu runs on.
///
/// A constant evaluates to itself on every unit. Otherwise \p offset has to be
/// an expression over constants and `uniform.query_map`s of per-unit constants,
/// which is what an offset computed from `ktdp.get_compute_tile_id` has become
/// by this point.
///
/// Fails when \p offset is neither, i.e. when it is only known while the
/// program runs. A symbol's definition has to be written down at compile time,
/// and such an offset cannot be.
mlir::FailureOr<Displacement> takeDisplacementApart(
    mlir::OpFoldResult offset, mlir::dataflow::ProgramUnitOp pu);

/// Returns the symbol id standing for \p value and declares in the entry block
/// of \p program what that symbol is.
///
/// A register filled with a runtime scalar uses a symbol the way a start
/// address does: the DataflowIR carries the id, and whatever resolves the
/// symbols writes the value in. The arithmetic computing the value is rebuilt
/// in the entry block
/// with one symbol per step, since a definition is a single operator over
/// symbols and constants.
///
/// If \p value is one of the run's inputs, that input's own symbol is returned
/// and nothing is declared.
///
/// Fails if \p value is not an expression over exactly one input, or if a step
/// of it cannot be written as a definition.
mlir::FailureOr<int64_t> declareScalarSymbol(mlir::Value value,
                                             mlir::func::FuncOp program,
                                             SymbolAllocator& symbols);

/// Emits the start address \p taken describes, displaced by \p displacement and
/// divided by \p word_size, at \p builder's insertion point, and returns what
/// the view should read.
///
/// \p word_size is the size in bytes of the word the unit reading the view
/// addresses in. The run hands its input addresses over as byte addresses, so
/// the symbol a program reads has to be the byte address divided by it -- the
/// division goes into the symbol's definition, after everything else, because
/// the operand the run writes over has to be the symbol alone.
///
/// One `symbol.create_symbol` where every grid element agrees, or one per
/// element gathered into a uniform map keyed on the unit -- which is what a
/// grid element's own displacement asks for just as much as an address that
/// already differs per element. The declaration saying what each of those
/// symbols is computed from goes at \p definitions_at, which has to be
/// somewhere \p taken's root is in scope and outside the region a single grid
/// element's program is read from.
mlir::FailureOr<mlir::Value> emitSymbolicStartAddress(
    const SymbolicAddress& taken, const Displacement& displacement,
    int64_t word_size, mlir::dataflow::ProgramUnitOp pu,
    SymbolAllocator& symbols, mlir::OpBuilder& builder,
    mlir::OpBuilder& definitions_at, mlir::Location loc);

/// Emits the `symbol.create_symbol` standing for symbol \p symbol_id at
/// \p builder's insertion point.
mlir::Value createSymbolicStartAddress(mlir::OpBuilder& builder,
                                       mlir::Location loc, int64_t symbol_id);

}  // namespace scheduler

namespace llvm {

/// So that SymbolAllocator can keep its derived ids in a DenseMap.
template <>
struct DenseMapInfo<scheduler::DerivedKey> {
  static scheduler::DerivedKey getEmptyKey() {
    return {DenseMapInfo<mlir::Value>::getEmptyKey(), 0, {}, 1, {}};
  }
  static scheduler::DerivedKey getTombstoneKey() {
    return {DenseMapInfo<mlir::Value>::getTombstoneKey(), 0, {}, 1, {}};
  }
  static unsigned getHashValue(const scheduler::DerivedKey& key) {
    return static_cast<unsigned>(
        hash_combine(key.root, key.offset, hash_combine_range(key.terms),
                     key.word_size, hash_combine_range(key.expression)));
  }
  static bool isEqual(const scheduler::DerivedKey& lhs,
                      const scheduler::DerivedKey& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_
