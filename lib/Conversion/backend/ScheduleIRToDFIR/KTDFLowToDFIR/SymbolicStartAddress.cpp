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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/SymbolicStartAddress.h"

#include <limits>

#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "symbolic-start-address"

using namespace scheduler;

namespace {

/// The outermost module enclosing \p op. Splitting a kernel into per-schedule
/// programs puts each program's function in a module of its own beside the one
/// holding the kernel, so a program function and the call that reaches it are
/// only ever siblings under this module -- never in one symbol table.
mlir::ModuleOp topLevelModuleOf(mlir::Operation* op) {
  mlir::ModuleOp outermost;
  for (mlir::Operation* parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parent))
      outermost = module;
  }
  return outermost;
}

/// The one call to \p func under \p top, or null if there is no call or more
/// than one.
///
/// Matched on the callee's name rather than through a symbol table: the call
/// sits beside a private declaration of \p func in the caller's own module,
/// which is a different operation from \p func itself.
mlir::func::CallOp soleCallTo(mlir::ModuleOp top, mlir::func::FuncOp func) {
  if (!top) return {};
  mlir::func::CallOp found;
  bool ambiguous = false;
  top.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() != func.getName()) return mlir::WalkResult::advance();
    if (found) {
      ambiguous = true;
      return mlir::WalkResult::interrupt();
    }
    found = call;
    return mlir::WalkResult::advance();
  });
  return ambiguous ? mlir::func::CallOp() : found;
}

/// \p value as an argument of its function's entry block, or a null
/// BlockArgument if it is not one. A block argument of some inner region -- a
/// loop's induction variable, a program unit's unit iter_arg -- is not an
/// input.
mlir::BlockArgument asFunctionArgument(mlir::Value value) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg) return {};
  auto func =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func || func.getBody().empty() ||
      arg.getOwner() != &func.getBody().front())
    return {};
  return arg;
}

/// How many call edges an input is followed along before giving up. One is
/// enough for the pipeline as it stands -- kernel to program -- so this only
/// guards against a cycle in malformed IR.
constexpr unsigned kMaxCallDepth = 8;

/// Whether \p op is one whose definition can be read back out of the IR.
///
/// The set of operators that can be *written* is larger than the set that can
/// be read, and a symbol written outside the readable set is a hard failure in
/// whatever resolves it rather than a silent loss. So only these are used, and
/// anything else in an address expression is reported here.
bool readableAsDefinition(mlir::Operation* op) {
  return mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                   mlir::arith::MulIOp, mlir::arith::DivSIOp,
                   mlir::arith::RemSIOp, mlir::arith::MinSIOp,
                   mlir::arith::CeilDivSIOp>(op);
}

/// The per-unit constants of a `uniform.query_map` over a mapping whose values
/// are all constants, keyed by the unit each is for. std::nullopt when \p value
/// is not such a query.
///
/// This is what a `ktdp.get_compute_tile_id` has become by the time addresses
/// are lowered, and multiplying or offsetting one is how an address comes to
/// vary with the grid element.
std::optional<llvm::DenseMap<mlir::Value, int64_t>> perUnitConstantsOf(
    mlir::Value value) {
  auto query = value.getDefiningOp<mlir::uniform::QueryMapOp>();
  if (!query) return std::nullopt;
  auto mapping =
      query.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
  if (!mapping) return std::nullopt;

  llvm::DenseMap<mlir::Value, int64_t> per_unit;
  for (auto [key, mapped] : llvm::zip(mapping.getKeys(), mapping.getValues())) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(mapped);
    if (!constant) return std::nullopt;
    per_unit[key] = *constant;
  }
  return per_unit;
}

/// The units \p pu runs on, in the order it lists them -- which is the order a
/// uniform map over them has to be keyed in.
llvm::SmallVector<mlir::Value> unitsOf(mlir::dataflow::ProgramUnitOp pu) {
  return llvm::SmallVector<mlir::Value>(pu.getUnits().begin(),
                                        pu.getUnits().end());
}

/// What \p value works out to on \p unit, or std::nullopt where it is not
/// something that has a value there at all.
///
/// A leaf is a constant, worth the same everywhere, or a per-grid query, worth
/// what the mapping says for this unit; anything else is folded from its
/// operands by the operator it is. The operators are those a symbol's
/// definition can be written with, so that a displacement worked out here is
/// one the same expression could have said.
std::optional<int64_t> evaluateAtUnit(mlir::Value value, mlir::Value unit) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return constant;

  if (std::optional<llvm::DenseMap<mlir::Value, int64_t>> per_unit =
          perUnitConstantsOf(value)) {
    const auto found = per_unit->find(unit);
    if (found == per_unit->end()) return std::nullopt;
    return found->second;
  }

  mlir::Operation* op = value.getDefiningOp();
  if (!op || !readableAsDefinition(op) || op->getNumOperands() != 2)
    return std::nullopt;

  std::optional<int64_t> lhs = evaluateAtUnit(op->getOperand(0), unit);
  std::optional<int64_t> rhs = evaluateAtUnit(op->getOperand(1), unit);
  if (!lhs || !rhs) return std::nullopt;

  // Checked, and an overflow is std::nullopt like anything else that cannot be
  // worked out: what the caller does with that is report the offset as one no
  // symbol can be written for, which is the right answer for a number that does
  // not fit either.
  int64_t result = 0;
  if (mlir::isa<mlir::arith::AddIOp>(op))
    return llvm::AddOverflow(*lhs, *rhs, result) ? std::nullopt
                                                 : std::optional(result);
  if (mlir::isa<mlir::arith::SubIOp>(op))
    return llvm::SubOverflow(*lhs, *rhs, result) ? std::nullopt
                                                 : std::optional(result);
  if (mlir::isa<mlir::arith::MulIOp>(op))
    return llvm::MulOverflow(*lhs, *rhs, result) ? std::nullopt
                                                 : std::optional(result);
  if (mlir::isa<mlir::arith::MinSIOp>(op)) return std::min(*lhs, *rhs);
  // A division by zero is undefined where the program runs and is not something
  // to work out an address from here either.
  if (*rhs == 0) return std::nullopt;
  if (mlir::isa<mlir::arith::DivSIOp>(op)) return *lhs / *rhs;
  if (mlir::isa<mlir::arith::RemSIOp>(op)) return *lhs % *rhs;
  // Rounding up is only worked out for a positive numerator and denominator --
  // an offset into a tensor and a size -- rather than reproducing what the
  // operation does with a negative one.
  if (mlir::isa<mlir::arith::CeilDivSIOp>(op) && *lhs >= 0 && *rhs > 0)
    return (*lhs + *rhs - 1) / *rhs;
  return std::nullopt;
}

}  // namespace

//===----------------------------------------------------------------------===//
// SymbolAllocator
//===----------------------------------------------------------------------===//

SymbolAllocator::SymbolAllocator(mlir::ModuleOp top) {
  // The kernels: the functions with bodies that nothing calls. Their arguments
  // are what the run takes in.
  llvm::SmallVector<mlir::func::FuncOp> kernels;
  top.walk([&](mlir::func::FuncOp func) {
    if (!func.getBody().empty() && !soleCallTo(top, func))
      kernels.push_back(func);
    // Nothing nests a function in another, so the body holds no kernel to find.
    return mlir::WalkResult::skip();
  });

  for (mlir::func::FuncOp kernel : kernels) {
    mlir::OpBuilder builder(kernel.getContext());
    builder.setInsertionPointToStart(&kernel.getBody().front());
    for (mlir::BlockArgument arg : kernel.getArguments()) {
      // An argument of any other type is not an address and no symbol stands
      // for it. Numbering still counts the position, so that what an id is
      // follows from the signature rather than from which arguments happened to
      // be addresses.
      if (mlir::isa<mlir::IndexType>(arg.getType())) {
        mlir::symbol::CreateIdOp::create(builder, arg.getLoc(), arg,
                                         next_input_);
        LDBG(1) << "  " << kernel.getName() << " argument "
                << arg.getArgNumber() << " is symbol " << next_input_;
      }
      // Ids only ever descend, and a negative id is what marks one as a symbol,
      // so a wrap would hand out a positive number that means something else.
      // It takes 2^63 arguments to get here, but an id that is silently not a
      // symbol is not a failure anything downstream would catch.
      if (next_input_ == std::numeric_limits<int64_t>::min())
        llvm::report_fatal_error(
            "symbol ids exhausted numbering the run's "
            "inputs");
      --next_input_;
    }
  }
  next_derived_ = next_input_;
}

std::pair<int64_t, bool> SymbolAllocator::derivedIdFor(const DerivedKey& key) {
  const auto found = derived_.find(key);
  if (found != derived_.end()) return {found->second, false};
  if (next_derived_ == std::numeric_limits<int64_t>::min())
    llvm::report_fatal_error("symbol ids exhausted deriving addresses");
  const int64_t id = next_derived_--;
  derived_.insert({key, id});
  return {id, true};
}

void SymbolAllocator::declareInputsIn(mlir::func::FuncOp func,
                                      mlir::OpBuilder& builder) const {
  for (mlir::BlockArgument arg : func.getArguments()) {
    std::optional<int64_t> input = inputSymbolIdFor(arg);
    // A constant the caller had, or something it computed: not an input, so no
    // symbol of the run stands for it.
    if (!input) continue;
    mlir::symbol::CreateIdOp::create(builder, arg.getLoc(), arg, *input);
  }
}

std::optional<int64_t> SymbolAllocator::inputSymbolIdFor(
    mlir::Value address) const {
  mlir::BlockArgument arg = asFunctionArgument(address);
  if (!arg) return std::nullopt;

  mlir::ModuleOp top = topLevelModuleOf(arg.getOwner()->getParentOp());

  // Walk up the calls that pass this argument along. The loop ends at the
  // top-level function -- the kernel, whose signature is the run's own inputs.
  for (unsigned depth = 0; depth <= kMaxCallDepth; ++depth) {
    auto func = mlir::cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
    mlir::func::CallOp call = soleCallTo(top, func);
    if (!call) {
      if (!mlir::isa<mlir::IndexType>(arg.getType())) return std::nullopt;
      return -static_cast<int64_t>(arg.getArgNumber()) - 1;
    }

    mlir::Value arg_passed = call.getOperand(arg.getArgNumber());
    mlir::BlockArgument outer = asFunctionArgument(arg_passed);
    // A constant, or something the caller computed: arg_passed is not an input.
    if (!outer) return std::nullopt;
    arg = outer;
  }

  LDBG(1) << "  gave up tracing an address through " << kMaxCallDepth
          << " calls";
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Taking an address apart
//===----------------------------------------------------------------------===//

mlir::FailureOr<std::optional<SymbolicAddress>>
scheduler::takeSymbolicAddressApart(mlir::Value address,
                                    mlir::dataflow::ProgramUnitOp pu,
                                    const SymbolAllocator& symbols) {
  SymbolicAddress taken;

  // The set of values this address is computed from, walked first only to
  // answer whether a run input is among them. That has to be settled before
  // anything else in the expression can be called an error. An operator whose
  // definition cannot be read back is fine over constants, where the address is
  // just a constant, and fatal over an input; only the whole expression says
  // which of the two this is.
  llvm::DenseSet<mlir::Value> reached;
  llvm::SmallVector<mlir::Value> pending{address};
  bool rooted = false;
  while (!pending.empty()) {
    mlir::Value value = pending.pop_back_val();
    if (!reached.insert(value).second) continue;
    if (symbols.inputSymbolIdFor(value)) {
      rooted = true;
      break;
    }
    if (mlir::Operation* op = value.getDefiningOp())
      for (mlir::Value operand : op->getOperands()) pending.push_back(operand);
  }
  if (!rooted) return std::optional<SymbolicAddress>();

  // Then depth-first again, collecting the ops in the order they have to be
  // rebuilt and classifying every leaf. A leaf is the input the address is
  // rooted at, a constant, or a per-grid query -- and nothing else, because
  // nothing else has a value this can say per grid element.
  llvm::SetVector<mlir::Value> visited;
  llvm::SmallVector<mlir::Value> worklist{address};
  llvm::SmallVector<mlir::Operation*> ops;
  llvm::SmallVector<llvm::DenseMap<mlir::Value, int64_t>> per_unit_constants;

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visited.insert(value)) continue;

    if (std::optional<int64_t> input = symbols.inputSymbolIdFor(value)) {
      if (taken.root && taken.root != value) {
        return address.getDefiningOp()->emitError(
            "start address is computed from more than one run input, so no "
            "single symbol stands for it");
      }
      taken.input = *input;
      taken.root = value;
      continue;
    }

    // A constant contributes itself and nothing per grid element.
    if (mlir::getConstantIntValue(value)) continue;

    if (std::optional<llvm::DenseMap<mlir::Value, int64_t>> per_unit =
            perUnitConstantsOf(value)) {
      taken.per_unit_leaves.push_back(value);
      per_unit_constants.push_back(std::move(*per_unit));
      continue;
    }

    mlir::Operation* op = value.getDefiningOp();
    if (!op) {
      // A block argument that is not an input: a loop induction variable, or a
      // unit iter_arg. An address the run has to resolve cannot be built from
      // one -- there would be nothing to write down for it.
      return address.getDefiningOp()->emitError(
          "start address is computed from a run input and from a value that "
          "only exists while the program runs, so no symbol stands for it");
    }
    if (!readableAsDefinition(op)) {
      return op->emitError(
          "start address is computed from a run input by an operation whose "
          "definition cannot be written as a symbol expression");
    }
    ops.push_back(op);
    for (mlir::Value operand : op->getOperands()) worklist.push_back(operand);
  }

  // Roots-last, so rebuilding in order always has its operands already built.
  for (mlir::Operation* op : llvm::reverse(ops)) taken.expression.push_back(op);

  // What each unit's copy of the per-grid leaves is. Every leaf has to say
  // something for every unit the program runs on, or the map cannot be keyed.
  if (!taken.per_unit_leaves.empty()) {
    for (mlir::Value unit : unitsOf(pu)) {
      llvm::SmallVector<int64_t> terms;
      for (const auto& per_unit : per_unit_constants) {
        const auto found = per_unit.find(unit);
        if (found == per_unit.end()) {
          return pu.emitError(
                     "a per-grid term of a symbolic start address says nothing "
                     "for one of the units the program runs on, so the address "
                     "cannot be resolved there")
                 << unit;
        }
        terms.push_back(found->second);
      }
      taken.per_unit_terms.emplace_back(unit, std::move(terms));
    }
  }

  LDBG(1) << "  start address is symbol " << taken.input << " through "
          << taken.expression.size() << " op(s), "
          << taken.per_unit_terms.size() << " per-grid value(s)";
  return std::optional<SymbolicAddress>(std::move(taken));
}

//===----------------------------------------------------------------------===//
// Emitting it
//===----------------------------------------------------------------------===//

namespace {

/// Gets the key identifying the address \p taken, \p terms, \p displacement and
/// \p word_size describe, over the first \p through steps of the expression --
/// a step of it rather than the whole, where the step needs a symbol of its
/// own.
DerivedKey keyFor(const SymbolicAddress& taken, llvm::ArrayRef<int64_t> terms,
                  int64_t displacement, int64_t word_size, size_t through) {
  DerivedKey key;
  key.root = taken.root;
  key.offset = displacement;
  key.terms.assign(terms.begin(), terms.end());
  key.word_size = word_size;
  for (mlir::Operation* op :
       llvm::ArrayRef<mlir::Operation*>(taken.expression).take_front(through)) {
    llvm::SmallVector<std::optional<int64_t>, 2> operands;
    for (mlir::Value operand : op->getOperands())
      operands.push_back(mlir::getConstantIntValue(operand));
    key.expression.push_back({op->getName(), std::move(operands)});
  }
  return key;
}

/// Rebuilds \p taken's expression with each per-grid leaf replaced by the
/// constant \p terms gives it, adds \p displacement, declares the result as a
/// symbol of its own and returns that symbol's id.
///
/// Where the expression is empty and nothing is displaced the address is the
/// input itself, so there is nothing to derive and the input's own id is the
/// answer.
int64_t declareDerived(const SymbolicAddress& taken,
                       llvm::ArrayRef<int64_t> terms, int64_t displacement,
                       int64_t word_size, SymbolAllocator& symbols,
                       mlir::OpBuilder& builder, mlir::Location loc) {
  // Whether the expression comes back to the input itself is only known once it
  // has been folded, and the id is only known to be new after that, so some of
  // what is built here can turn out not to be wanted. Those are erased rather
  // than left dead in the IR: what erases the IR later has no reason to take
  // them in an order that keeps a use alive, and a dead operand chain is how
  // one ends up destroyed while something still points at it.
  llvm::SmallVector<mlir::Operation*> built;
  auto note = [&built](mlir::Value value) {
    if (mlir::Operation* op = value.getDefiningOp()) built.push_back(op);
    return value;
  };
  auto discardBuilt = [&built] {
    for (mlir::Operation* op : llvm::reverse(built))
      if (op->use_empty()) op->erase();
  };

  mlir::IRMapping mapping;
  for (auto [leaf, term] : llvm::zip(taken.per_unit_leaves, terms))
    mapping.map(leaf,
                note(mlir::arith::ConstantIndexOp::create(builder, loc, term)));

  // Folded as it is built, because with only one leaf left unknown most of the
  // expression is constant here -- the grid element whose share starts at the
  // tensor's own base adds nothing, and its address is the input itself.
  // Leaving that as `input + 0` and letting canonicalization fold it later
  // would leave a symbol declared on a value nothing computes, which reads back
  // as a symbol with no definition at all.
  mlir::Value result = taken.root;
  for (auto [index, op] : llvm::enumerate(taken.expression)) {
    mlir::Operation* clone = builder.clone(*op, mapping);
    llvm::SmallVector<mlir::Value> folded;
    if (mlir::succeeded(builder.tryFold(clone, folded)) && !folded.empty()) {
      mapping.map(op->getResult(0), folded.front());
      result = note(folded.front());
      continue;
    }
    result = note(clone->getResult(0));

    // A definition is one operator over symbols and constants, so a step read
    // by another step needs a symbol of its own to be read as an operand. The
    // last step needs none: what follows it reads it as part of a single
    // definition --
    // `(x + y) / word` -- or is the symbol below.
    const bool last = index + 1 == taken.expression.size();
    const bool read_by_what_follows =
        displacement == 0 &&
        (word_size == 1 ||
         mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp>(clone));
    if (last && read_by_what_follows) continue;

    // On this rebuild of the step even where the id is not fresh: two addresses
    // sharing a step share its id, but each rebuilds it, and what the next step
    // reads is this rebuild.
    const int64_t step = symbols
                             .derivedIdFor(keyFor(taken, terms,
                                                  /*displacement=*/0,
                                                  /*word_size=*/1, index + 1))
                             .first;
    built.push_back(mlir::symbol::CreateIdOp::create(builder, loc, result, step)
                        .getOperation());
  }
  if (displacement != 0) {
    mlir::Value by =
        note(mlir::arith::ConstantIndexOp::create(builder, loc, displacement));
    result = note(mlir::arith::AddIOp::create(builder, loc, result, by));
  }

  // `result` is the byte address now: the input, plus this grid element's
  // share, plus whatever displaced it. Where nothing was added to it, it is the
  // input itself and the input's own symbol stands for it.
  const bool derived_in_bytes = result != taken.root;

  if (word_size == 1) {
    if (!derived_in_bytes) {
      discardBuilt();
      return taken.input;
    }
    const auto [id, fresh] = symbols.derivedIdFor(
        keyFor(taken, terms, displacement, word_size, taken.expression.size()));
    // Declared once. A second declaration of the same id would leave two
    // symbols meaning one address, so what was built to get here is thrown away
    // instead.
    if (!fresh) {
      discardBuilt();
      return id;
    }
    mlir::symbol::CreateIdOp::create(builder, loc, result, id);
    return id;
  }

  // The unit addresses in words of `word_size` bytes and the run hands its
  // inputs over as byte addresses, so the symbol a program reads is the byte
  // address divided by that -- last, after everything else the address is built
  // from.
  const auto [id, fresh] = symbols.derivedIdFor(
      keyFor(taken, terms, displacement, word_size, taken.expression.size()));
  if (!fresh) {
    discardBuilt();
    return id;
  }

  // The byte address stays unnamed, and the definition is the division over the
  // expression that built it -- `(input + share) / word_size` rather than a
  // symbol for the sum and a division over that. Whatever reads definitions has
  // an operator for that shape, and a symbol nothing waits for would be one
  // more thing for the run to resolve for no reason.
  mlir::Value bytes_per_word =
      mlir::arith::ConstantIndexOp::create(builder, loc, word_size);
  mlir::Value in_words =
      mlir::arith::DivSIOp::create(builder, loc, result, bytes_per_word);
  mlir::symbol::CreateIdOp::create(builder, loc, in_words, id);
  return id;
}

}  // namespace

int64_t scheduler::Displacement::at(mlir::Value unit) const {
  for (const auto& [key, offset] : per_unit)
    if (key == unit) return offset;
  return common_offset;
}

void scheduler::Displacement::scaleBy(int64_t factor) {
  common_offset *= factor;
  for (auto& [unit, offset] : per_unit) offset *= factor;
}

mlir::FailureOr<scheduler::Displacement> scheduler::takeDisplacementApart(
    mlir::OpFoldResult offset, mlir::dataflow::ProgramUnitOp pu) {
  Displacement displaced;

  // A constant, whether it arrived as one or was folded into one: the same
  // wherever the program runs, and the whole answer.
  if (auto attr = mlir::dyn_cast<mlir::Attribute>(offset)) {
    displaced.common_offset = llvm::cast<mlir::IntegerAttr>(attr).getInt();
    return displaced;
  }
  auto value = llvm::cast<mlir::Value>(offset);
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value)) {
    displaced.common_offset = *constant;
    return displaced;
  }

  // Otherwise the offset has to be worth something on every unit the program
  // runs on -- an expression over the grid element's own tile id and constants.
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> per_unit;
  for (mlir::Value unit : unitsOf(pu)) {
    std::optional<int64_t> by = evaluateAtUnit(value, unit);
    if (!by) {
      mlir::Operation* site = value.getDefiningOp();
      return (site ? site->emitError() : pu.emitError())
             << "the start address is computed from a run input and displaced "
                "by a value only known while the program runs, so no symbol "
                "stands for it";
    }
    per_unit.emplace_back(unit, *by);
  }

  // Every unit came to the same number after all -- an access tile at a fixed
  // offset into the view, said the long way round. One number, and no map.
  const bool uniform = llvm::all_of(per_unit, [&](const auto& entry) {
    return entry.second == per_unit.front().second;
  });
  if (uniform) {
    displaced.common_offset = per_unit.front().second;
    return displaced;
  }

  displaced.per_unit = std::move(per_unit);
  LDBG(1) << "  displaced by a grid element's own offset, on "
          << displaced.per_unit.size() << " unit(s)";
  return displaced;
}

mlir::FailureOr<mlir::Value> scheduler::emitSymbolicStartAddress(
    const SymbolicAddress& taken, const Displacement& displacement,
    int64_t word_size, mlir::dataflow::ProgramUnitOp pu,
    SymbolAllocator& symbols, mlir::OpBuilder& builder,
    mlir::OpBuilder& definitions_at, mlir::Location loc) {
  // The same everywhere: one symbol, and no map to choose between copies of it.
  // Which is the common case -- a tensor read whole rather than a slab of it
  // per grid element -- and worth not building a map for.
  if (taken.per_unit_terms.empty() && !displacement.varies()) {
    const int64_t id = declareDerived(taken, {}, displacement.common_offset,
                                      word_size, symbols, definitions_at, loc);
    return createSymbolicStartAddress(builder, loc, id);
  }

  // The address differs per grid element, either because the expression it is
  // computed from does or because what displaces it does. Where only the
  // displacement differs the expression has no per-grid leaf and every unit
  // rebuilds the same one; the units are still what the map is keyed on.
  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t>>>
      per_unit_terms = taken.per_unit_terms;
  if (per_unit_terms.empty())
    for (mlir::Value unit : unitsOf(pu))
      per_unit_terms.emplace_back(unit, llvm::SmallVector<int64_t>{});

  // One per grid element, gathered into a map keyed on the unit and queried by
  // the iter_arg -- the same shape the units themselves are resolved through.
  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;
  for (const auto& [unit, terms] : per_unit_terms) {
    const int64_t id = declareDerived(taken, terms, displacement.at(unit),
                                      word_size, symbols, definitions_at, loc);
    keys.push_back(unit);
    values.push_back(createSymbolicStartAddress(builder, loc, id));
  }

  auto mapping = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  auto query = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), mapping.getResult(),
      pu.getRegion().front().getArgument(0));
  return query.getResult();
}

mlir::Value scheduler::createSymbolicStartAddress(mlir::OpBuilder& builder,
                                                  mlir::Location loc,
                                                  int64_t symbol_id) {
  auto symbol = mlir::symbol::CreateSymbolOp::create(builder, loc,
                                                     builder.getIndexType());
  // The id lives in an attribute rather than in the operation's own arguments;
  // `SymbolId` is the name symbol::CreateSymbolOp::getSymbolID reads.
  symbol->setAttr("SymbolId", builder.getI64IntegerAttr(symbol_id));
  return symbol.getResult();
}

//===----------------------------------------------------------------------===//
// Symbols for runtime scalars
//===----------------------------------------------------------------------===//

namespace {

/// Gets \p op as one step of an expression: its operator, and the constant each
/// of its operands carries.
ExpressionStep stepOf(mlir::Operation* op) {
  ExpressionStep step{op->getName(), {}};
  for (mlir::Value operand : op->getOperands())
    step.operands.push_back(mlir::getConstantIntValue(operand));
  return step;
}

/// Gets the value \p id is already declared on in \p entry, null if it is not.
mlir::Value declaredIn(mlir::Block& entry, int64_t id) {
  for (auto declaration : entry.getOps<mlir::symbol::CreateIdOp>())
    if (declaration.getSymbolId() == id) return declaration.getInput();
  return nullptr;
}

/// Collects into \p steps the operations computing \p value, operands before
/// results, and sets \p root to the input they start from.
///
/// Fails if there is no single input to start from, or if a step cannot be
/// written as a definition.
mlir::LogicalResult collectSteps(mlir::Value value,
                                 const SymbolAllocator& symbols,
                                 mlir::Value& root,
                                 llvm::SmallVectorImpl<mlir::Operation*>& steps,
                                 llvm::DenseSet<mlir::Operation*>& seen) {
  if (symbols.inputSymbolIdFor(value).has_value()) {
    if (root && root != value) return mlir::failure();
    root = value;
    return mlir::success();
  }

  mlir::Operation* op = value.getDefiningOp();
  if (!op) return mlir::failure();
  if (op->getNumResults() != 1 || op->getNumRegions() != 0 ||
      !mlir::isMemoryEffectFree(op)) {
    return mlir::failure();
  }
  if (!seen.insert(op).second) return mlir::success();

  for (mlir::Value operand : op->getOperands())
    if (mlir::failed(collectSteps(operand, symbols, root, steps, seen)))
      return mlir::failure();

  steps.push_back(op);
  return mlir::success();
}

}  // namespace

mlir::FailureOr<int64_t> scheduler::declareScalarSymbol(
    mlir::Value value, mlir::func::FuncOp program, SymbolAllocator& symbols) {
  mlir::Value root;
  llvm::SmallVector<mlir::Operation*> steps;
  llvm::DenseSet<mlir::Operation*> seen;
  if (mlir::failed(collectSteps(value, symbols, root, steps, seen)) || !root) {
    LDBG(1) << "  " << value << " is not an expression over one input";
    return mlir::failure();
  }

  int64_t id = *symbols.inputSymbolIdFor(root);
  if (steps.empty()) return id;

  // The arithmetic is rebuilt next to the program: the DataflowIR handed to
  // code generation must not read a value only a caller could supply.
  mlir::Block& entry = program.getBody().front();
  mlir::OpBuilder builder(program.getContext());
  builder.setInsertionPoint(entry.getTerminator());

  mlir::IRMapping mapping;
  DerivedKey key;
  key.root = root;
  for (mlir::Operation* step : steps) {
    mlir::Operation* clone = builder.clone(*step, mapping);
    if (clone->hasTrait<mlir::OpTrait::ConstantLike>()) continue;

    key.expression.push_back(stepOf(step));
    id = symbols.derivedIdFor(key).first;

    // One declaration per id, since two would say the symbol is two different
    // things.
    if (mlir::Value declared = declaredIn(entry, id)) {
      mapping.map(step->getResult(0), declared);
      clone->erase();
      continue;
    }
    mlir::symbol::CreateIdOp::create(builder, step->getLoc(),
                                     clone->getResult(0), id);
  }
  return id;
}
