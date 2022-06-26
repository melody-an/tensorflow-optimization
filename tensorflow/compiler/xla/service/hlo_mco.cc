/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_mco.h"

#include <set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
// #include "tensorflow/compiler/xla/service/hlo_instruction.cc"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

using DFSStack = absl::InlinedVector<std::pair<int, HloInstruction*>, 16>;
void DebugPrint(std::string functionName, std::string content) {
  std::cout << "[" << functionName << "]: " << content << std::endl;
}
void PrintCycle(const HloInstruction* child, DFSStack* dfs_stack) {
  // This set contains HloInstructions from the top of `DFSStack` that might
  // belong to the cycle, i.e. if  DFSStack :=[back,...,child,...,top], then
  // `subgraph` := {child,...,top}.
  absl::flat_hash_set<const HloInstruction*> subgraph;
  while (!dfs_stack->empty() && dfs_stack->back().second != child) {
    subgraph.insert(dfs_stack->back().second);
    dfs_stack->pop_back();
  }
  // Start dfs at `child` and find a cycle with all nodes in `subgraph`.
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 16> dfs;
  dfs.push_back(child);
  while (!dfs.empty()) {
    bool found_next_instr = false;
    for (const auto& user : dfs.back()->users()) {
      if (user == child) {
        dfs.push_back(child);
        LOG(INFO) << "\n\nDirected cycle:\n  "
                  << absl::StrJoin(
                         dfs, "\n  ",
                         [](std::string* out, const HloInstruction* instr) {
                           out->append(instr->name());
                         });
        return;
      }
      if (!subgraph.contains(user) || visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      dfs.push_back(user);
      found_next_instr = true;
    }
    if (!found_next_instr) {
      dfs.pop_back();
    }
  }
}

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

// perform a pre-order DFS to find an existing chain
template <typename Visitor>
static StatusOr<std::deque<HloInstruction*>> DetectMatrixChainPreorderDFS(
    HloInstruction* root, Visitor* visitor,
    bool ignore_control_predecessors = false) {
  // Calculating the instruction count within a module can be expensive on large
  // models so only do it if the visit state is empty. This will help when the
  // same visitor is reused across many computations of a single module.
  if (visitor->VisitStateCapacity() == 0) {
    visitor->ReserveVisitStates(root->GetModule()->instruction_count());
  }

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);
  // used to record roots of new copied chains
  std::deque<HloInstruction*> new_chain_roots;
  DebugPrint("DetectMatrixChainPreorderDFS",
             "Start, root = " + root->name() +
                 "opcode = " + HloOpcodeString(root->opcode()));

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0)
        << "[DetectMatrixChainPreorderDFS] " << current_id << ": "
        << current_node << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "[DetectMatrixChainPreorderDFS] "
              << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    dfs_stack.pop_back();
    VLOG(2) << "[DetectMatrixChainPreorderDFS] "
            << "Visiting HLO %" << current_node->name();
    DebugPrint("DetectMatrixChainPreorderDFS",
               "Visiting HLO = " + current_node->name() + " opcode = " +
                   HloOpcodeString(current_node->opcode()) + " is_matmul = " +
                   std::to_string(current_node->opcode() == HloOpcode::kDot));
    bool is_matmul_node = (current_node->opcode() == HloOpcode::kDot);
    visitor->Preprocess(current_node);
    visitor->SetVisitState(current_id, Visitor::kVisited);
    if (!is_matmul_node) {
      // for ohter op, we just target the its child nodes in the current branch
      // as a single operand
      continue;
    }
    DebugPrint(
        "DetectMatrixChainPreorderDFS",
        "current_node = " + current_node->name() +
            " opcode = " + HloOpcodeString(current_node->opcode()) +
            " users.size = " + std::to_string(current_node->users().size()));
    if (current_node != root && current_node->users().size() > 1) {
      // for reused sub matrix chain, need to copy
      // TODO: 在这里打印下原版和clone的name以及user数量等信息
      // DebugPrint(
      //     "DetectMatrixChainPreorderDFS",
      //     "need copy current_node = " + current_node->name() +
      //         " opcode = " + HloOpcodeString(current_node->opcode()) +
      //         " users.size = " +
      //         std::to_string(current_node->users().size()));
      // auto status_or = HloMCO::CopyResuableSubgraph(current_node);
      // HloInstruction* new_chain = std::move(status_or).ValueOrDie();
      // DebugPrint(
      //     "DetectMatrixChainPreorderDFS",
      //     "after copy current_node = " + current_node->name() +
      //         " opcode = " + HloOpcodeString(current_node->opcode()) +
      //         " users.size = " +
      //         std::to_string(current_node->users().size()));
      // DebugPrint(
      //     "DetectMatrixChainPreorderDFS",
      //     "after copy new_chain_root = " + new_chain->name() +
      //         " opcode = " + HloOpcodeString(new_chain->opcode()) +
      //         " users.size = " + std::to_string(new_chain->users().size()));
      // new_chain_roots.emplace_back(new_chain);
    }

    const size_t old_dfs_stack_size = dfs_stack.size();
    // current_node is a dot op, must have 2 operands
    CHECK_EQ(current_node->operands().size(), 2);
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "DetectMatrixChainPreorderDFS A cycle is detected while visiting "
            "instruction %s",
            current_node->ToString());
      }
      DebugPrint("DetectMatrixChainPreorderDFS",
                 "current_node = " + current_node->name() +
                     " opcode = " + HloOpcodeString(current_node->opcode()) +
                     " Add Child " + child->name() +
                     " opcode = " + HloOpcodeString(child->opcode()));
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return new_chain_roots;
}

}  // namespace

Status ParentsDetector::Preprocess(HloInstruction* hlo) {
  // add operand-parents relation of current node and its operands in to
  // global_operand_parent_map
  for (auto op : hlo->operands()) {
    if (global_operand_parent_map.contains(op)) {
      global_operand_parent_map[op].emplace_back(hlo);
    } else {
      global_operand_parent_map.insert({op, {hlo}});
    }
  }
  return Status::OK();
}

Status ChainRecorder::Preprocess(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kDot) {
    chain_map[chain_root].emplace_back(hlo);
    DebugPrint("ChainRecorder::Preprocess",
               "Add node: " + hlo->name() + " to root: " + chain_root->name() +
                   " chain_map[" + chain_root->name() +
                   "].size = " + std::to_string(chain_map[chain_root].size()));
  } else {
    DebugPrint("ChainRecorder::Preprocess",
               "Skip dot node: " + hlo->name() +
                   " opcode = " + HloOpcodeString(hlo->opcode()));
  }
  return Status::OK();
}
Status MatrixChainDetector::DetectMatrixChain(HloInstruction* chain_root) {
  // TOOD: 把这个地方封装一下
  std::deque<HloInstruction*> chain_roots{chain_root};
  while (!chain_roots.empty()) {
    HloInstruction* cur_root = chain_roots.front();
    chain_roots.pop_front();
    DebugPrint("MatrixChainDetector::MatrixChainDetector",
               "Find a new chain_root:" + cur_root->name());
    ChainRecorder chain_recorder(cur_root);
    auto status_or = DetectMatrixChainPreorderDFS(cur_root, &chain_recorder);
    std::deque<HloInstruction*> new_copied_chain_roots =
        std::move(status_or).ValueOrDie();
    DebugPrint("MatrixChainDetector::MatrixChainDetector",
               "new_copied_chain_roots.size() = " +
                   std::to_string(new_copied_chain_roots.size()));
    // also needs to optimize new copied matrix chains
    DebugPrint("MatrixChainDetector::MatrixChainDetector",
               "Before insert chain_roots.size() = " +
                   std::to_string(chain_roots.size()));
    chain_roots.insert(chain_roots.end(), new_copied_chain_roots.begin(),
                       new_copied_chain_roots.end());
    DebugPrint("MatrixChainDetector::MatrixChainDetector",
               "After insert chain_roots.size() = " +
                   std::to_string(chain_roots.size()));

    auto chain = chain_recorder.GetChain(cur_root);
    DebugPrint(
        "MatrixChainDetector::MatrixChainDetector",
        "Before insert chain_map.size() = " + std::to_string(chain_map.size()));
    chain_map.insert({cur_root, chain});
    DebugPrint(
        "MatrixChainDetector::MatrixChainDetector",
        "After insert chain_map.size() = " + std::to_string(chain_map.size()));
  }
  DebugPrint("MatrixChainDetector::MatrixChainDetector",
             "Finished chain_map.size() = " + std::to_string(chain_map.size()));
  return Status::OK();
}
Status MatrixChainDetector::Preprocess(HloInstruction* hlo) {
  DebugPrint("MatrixChainDetector::Preprocess",
             "inst.name:" + hlo->name() +
                 " opcode = " + HloOpcodeString(hlo->opcode()));
  // skip 2D dot op but if it is the root_instruction, then it must be the root
  // of a matrix chain
  if (hlo->opcode() == HloOpcode::kDot && hlo->shape().dimensions_size() == 2) {
    if (hlo != hlo->parent()->root_instruction()) {
      return Status::OK();
    } else {
      // 比如一种情况，root是dot，但是其中一个分支有一个operand不是这个chain了，
      // 但是这个operand的后代中又有chain.
      //
      // 刚刚细想了一下又觉得这么处理是可以的，因为MatrixChainDetector是postorder，所以当处理到root时
      // 其他子节点的chain一定已经被检测并优化了
      DebugPrint("MatrixChainDetector::Preprocess",
                 "root_instruction is dot, inst.name:" + hlo->name() +
                     " opcode = " + HloOpcodeString(hlo->opcode()));
      TF_RETURN_IF_ERROR(DetectMatrixChain(hlo));
      DebugPrint(
          "MatrixChainDetector::Preprocess",
          "After DetectMatrixChain(" + hlo->name() +
              ") chain_map.size() = " + std::to_string(chain_map.size()));
      return Status::OK();
    }
  }
  for (auto op : hlo->operands()) {
    DebugPrint("MatrixChainDetector::Preprocess",
               "operand.name:" + op->name() +
                   " opcode = " + HloOpcodeString(op->opcode()));
    if (op->opcode() == HloOpcode::kDot) {
      // Only consider 2D dot product for now
      const DotDimensionNumbers& dnums = op->dot_dimension_numbers();
      if (dnums.lhs_contracting_dimensions_size() != 1 ||
          dnums.rhs_contracting_dimensions_size() != 1 ||
          dnums.lhs_batch_dimensions_size() != 0 ||
          dnums.rhs_batch_dimensions_size() != 0 ||
          op->shape().dimensions_size() != 2) {
        VLOG(10)
            << "[MatrixChainDetector]: Can only optimize 2D, non-batch dot "
               "operations.";
        DebugPrint("MatrixChainDetector::Preprocess",
                   "Can only optimize 2D non-batch dot operations.");
        continue;
      }
      // current node != kDot, child op = kDot, child op is the root of a matrix
      // chain
      if (chain_map.contains(op)) {
        // find a reused chain which has already been added into chain_map
        continue;
      }
      TF_RETURN_IF_ERROR(DetectMatrixChain(op));
      DebugPrint(
          "MatrixChainDetector::Preprocess",
          "After DetectMatrixChain(" + hlo->name() +
              ") chain_map.size() = " + std::to_string(chain_map.size()));
    }
  }
  return Status::OK();
}

StatusOr<HloInstruction*> HloMCO::CopyResuableSubgraph(HloInstruction* inst) {
  std::vector<HloInstruction*> users;
  // TODO:
  // 并不能直接随便选取要替换的parent，而是应该替换掉除了当前分支parent之外的
  // 如果替换了当前分支的parent，那么就会造成有一个没有访问的parent仍然指向这个节点，但是应该是指向
  // clone的节点的，就会造成哪个
  users.assign(inst->users().begin() + 1, inst->users().end());
  HloInstruction* new_inst = inst->parent()->AddInstruction(inst->Clone());
  inst->ReplaceUsesWith(users, new_inst);
  return new_inst;
}

StatusOr<HloInstruction*> HloMCO::ConstructOptimalChain(
    HloInstruction* orig_root, std::vector<std::vector<int64>>& solution,
    std::vector<HloInstruction*>& chain_instructions) {
  DebugPrint("HloMCO::ConstructOptimalChain",
             "Start, root = " + orig_root->name());
  HloInstruction* optimal_root = nullptr;
  std::vector<HloInstruction*> subgraph_stack;
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions, 0, chain_instructions.size() - 1,
      subgraph_stack));
  DebugPrint("HloMCO::ConstructOptimalChain",
             "subgraph_stack.size = " + std::to_string(subgraph_stack.size()));
  CHECK_EQ(subgraph_stack.size(), 1);
  optimal_root = subgraph_stack.back();
  return optimal_root;
}

Status HloMCO::ConstructOptimalChainHelper(
    HloInstruction* orig_root, std::vector<std::vector<int64>>& solution,
    std::vector<HloInstruction*>& chain_instructions, int64 start_index,
    int64 end_index, std::vector<HloInstruction*>& subgraph_stack) {
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "Start, root = " + orig_root->name() + " (start,end) = (" +
          std::to_string(start_index) + "," + std::to_string(end_index) + ") " +
          "subgraph_stack.size = " + std::to_string(subgraph_stack.size()));
  auto create_dot = [&](HloInstruction* l, HloInstruction* r) {
    // DebugPrint();

    //     std::cout
    // << "[HloMCO::ConstructOptimalChainHelper::create_dot] "
    // << "Start: "
    // << " operand1 = " << l->name() << " shape = " << l->shape().ToString()
    // << " operand2 = " << r->name() + " shape = " << r->shape().ToString()
    // << std::endl;
    std::string temp_string = "Start:  operand1 = " + l->name() +
                              " shape = " + l->shape().ToString() +
                              " operand2 = " + r->name() +
                              " shape = " + r->shape().ToString();
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot", temp_string);
    const Shape lhs_shape = l->shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    auto status_or = ShapeInference::InferDotOpShape(l->shape(), r->shape(),
                                                     dimension_numbers);
    Shape output_shape = std::move(status_or).ValueOrDie();

    // TF_ASSIGN_OR_RETURN(
    //     auto output_shape,
    //     ShapeInference::InferDotOpShape(
    //         chain_instructions[start_index]->shape(),
    //         chain_instructions[end_index]->shape(), dimension_numbers));
    temp_string = "InferDotOpShape:  operand1 = " + l->name() +
                  " shape = " + l->shape().ToString() +
                  " operand2 = " + r->name() +
                  " shape = " + r->shape().ToString() +
                  " inferred_output_shape = " + output_shape.ToString();
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot", temp_string);

    // auto new_matmul_inst_ptr = HloInstruction::CreateDot(
    //     output_shape, l, r, dimension_numbers, l->precision_config());
    // TODO:
    // 感觉bug的原因可能是因为这个CreateDot产生的是个临时变量？然后后续返回后就失效了
    // 所以是不是应该把subgraph_stack改成存stack
    // for newly created instruction, we need to save it to the computation
    HloInstruction* new_matmul_inst_ptr = l->parent()->AddInstruction(
        HloInstruction::CreateDot(output_shape, l, r, dimension_numbers,
                                  orig_root->precision_config()));
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot",
               "create new matmul: " + new_matmul_inst_ptr->name() +
                   "shape = " + new_matmul_inst_ptr->shape().ToString() +
                   " operand1 = " + l->name() + " operand2 = " + r->name());
    subgraph_stack.emplace_back(new_matmul_inst_ptr);
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot",
               "After insert subgraph_stack.size= " +
                   std::to_string(subgraph_stack.size()) +
                   " top = " + subgraph_stack.back()->name() +
                   " shape = " + subgraph_stack.back()->shape().ToString());
  };

  if (start_index == end_index) {
    DebugPrint(
        "HloMCO::ConstructOptimalChainHelper",
        "Add single operand: " + chain_instructions[start_index]->name());
    // for single operand, it has already been stored in the compoutation
    subgraph_stack.emplace_back(chain_instructions[start_index]);
    DebugPrint("HloMCO::ConstructOptimalChainHelper",
               "After insert single operand subgraph_stack.size= " +
                   std::to_string(subgraph_stack.size()) +
                   " top = " + subgraph_stack.back()->name());
    return Status::OK();
  }

  if (start_index == end_index - 1) {
    // construction a new matmul op
    DebugPrint("HloMCO::ConstructOptimalChainHelper",
               "Add single matmul: " + chain_instructions[start_index]->name() +
                   " * " + chain_instructions[end_index]->name());
    create_dot(chain_instructions[start_index], chain_instructions[end_index]);
    DebugPrint("HloMCO::ConstructOptimalChainHelper",
               "After insert single matmul subgraph_stack.size= " +
                   std::to_string(subgraph_stack.size()) +
                   " top = " + subgraph_stack.back()->name());
    return Status::OK();
  }
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "First interval = [" + std::to_string(start_index) + "," +
                 std::to_string(solution[start_index][end_index]) + "] " +
                 "Second interval = [" +
                 std::to_string(solution[start_index][end_index] + 1) + "," +
                 std::to_string(end_index) + "]");
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions, start_index,
      solution[start_index][end_index], subgraph_stack));
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions,
      solution[start_index][end_index] + 1, end_index, subgraph_stack));

  // since this is a stack, the right_operand is on the top of left_operand
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "Before combile subgraph_stack.size= " +
                 std::to_string(subgraph_stack.size()) +
                 " top = " + subgraph_stack.back()->name());
  HloInstruction* right_operand = subgraph_stack.back();
  subgraph_stack.pop_back();
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "combile right_operand =" + right_operand->name() +
          " subgraph_stack.size= " + std::to_string(subgraph_stack.size()));
  HloInstruction* left_operand = subgraph_stack.back();
  subgraph_stack.pop_back();
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "combile left_operand =" + left_operand->name() +
          " subgraph_stack.size= " + std::to_string(subgraph_stack.size()));
  create_dot(left_operand, right_operand);
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "After combile subgraph_stack.size= " +
                 std::to_string(subgraph_stack.size()) +
                 " top = " + subgraph_stack.back()->name());

  return Status::OK();
}

StatusOr<HloInstruction*> HloMCO::ComputeOptimalChainOrder(
    HloInstruction* root, std::vector<HloInstruction*>& chain) {
  DebugPrint("HloMCO::ComputeOptimalChainOrder",
             "chain_root = " + root->name() +
                 " chain_length = " + std::to_string(chain.size()));
  HloInstruction* optimal_root = nullptr;
  int64 chain_length = chain.size();
  // sizes[i] stores the number of rows of operand[i]
  // sizes[i+1] stores the number of columns of operand[i]
  std::vector<int64> sizes(chain_length + 1, 0);
  for (auto i = 0; i < chain_length; ++i) {
    CHECK_LE(chain[i]->shape().rank(), 2);
    if (chain[i]->shape().rank() == 1) {
      // vector operand
      // TODO: a vector in XLA is row vector or column vector? For now consider
      // it as column vector
      sizes[i] = chain[i]->shape().dimensions(0);
      sizes[i + 1] = 1;
    } else if (chain[i]->shape().rank() == 2) {
      // matrix operand
      sizes[i] = chain[i]->shape().dimensions(0);
      sizes[i + 1] = chain[i]->shape().dimensions(1);
    }
  }
  // solution[i][j] stores optimal break point in
  // subexpression from i to j.
  std::vector<std::vector<int64>> solution(chain_length,
                                           std::vector<int64>(chain_length, 0));
  /* costs[i,j] = Minimum number of scalar multiplications
        needed to compute the matrix A[i]A[i+1]...A[j] =
        A[i..j] */
  std::vector<std::vector<int64>> costs(
      chain_length,
      std::vector<int64>(chain_length, std::numeric_limits<int64>::max()));
  // cost is zero when multiplying one matrix.
  for (int64 i = 0; i < chain_length; i++) costs[i][i] = 0;

  // L is chain length.
  // Dynamic Programming to find the optimal computing order
  for (int64 L = 2; L <= chain_length; L++) {
    for (int64 i = 0; i <= chain_length - L; i++) {
      // L = 2:             L = n:
      // i = 0 -> n-2       i = 0 -> 0
      // j = 1 -> n-1       j = n-1 -> n-1
      int64 j = i + L - 1;
      // [i,j] is the [start,end] index of the current subchain
      for (int64 k = i; k <= j - 1; k++) {
        // compute
        int64 cost = costs[i][k] + costs[k + 1][j] +
                     sizes[i] * sizes[k + 1] * sizes[j + 1];
        if (cost < costs[i][j]) {
          costs[i][j] = cost;
          // Each entry solution[i,j]=k shows
          // where to split the product arr
          // i,i+1....j to [i,k] * [k+1,j] for the minimum cost.
          solution[i][j] = k;
        }
      }
    }
  }
  auto status_or = ConstructOptimalChain(root, solution, chain);
  optimal_root = std::move(status_or).ValueOrDie();
  return optimal_root;
}

StatusOr<bool> HloMCO::ChainOptimize(
    HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
        chain_map) {
  DebugPrint("HloMCO::ChainOptimize", "Start");
  bool changed = false;
  for (auto& item : chain_map) {
    DebugPrint("HloMCO::ChainOptimize",
               "optimize chain_root = " + item.first->name());
    auto status_or = ComputeOptimalChainOrder(item.first, item.second);
    HloInstruction* new_instruction = std::move(status_or).ValueOrDie();
    DebugPrint(
        "HloMCO::ChainOptimize",
        "Finish optimization, new_chain_root = " + new_instruction->name());
    DebugPrint("HloMCO::ChainOptimize",
               "Before replace, chain_root.user.size = " +
                   std::to_string(item.first->users().size()));

    item.first->ReplaceAllUsesWith(new_instruction);
    if (new_instruction == item.first->parent()->root_instruction()) {
      DebugPrint("HloMCO::ChainOptimize",
                 "Replace computation root success, new_root: " +
                     new_instruction->name());
    }
    DebugPrint("HloMCO::ChainOptimize",
               "After replace, chain_root.user.size = " +
                   std::to_string(item.first->users().size()));

    changed = true;
  }
  return changed;
}

StatusOr<bool> HloMCO::Run(HloModule* module) {
  bool changed = false;
  TF_RET_CHECK(!module->name().empty());

  if (module->entry_computation()->IsFusionComputation()) {
    return InvalidArgument(
        "Module entry computation cannot be a fusion computation");
  }
  DebugPrint("HloMCO::Run", "Start Run");
  // std::cout << "[HloMCO::Run]:\n" << hlo_text << std::endl;
  for (auto* computation : module->computations()) {
    DebugPrint("HloMCO::Run", "computation: " + computation->ToString());
    DebugPrint("HloMCO::Run", "start matriox_chain_detector");
    MatrixChainDetector matrix_chain_detector;
    // detection matrix chain on the whithin the computation
    // TF_RETURN_IF_ERROR(computation->Accept(&instruction_verifier));
    TF_RETURN_IF_ERROR(computation->Accept(&matrix_chain_detector));
    DebugPrint("HloMCO::Run", "finish matriox_chain_detector");
    auto chain_map = matrix_chain_detector.GetChainMap();
    DebugPrint("HloMCO::Run",
               "chain_map.size = " + std::to_string(chain_map.size()));
    TF_ASSIGN_OR_RETURN(bool changed_for_computation,
                        ChainOptimize(computation, chain_map));
    changed |= changed_for_computation;
    computation->Cleanup();
    DebugPrint("HloMCO::Run",
               "After optimization computation: " + computation->ToString());
  }
  // After these pass, also need to run HloDCE and HloCSE
  return changed;
}

}  // namespace xla
