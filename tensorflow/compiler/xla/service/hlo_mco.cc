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

template <typename Visitor>
static Status ConstructChainPreorderDFS(
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

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0)
        << "[ConstructChainPreorderDFS] " << current_id << ": " << current_node
        << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "[ConstructChainPreorderDFS] "
              << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    dfs_stack.pop_back();
    VLOG(2) << "[ConstructChainPreorderDFS] "
            << "Visiting HLO %" << current_node->name();
    bool is_matmul_node = visitor->Preprocess(current_node);
    visitor->SetVisitState(current_id, Visitor::kVisited);

    if (!is_matmul_node) {
      // for ohter op, we just target the following nodes in the current branch
      // as a single operand
      continue
    }

    const size_t old_dfs_stack_size = dfs_stack.size();
    CHECK_EQ(current_node->operands().size(), 2)
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "[ConstructChainPreorderDFS] A cycle is detected while visiting "
            "instruction %s",
            current_node->ToString());
      }
    }

    if (!ignore_control_predecessors) {
      for (HloInstruction* child : current_node->control_predecessors()) {
        if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
          PrintCycle(child, &dfs_stack);
          return FailedPrecondition(
              "ConstructChainPreorderDFS] A cycle is detected while visiting "
              "instruction %s",
              current_node->ToString());
        }
      }
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return Status::OK();
}

}  // namespace

Status ParentsDetector::Preprocess(HloInstruction* hlo) {
  // add operand-parents relation of current node and its operands in to
  // global_operand_parent_map
  for (auto op : hlo->operands()) {
    if (global_operand_parent_map.contains(op)) {
      global_operand_parent_map[op].emplace_back(hlo);
    } else {
      global_operand_parent_map.insert(op, {hlo});
    }
  }
  return Status::OK();
}

bool ChainRecorder::Preprocess(HloInstruction* hlo) {
  bool is_matmul = true;
  if (hlo->opcode() != HloOpcode::kDot) {
    chain_map[chain_root].emplace_back(hlo);
    is_matmul = false
  }
  return is_matmul;
}

Status MatrixChainDetector::Preprocess(HloInstruction* hlo) {
  const DotDimensionNumbers& dnums = hlo->dot_dimension_numbers();
  if (hlo->opcode() != HloOpcode::kDot) {
    if (dnums.lhs_contracting_dimensions_size() != 1 ||
        dnums.rhs_contracting_dimensions_size() != 1 ||
        dnums.lhs_batch_dimensions_size() != 0 ||
        dnums.rhs_batch_dimensions_size() != 0 ||
        dot->shape().dimensions_size() != 2) {
      VLOG(10) << "MatrixChainDetector: Can only optimize 2D, non-batch dot "
                  "operations.";
      return Status::OK();
    }
    for (auto op : hlo->operands()) {
      if (op->opcode() == HloOpcode::kDot) {
        ChainRecorder chain_recorder(op);
        ConstructChainPreorderDFS(op, chain_recorder);
        if (chain_recorder.GetChainLength(op) < 3) {
          // Single dot operation doesn't need optimize
          chain_recorder.RemoveChain(op);
        } else {
          auto chain = chain_recorder.GetChain(op);
          chain_map.insert(
              op, std::vector<HloInstruction*>(chain.begin(), chain.end()));
        }
      }
    }
  }
  return Status::OK();
}

Status MatrixChainDetector::Postprocess(HloInstruction* hlo) {
  return Status::OK();
}

Status MatrixChainDetector::HandleDot(HloInstruction* dot) {
  return Status::OK();
}

Status HloMCO::CopyResuableSubgraph(
    HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
        chain_map) {
  for (auto& item : chain_map) {
    if (item.first.users().size() > 1) {
      // Store a snapshot of users before DeepCopyInstruction, as
      // DeepCopyInstruction introduces new users of the instruction.
      std::vector<HloInstruction*> users;
      users.assign(item.first.users().begin() + 1, item.first.users().end());
      auto statusor = computation->DeepCopyInstruction(item.first);
      HloInstruction* new_instruction = std::move(statusor).ValueOrDie();
      item.first.ReplaceUsesWith(users, new_instruction);
    }
  }
  return Status::OK();
}

StatusOr<HloInstruction*> ConstructOptimalChain(
    std::vector<std::vector<int64>>& solution,
    std::vector<HloInstruction*>& chain_instructions) {
  HloInstruction* optimal_root = nullptr;
  absl::InlinedVector<HloInstruction*, 16> subgraph_stack;
  ConstructOptimalChainHelper(solution, chain_instructions, 0,
                              chain_instructions.size() - 1, subgraph_stack);
  CHECK_EQ(subgraph_stack.size(), 1);
  optimal_root = subgraph_stack.back();
  return optimal_root;
}

Status HloMCO::ConstructOptimalChainHelper(
    std::vector<std::vector<int64>>& solution,
    std::vector<HloInstruction*>& chain_instructions, int64 start_index,
    int64 end_index, absl::InlinedVector<HloInstruction*, 16>& subgraph_stack) {
  auto create_dot = [&](HloInstruction* l, HloInstruction* r) {
    const Shape* lhs_shape = l.shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape->dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    TF_ASSIGN_OR_RETURN(
        auto output_shape,
        ShapeInference::InferDotOpShape(
            chain_instructions[start_index]->shape(),
            chain_instructions[end_index]->shape(), dimension_numbers));

    auto new_matmul_inst_ptr = HloInstruction::CreateDot(
        output_shape, l, r, dimension_numbers, l.precision_config());
    HloInstruction* new_matmul_inst = new_matmul_inst_ptr.get();
    subgraph_stack.emplace_back(new_matmul_inst);
  };

  if (start_index == end_index) {
    subgraph_stack.emplace_back(chain_instructions[start_index]);
    return Status::OK();
  }

  if (start_index == end_index - 1) {
    // construction a new matmul op
    create_dot(chain_instructions[start_index], chain_instructions[end_index]);
    return Status::OK();
  }

  ConstructOptimalChainHelper(solution, chain_instructions, start_index,
                              solution[start_index][end_index]);
  ConstructOptimalChainHelper(solution, chain_instructions,
                              solution[start_index][end_index] + 1, end_index);
  // since this is a stack, the right_operand is on the top of left_operand
  HloInstruction* right_operand = subgraph_stack.back();
  subgraph_stack.pop();
  HloInstruction* left_operand = subgraph_stack.back();
  subgraph_stack.pop();
  creat_dot(left_operand, right_operand);

  return Status::OK();
}

StatusOr<HloInstruction*> HloMCO::ComputeOptimalChainOrder(
    HloInstruction* root, std::vector<HloInstruction*>& chain) {
  HloInstruction* optimal_root = nullptr;
  int64 chain_length = chain.size();
  // sizes[i] stores the number of columns of operand[i]
  // sizes[i+1] stores the number of columns of operand[i]
  std::vector<int64> sizes(chain_length + 1, 0);
  for (auto i = 0; i < chain_length; ++i) {
    CHECK_LE(chain[i]->shape().rank(), 2);
    if (chain[i]->shape().rank() == 1) {
      // vector operand
      sizes[i] = 1;
      sizes[i + 1] = chain[i]->shape().dimension(0);
    } else if (chain[i]->shape().rank() == 2) {
      // matrix operand
      sizes[i] = chain[i]->shape().dimension(0);
      sizes[i + 1] = chain[i]->shape().dimension(1);
    }
  }
  // solution[i][j] stores optimal break point in
  // subexpression from i to j.
  std::vector<std::vector<int64>> solution(chain_length,
                                           vector<int>(chain_length, 0));
  /* costs[i,j] = Minimum number of scalar multiplications
        needed to compute the matrix A[i]A[i+1]...A[j] =
        A[i..j] */
  std::vector<std::vector<int64>> costs(
      chain_length,
      std::vector<int>(chain_length, std::numeric_limits<int64>::max()));
  // cost is zero when multiplying one matrix.
  for (int i = 0; i < chain_length; i++) costs[i][i] = 0;

  // L is chain length.
  // Dynamic Programming to find the optimal computing order
  for (int L = 2; L <= chain_length; L++) {
    for (int i = 0; i <= chain_length - L; i++) {
      // L = 2:             L = n:
      // i = 0 -> n-2       i = 0 -> 0
      // j = 1 -> n-1       j = n-1 -> n-1
      int j = i + L - 1;
      // [i,j] is the [start,end] index of the current subchain
      for (int k = i; k <= j - 1; k++) {
        // compute
        auto first_cost = chain[i].shape();
        int cost = costs[i][k] + costs[k + 1][j] + size[i] +
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
  auto status_or = ConstructOptimalChain(solution, chain);
  optimal_root = = std::move(status_or).ValueOrDie();
  return optimal_root;
}

StatusOr<bool> HloMCO::ChainOptimize(
    HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
        chain_map,
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
        global_operand_parent_map) {
  bool changed = false;
  TF_RETURN_IF_ERROR(CopyResuableSubgraph(computation, chain_map));

  for (auto& item : chain_map) {
    auto status_or = ComputeOptimalChainOrder(item.first, item.second);
    HloInstruction* new_instruction = std::move(status_or).ValueOrDie();
    // TODO: 是不是应该让CopyResuableSubgraph 传回来每个chain
    // TODO:
    // 要替换的user，因为如果之前CopyResuableSubgraph中DeepCopyInstruction会
    // TODO：introduce new user的话，不确定如果现在replace
    // TODO: all的话，替换掉copy引入的user会不会有问题，但感觉应该没有
    item.first.ReplaceAllUsesWith(new_instruction);
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

  for (auto* computation : module->computations()) {
    ParentsDetector parents_detector;
    // first establish the parents-children relationships whithin the
    // computation
    TF_RETURN_IF_ERROR(computation->Accept(&parents_detector));
    MatrixChainDetector matrix_chain_detector();
    // detection matrix chain on the whithin the computation
    TF_RETURN_IF_ERROR(computation->Accept(&matrix_chain_detector));
    TF_ASSIGN_OR_RETURN(
        bool changed_for_computation,
        ChainOptimize(computation, matrix_chain_detector.GetChainMap(),
                      parents_detector.GetGlobalOperandParentMap()));
    changed |= changed_for_computation;
  }

  return changed;
}

}  // namespace xla
