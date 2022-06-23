

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which performs matrix chain optimization. The pass
// iterates over the instructions in topological order to detect matrix chain
// and then performa a MCO algorithm to produce high efficient matrix chain
// solutions
class HloMCO : public HloModulePass {
 public:
  explicit HloMCO(bool only_fusion_computations = false)
      : only_fusion_computations_(only_fusion_computations) {}
  ~HloMCO() override = default;
  absl::string_view name() const override { return "mco"; }

  // Run MCO on the given module. Returns whether the module was changed
  // (matrix chains were found and optimizesd).
  StatusOr<bool> Run(HloModule* module) override;
  StatusOr<bool> ChainOptimize(
      HloComputation* computation,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
          chain_map,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
          global_operand_parent_map);
  Status CopyResuableSubgraph(
      HloComputation* computation,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
          global_operand_parent_map);
  StatusOr<HloInstruction*> ComputeOptimalChainOrder(
      HloInstruction* root, std::vector<HloInstruction*>& chain);
  StatusOr<HloInstruction*> ConstructOptimalChain(
      std::vector<std::vector<int64>>& solution,
      std::vector<HloInstruction*>& chain_instructions);
  Status ConstructOptimalChainHelper(
      std::vector<std::vector<int64>>& solution,
      std::vector<HloInstruction*>& chain_instructions, int64 start_index,
      int64 end_index,
      absl::InlinedVector<HloInstruction*, 16>& subgraph_stack);

 private:
  const bool only_fusion_computations_;
};

class ParentsDetector : public DfsHloVisitorWithDefault {
 public:
  ParentsDetector() {}
  Status DefaultAction(HloInstruction*) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;
  Status FinishVisit(HloInstruction*) override { return Status::OK(); }
  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetGlobalOperandParentMap() {
    return global_operand_parent_map;
  }

 private:
  // Each kv pair is (instruction_ptr, vector<parent_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      global_operand_parent_map;
};

class ChainRecorder : public DfsHloVisitorWithDefault {
 public:
  ChainRecorder(HloInstruction* input_chain_root)
      : chain_root(input_chain_root) {}
  Status DefaultAction(HloInstruction*) override { return Status::OK(); }
  bool Preprocess(HloInstruction* hlo) override;
  Status FinishVisit(HloInstruction*) override { return Status::OK(); }
  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetChainMap() {
    return chain_map;
  }
  int64 GetChainLength(HloInstruction* root) { return chain_map[root].size(); }
  std::vector<HloInstruction*> GetChain(HloInstruction* root) {
    return chain_map[root];
  }
  Status RemoveChain(HloInstruction* root) {
    chain_map.erase(root);
    return Status::OK();
  }

 private:
  // Each kv pair is (chain_root_ptr, vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
  HloInstruction* chain_root;
};

class MatrixChainDetector : public DfsHloVisitorWithDefault {
 public:
  MatrixChainDetector() : {}

  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetChainMap() {
    return chain_map;
  }
  Status DefaultAction(HloInstruction*) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* hlo) override;

  Status HandleDot(HloInstruction* dot) override;
  // Status HandleBroadcast(HloInstruction* broadcast) override;
  // Status HandleReshape(HloInstruction* reshape) override;
  // Status HandleTranspose(HloInstruction* transpose) override;

  Status FinishVisit(HloInstruction*) override { return Status::OK(); }

 private:
  // Each kv pair is (chain_root_ptr, vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
