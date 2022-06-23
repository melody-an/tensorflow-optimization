

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

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
      std::vector<HloInstruction*> need_to_be_removed_instructions);
  static StatusOr<HloInstruction*> CopyResuableSubgraph(HloInstruction* inst);
  StatusOr<HloInstruction*> ComputeOptimalChainOrder(
      HloInstruction* root, std::vector<HloInstruction*>& chain);
  StatusOr<HloInstruction*> ConstructOptimalChain(
      HloInstruction* orig_root, std::vector<std::vector<int64>>& solution,
      std::vector<HloInstruction*>& chain_instructions);
  Status ConstructOptimalChainHelper(
      HloInstruction* orig_root, std::vector<std::vector<int64>>& solution,
      std::vector<HloInstruction*>& chain_instructions, int64 start_index,
      int64 end_index, std::vector<HloInstruction*>& subgraph_stack);
  const PrecisionConfig& precision_config() const { return precision_config_; }
  Status SetPrecisionConfig(PrecisionConfig& precision_config) {
    precision_config_ = precision_config;
  }

 private:
  const bool only_fusion_computations_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

class ParentsDetector : public DfsHloVisitorWithDefault {
 public:
  ParentsDetector() {}
  Status DefaultAction(HloInstruction* hlo) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;
  Status FinishVisit(HloInstruction* hlo) override { return Status::OK(); }
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
  Status DefaultAction(HloInstruction* hlo) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;
  Status FinishVisit(HloInstruction* hlo) override { return Status::OK(); }
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
  std::vector<HloInstruction*>& GetToBeRemovedInstructions() {
    return need_to_be_removed_instructions;
  }

 private:
  // Each kv pair is (chain_root_ptr, vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
  HloInstruction* chain_root;
  std::vector<HloInstruction*> need_to_be_removed_instructions;
};

class MatrixChainDetector : public DfsHloVisitorWithDefault {
 public:
  MatrixChainDetector() {}

  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetChainMap() {
    return chain_map;
  }
  Status DefaultAction(HloInstruction* hlo) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;
  // Status Postprocess(HloInstruction* hlo) override;

  // Status HandleDot(HloInstruction* dot) override;
  // Status HandleBroadcast(HloInstruction* broadcast) override;
  // Status HandleReshape(HloInstruction* reshape) override;
  // Status HandleTranspose(HloInstruction* transpose) override;

  Status FinishVisit(HloInstruction* hlo) override { return Status::OK(); }
  Status DetectMatrixChain(HloInstruction* chain_root);
  std::vector<HloInstruction*>& GetToBeRemovedInstructions() {
    return need_to_be_removed_instructions;
  }

  Status InsertToBeRemovedInstructions(std::vector<HloInstruction*>& vec) {
    need_to_be_removed_instructions.insert(need_to_be_removed_instructions.end(),
                                        vec.begin(), vec.end());
    return Status::OK();
  }

 private:
  // Each kv pair is (chain_root_ptr, vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
  std::vector<HloInstruction*> need_to_be_removed_instructions;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
