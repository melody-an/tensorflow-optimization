#include "tensorflow/compiler/xla/service/hlo_mco.h"

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

namespace xla {
namespace {
class HloMCOTest : public HloTestBase {
 protected:
  HloMCOTest() {}
};
void DumpToFileInDirImpl(std::string dir, std::string filename,
                         std::string contents) {
  // const string& dir = opts.dump_to;
  std::cout << "Dumping " << filename << " to " << dir;

  tensorflow::Env* env = tensorflow::Env::Default();
  // Two threads can race to observe the absence of the dump directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA debug data: " << status;
      return;
    }
  }

  string file_path = tensorflow::io::JoinPath(dir, SanitizeFileName(filename));
  auto status = tensorflow::WriteStringToFile(env, file_path, contents);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << file_path << ": "
               << status;
  }
}

TEST_F(HloMCOTest, PureOptimalMatrixChain) {
  // Test matrix chain only consists of matrices and the chain is already
  // optimal
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule PureOptimalMatrixChain
main{
  %A = f32[10,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot1 = f32[10,30]{1,0} dot(f32[10,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %C = f32[30,40]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[10,40]{1,0} dot(f32[10,30]{1,0} %dot1, f32[30,40]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_1"}
  %D = f32[40,30]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  ROOT %dot3 = f32[10,30]{1,0} dot(f32[10,40]{1,0} %dot2, f32[40,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_2"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  std::string dir = "/vol/bitbucket/ya321/codes/MscProject/test_output/";
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, PureMatrixChain) {
  // Test matrix chain only consists of matrices
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule PureMatrixChain
main{
  %A = f32[40,20]{1,0} parameter(0)
  %B = f32[20,30]{1,0} parameter(1)
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %D = f32[10,30]{1,0} parameter(3)
  ROOT %dot3 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  std::string dir = "/vol/bitbucket/ya321/codes/MscProject/test_output/";
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, MatrixChainAsSubgraph) {
  // Test opotimization in graph which contain a matrix cahin as a subgraph
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule MatrixChainAsSubgraph
main{
  %A = f32[40,20]{1,0} parameter(0)
  %B = f32[20,30]{1,0} parameter(1)
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %D = f32[10,30]{1,0} parameter(3)
  %dot3 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %E = f32[40,30]{1,0} parameter(4)
  ROOT %add = f32[40,30] add(%E, %dot3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  std::string dir = "/vol/bitbucket/ya321/codes/MscProject/test_output/";
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}
TEST_F(HloMCOTest, MatrixVectorChainAsSubgraph) {
  // Test opotimization in graph which contain a matrix cahin as a subgraph
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule MatrixVectorChainAsSubgraph
main{
  %A = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %C = f32[30]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40]{0} dot(f32[40,30]{1,0} %dot1, f32[30]{0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum/Einsum"}
  %D = f32[40]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot3 = f32[] dot(f32[40]{0} %dot2, f32[40]{0} %D), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum_1/Einsum"}
  %E = f32[] parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  ROOT %add1 = f32[] add(f32[] %dot3, f32[] %E), metadata={op_type="AddV2" op_name="add"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  std::string dir = "/vol/bitbucket/ya321/codes/MscProject/test_output/";
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_FALSE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dir;
  DumpToFileInDirImpl(dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

}  // namespace
}  // namespace xla