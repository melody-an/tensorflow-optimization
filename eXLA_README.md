# eXLA-v2

This project is built on top of TensorFlow, XLA and eXLA. So the basic usage and building steps follow https://www.tensorflow.org/install/source?hl=en#setup_for_linux_and_macos and https://www.tensorflow.org/xla. 

Note: As this project is an extension to eXLA, which is based on TensorFlow, the code base contains all the code of TensorFlow as well as the code of eXLA. The following files are new contributions to this project:
* tensorflow/compiler/xla/service/hlo\_mco.h
* tensorflow/compiler/xla/service/hlo\_mco.cc
* tensorflow/compiler/xla/service/ reshape\_sinker.h
* tensorflow/compiler/xla/service/reshape\_sinker.cc
* tensorflow/compiler/xla/service/tensor\_splitter\_v2.h
* tensorflow /compiler/xla/service/tensor\_splitter\_v2.cc

## Installation
Here are a simplified version of building the project form scratch:
1. Install Bazelisk: https://www.tensorflow.org/install/source?hl=en#install_bazel
2. git clone https://github.com/melody-an/tensorflow-optimization.git
3. git checkout r2.9
4. Configure compliation settings follow the instructions from https://www.tensorflow.org/install/source?hl=en#configure_the_build
    ```bash
    ./configure
    ```
5. Compile TensorFlow form source and build pip package
    ```bash
    TF_PIP_PATH=~/Package/tf-pip
    rm -rf $TF_PIP_PATH 
    bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package &&
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package $TF_PIP_PATH
    ```
6. Install the generated TensorFlow package
    ```bash
    pip uninstall -y tensorflow tensorflow-estimator &&
    pip install -U $TF_PIP_PATH/tensorflow-*.whl
    ```
Note: Since TensorFlow is a super large project so the compilation may take several hours to complete.
## User Guide
This project is an improved version of [eXLA](https://arxiv.org/abs/2206.14148). This repo https://github.com/awav/gambit includes some benchmarks and guides of eXLA. 
This project introduce two optimisation pass in eXLA, one is for matrix chain optimisation and another is for improved tensor-splitter. 
If you want to use our optimisation passes, you need to first enable XLA compilation in your code, there are multiple ways:
1. Use decorator
    ```python
    @tf.function(jit_compile=True)
    def test_einsum(A,B,v):
        return tf.einsum('ij,i->j',tf.einsum('ki,kj->ij',A,B),v) 
    ```
2. Use `tf.function` to wrap your function and then use the wrapped function
    ```python
    def test_einsum(A,B,v):
        return tf.einsum('ij,i->j',tf.einsum('ki,kj->ij',A,B),v) 
    compiled_test_einsum = tf.function(test_einsum, jit_compile=True)
    ```
When you want to execute the code with XLA, you need to use the following commands:
```bash

# Enable our new passes
TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_disable_hlo_passes=tensor-splitter --xla_tensor_size_threshold=1GB --xla_tensor_split_size=100MB" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python xxx.py
# Disable our new passes
TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_disable_hlo_passes=tensor-splitter-v2,mco --xla_tensor_size_threshold=1GB --xla_tensor_split_size=100MB" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python xxx.py
```

For `tensor-splitter-v2` which split large tensors in a computational graph nees 2 XLA flags. `--xla_tensor_size_threshold` is used to indicate the threshold of tensor size which will be regarded as large tensors.  `--xla_tensor_split_size` is used to indicate the tensor size after performing splitting, which shoud be less than or equal to the threshold. For example `--xla_tensor_size_threshold=1GB --xla_tensor_split_size=100MB`. In order to save memory as you need, you need to select the two values.

## Experimental Scripts
After finishing the above installation steps, if you want to run our experimental scripts in the report, you need to clone the repo https://github.com/melody-an/gambit.git, which is the test repo for eXLA but we add the following new scripts. All the other scripts in the repo is benchmarks already exists in the original eXLA.
1. MCO: https://github.com/melody-an/gambit/blob/submission/bench/bench_mco_plot_data.py
    ```bash
    # bench with mco
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-exp-mco" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_disable_hlo_passes=tensor-splitter,tensor-splitter-v2,dot-order-optimizer" python ./bench_mco_plot_data.py --repeat 3 --dump-name "mco" 2>&1 | tee output-exp-mco.log
    # bench without mco
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-exp-mco" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_disable_hlo_passes=mco,tensor-splitter,tensor-splitter-v2,dot-order-optimizer" python ./bench_mco_plot_data.py --repeat 3 --dump-name "none" 2>&1 | tee output-exp-mco.log
    ```
2. ViT: https://github.com/melody-an/gambit/blob/submission/bench/exp_vit_data.py
   ```bash
    #v2
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-bench-vit" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=25MB --xla_tensor_split_size=2MB --xla_disable_hlo_passes=tensor-splitter" python ./exp_vit_data.py 2>&1 --compile "xla" --dump-name "v2" --repeat 3 2>&1 | tee output-exp-vit.log
    #v1
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-bench-vit" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=25MB --xla_tensor_split_size=2MB --xla_disable_hlo_passes=tensor-splitter-v2,mco" python ./exp_vit_data.py 2>&1 --compile "xla" --dump-name "v1" --repeat 3 2>&1 | tee output-exp-vit.log
    #non
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-bench-vit" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=25MB --xla_tensor_split_size=2MB --xla_disable_hlo_passes=tensor-splitter-v2,tensor-splitter,mco,reshape-sinker" python ./exp_vit_data.py 2>&1 --compile "xla" --dump-name "none" --repeat 3 2>&1 | tee output-exp-vit.log
    ```
3. SGPR: https://github.com/melody-an/gambit/blob/submission/bench/exp_sgpr_plot_data.py
    ```bash
    # v2
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-bench-spgr" XLA_FLAGS="--xla_tensor_size_threshold=1GB --xla_tensor_split_size=100MB --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_disable_hlo_passes=tensor-splitter,mco" python ./exp_sgpr_plot_data.py  --compile "xla" --dump-name "v2"  --repeat 3 | tee output.log
    #v1
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-bench-spgr" XLA_FLAGS="--xla_tensor_size_threshold=1GB --xla_tensor_split_size=100MB --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_disable_hlo_passes=tensor-splitter-v2,mco,reshape-sinker" python ./exp_sgpr_plot_data.py  --compile "xla" --dump-name "v1"  --repeat 3 | tee output.log
    #non
    TF_CPP_MIN_LOG_LEVEL=3 DUMPDIR="xla-bench-spgr" XLA_FLAGS="--xla_tensor_size_threshold=1GB --xla_tensor_split_size=100MB --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_disable_hlo_passes=tensor-splitter-v2,tensor-splitter,mco,reshape-sinker" python ./exp_sgpr_plot_data.py  --compile "xla" --dump-name "none"  --repeat 3 | tee output.log
    ```


