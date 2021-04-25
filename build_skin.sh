#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
export EXAMPLE_ROOT=./src/
mkdir dpcpp
cd dpcpp
cmake ../src -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp -DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL -DDNNL_VERBOSE=ON -Ddnnl_DIR=/glob/development-tools/versions/oneapi/2021.2/inteloneapi/dnnl/2021.2.0/lib/cmake/dnnl
make onednn-training-skin-cpp

