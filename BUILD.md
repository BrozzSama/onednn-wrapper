# Building and running

This page describes how to build and run an application that used oneDNN wrapper and Intel Devcloud. The procedure has not been tested on a local machine, but as long as the header files are specified correctly it should run reliably.

# Building

Building is done through CMake, to make this process easier there are some samples build scripts that can be used. What these script do is simply set the correct variable for oneDNN and build the application by running:

    mkdir dpcpp
    cd dpcpp
    cmake ../src -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp -DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL -DDNNL_VERBOSE=ON -Ddnnl_DIR=/glob/development-tools/versions/oneapi/2021.2/inteloneapi/dnnl/2021.2.0/lib/cmake/dnnl
    make filename-cpp

The folder dpcpp is created to separated the built file from the source code, the DNNL_VERBOSE builds the application with the verbose symbols, which allow to set the DNNL_VERBOSE for debugging, the make command receives the filename with the underscores and dots converted to dashes.

# Running

To run an application the suggested procedure is to queue the job on Intel DevCloud. This is achieved by using the q script, which has a modifiable property variable that allows to choose between different configurations. To run the application you can use the run_gpu and run_cpu scripts, which choose the proper engine and run the application, for example if we wanted to run the skin dataset example we would do:

    ./q ./run_gpu.sh skin config/config_skint_gpu.json

Note that the run_gpu and run_cpu scripts are just suggestions on how to run the application, once the binary file is compiled you can choose whatever options suits you to queue and run the jobs.




