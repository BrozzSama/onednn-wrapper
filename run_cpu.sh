#!/bin/bash
source $ONEAPI_INSTALL/setvars.sh --force > /dev/null 2>&1
echo "########## Executing the run"
DNNL_VERBOSE=1 ./dpcpp/onednn-training-$1-cpp cpu $2
echo "########## Done with the run"
