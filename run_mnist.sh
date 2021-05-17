#!/bin/bash
source $ONEAPI_INSTALL/setvars.sh --force > /dev/null 2>&1
echo "########## Executing the run"
./dpcpp/onednn-training-mnist-cpp cpu
echo "########## Done with the run"