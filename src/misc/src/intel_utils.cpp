/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "intel_utils.h"

dnnl::engine::kind validate_engine_kind(dnnl::engine::kind akind) {
    // Checking if a GPU exists on the machine
    if (akind == dnnl::engine::kind::gpu) {
        if (dnnl::engine::get_count(dnnl::engine::kind::gpu) == 0) {
            std::cout << "Application couldn't find GPU, please run with CPU "
                         "instead.\n";
            exit(0);
        }
    }
    return akind;
}

