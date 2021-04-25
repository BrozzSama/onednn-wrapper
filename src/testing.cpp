/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

/// @example cnn_training_bf16.cpp
/// @copybrief cnn_training_bf16_cpp
///
/// @page cnn_training_bf16_cpp CNN bf16 training example
/// This C++ API example demonstrates how to build an AlexNet model training
/// using the bfloat16 data type.
///
/// The example implements a few layers from AlexNet model.
///
/// @include cnn_training_bf16.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <random>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

#include "../include/npy.hpp"
#include "../include/util.hpp"
#include "../include/layers_fwd.hpp"
#include "../include/layers_bwd_data.hpp"
#include "../include/layers_bwd_weights.hpp"
#include "../include/losses.hpp"
#include "../include/weights_update.hpp"

using namespace dnnl;

void simple_net(engine::kind engine_kind)
{

    

    std::cout << "Hello, time to test some stuff, let's go.\n";

    std::cout << "Initializing engine...\n";
    auto eng = engine(engine_kind, 0);
    stream s(eng);

    // Write to memory and read from memory

    
    std::cout << "Allocating oneDNN memory\n";
    dnnl::memory::dims dim_nc = {64, 1};

    auto loss_diff_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto loss_diff = dnnl::memory(loss_diff_md, eng);

    std::vector<float> init_values(product(dim_nc)), read_values(product(dim_nc));

    for (int i = 0; i<init_values.size(); i++){
        init_values[i] = 65.f;
    }

    std::cout << "Writing a vector to memory\n";
    write_to_dnnl_memory((void*) init_values.data(), loss_diff);
    read_from_dnnl_memory(read_values.data(), loss_diff);
    s.wait();
    print_vector2(read_values);


}

int main(int argc, char **argv)
{
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
