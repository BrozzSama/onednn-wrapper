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
    dnnl::memory::dims input_dim = {2, 4};
    dnnl::memory::dims output_dim = {2, 2};

    std::vector<float> init_values(input_dim[0] * input_dim[1]) = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> init_values_weights()
    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args;

    auto input_memory = memory({{input_dim}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(init_values.data(), input_memory);

    int fc1 = Dense(fc1_src_dims, fc1_output_size, input, net_fwd, net_fwd_args, eng);

    // Fix the weights

    write_to_dnnl_memory()

    std::cout << "Writing a vector to memory\n";
    
    read_from_dnnl_memory(read_values.data(), loss_diff);
    s.wait();
    print_vector2(read_values);


}

int main(int argc, char **argv)
{
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
