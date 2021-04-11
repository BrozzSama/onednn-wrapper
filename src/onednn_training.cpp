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
//#include "../include/util.hpp"
#include "../include/layers.hpp"


using namespace dnnl;

void simple_net(engine::kind engine_kind)
{
        // RNG for ALL purposes
        std::default_random_engine generator;

        auto path = "data/skull_32_vessel.npy";
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<double> net_src;

        shape.clear();
        net_src.clear();

        npy::LoadArrayFromNumpy(path, shape, fortran_order, net_src);

        std::cout << "shape: ";
        for (size_t i = 0; i < shape.size(); i++)
                std::cout << shape[i] << ", ";
        std::cout << "\n";

        using tag = memory::format_tag;
        using dt = memory::data_type;

        auto eng = engine(engine_kind, 0);
        stream s(eng);

        // Vector of primitives and their execute arguments
        std::vector<primitive> net_fwd, net_bwd;
        std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;

        const int batch = shape[0];
        const int patch_size = shape[1];
        const int stride = 1;
        const int kernel_size = 3;
        const int n_kernels = 64;

        // Compute the padding to preserve the same dimension in input and output
        // const int padding = (shape[1] - 1) * stride - shape[1] + kernel_size;
        // padding /= 2;
        int padding = kernel_size - 1;
        padding /= 1;

        // Load input inside engine
        memory::dims input_dim = {batch, 1, patch_size, patch_size};
        auto input_memory = memory({{input_dim}, dt::f32, tag::nchw}, eng);
        write_to_dnnl_memory(net_src.data(), input_memory);

        std::cout << "I wrote the data in Source!\n";

        // pnetcls: conv
        // {batch, 1, 32, 32} (x) {64, 1, 3, 3} -> {batch, 96, 55, 55}
        // strides: {4, 4}

        int conv1 = Conv2D(batch, patch_size, n_kernels, kernel_size, stride, padding, 1, 
               algorithm::eltwise_relu, input_memory, net_fwd, net_fwd_args, eng);

        std::cout << "I created the first convolutional layer: " << net_fwd_args.size() << "!\n";

        // pnetcls: conv
        // {batch, 1, 32, 32} (x) {64, 1, 3, 3} -> {batch, 96, 55, 55}
        // strides: {4, 4}

        int conv2 = Conv2D(batch, patch_size, n_kernels, kernel_size, stride, padding, 1, 
               algorithm::eltwise_relu, net_fwd_args[conv1][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);

        std::cout << "I created the second convolutional layer!\n";

        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        memory::dims fc1_src_dims = {batch, n_kernels, patch_size, patch_size};
        int fc1_output_size = 128;
        int fc1 = Dense(fc1_src_dims, fc1_output_size, algorithm::eltwise_relu, 
                             net_fwd_args[conv2][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);

        std::cout << "I created the first dense layer!\n";

        // PnetCLS: Fully Connected 2
        // {batch, fc1_output_size} -> {batch, 1}

        memory::dims fc2_src_dims = {batch, fc1_output_size};
        int fc2_output_size = 1;
        int fc2 = Dense(fc2_src_dims, fc2_output_size, algorithm::eltwise_logistic, 
                             net_fwd_args[fc1][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);

        std::cout << "I created the second dense layer!\n";


        //-----------------------------------------------------------------------
        //----------------- Backward Stream -------------------------------------
        // ... user diff_data in float data type ...

        std::cout << "Creating the second Dense layer (back) using forward index: " << fc2 << "\n"; 
        int fc2_back = Dense_back(net_fwd_args[fc2], algorithm::eltwise_logistic, net_bwd, net_bwd_args, eng);
        std::cout << "Creating the first Dense layer (back) using forward index: " << fc1 << "\n"; 
        int fc1_back = Dense_back(net_fwd_args[fc1], algorithm::eltwise_logistic, net_bwd, net_bwd_args, eng);
        std::cout << "Creating the second convolutional layer (back) using forward index: " << conv2 << "\n"; 
        int conv2_back = Conv2D_back(net_fwd_args[conv2], stride, padding, 1, algorithm::eltwise_relu, net_bwd, net_bwd_args, eng);
        std::cout << "Creating the first convolutional layer (back) using forward index: " << conv1 << "\n"; 
        int conv1_back = Conv2D_back(net_fwd_args[conv1], stride, padding, 1, algorithm::eltwise_relu, net_bwd, net_bwd_args, eng);
        

        // didn't we forget anything?
        assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
        assert(net_bwd.size() == net_bwd_args.size() && "something is missing");

        int n_iter = 50; // number of iterations for training
        // execute
        while (n_iter)
        {
                // forward

                std::cout << "Iteration # " << n_iter << "\n";

                for (size_t i = 0; i < net_fwd.size(); ++i)
                        net_fwd.at(i).execute(s, net_fwd_args.at(i));

                // Compute loss and write to diff_dst

                
                // update net_diff_dst
                // auto net_output = pool_user_dst_memory.get_data_handle();
                // ..user updates net_diff_dst using net_output...
                // some user defined func update_diff_dst(net_diff_dst.data(),
                // net_output)

                for (size_t i = 0; i < net_bwd.size(); ++i)
                        net_bwd.at(i).execute(s, net_bwd_args.at(i));
                // update weights and bias using diff weights and bias
                //
                // auto net_diff_weights
                //     = conv_user_diff_weights_memory.get_data_handle();
                // auto net_diff_bias = conv_diff_bias_memory.get_data_handle();
                //
                // ...user updates weights and bias using diff weights and bias...
                //
                // some user defined func update_weights(conv_weights.data(),
                // conv_bias.data(), net_diff_weights, net_diff_bias);

                --n_iter;
        }

        s.wait();
}

int main(int argc, char **argv)
{
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
