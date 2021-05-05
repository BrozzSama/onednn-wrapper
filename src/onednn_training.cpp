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
        // RNG for ALL purposes
        std::default_random_engine generator;

        // Load dataset
        auto dataset_path = "shuffled_data/full_dataset.npy";
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<double> dataset;

        shape.clear();
        dataset.clear();

        npy::LoadArrayFromNumpy(dataset_path, shape, fortran_order, dataset);

        std::cout << "shape: ";
        for (size_t i = 0; i < shape.size(); i++)
                std::cout << shape[i] << ", ";
        std::cout << "\n";
        
        // Load labels
        auto labels_path = "shuffled_data/labels.npy";
        std::vector<unsigned long> shape_labels;
        std::vector<double> dataset_labels;

        shape_labels.clear();
        dataset_labels.clear();

        npy::LoadArrayFromNumpy(labels_path, shape_labels, fortran_order, dataset_labels);

        using tag = memory::format_tag;
        using dt = memory::data_type;

        auto eng = engine(engine_kind, 0);
        stream s(eng);

        // Vector of primitives and their execute arguments
        std::vector<primitive> net_fwd, net_bwd_data, net_bwd_weights, net_sgd;
        std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_data_args, net_bwd_weights_args, net_sgd_args;

        const int batch = shape[0];
        const int patch_size = shape[1];
        const int stride = 1;
        const int kernel_size = 3;
        const int n_kernels = 64;
        const float learning_rate = 0.0001;
        // Compute the padding to preserve the same dimension in input and output
        // const int padding = (shape[1] - 1) * stride - shape[1] + kernel_size;
        // padding /= 2;
        int padding = kernel_size - 1;
        padding /= 1;

        // Load inputs inside engine
        memory::dims input_dim = {batch, 1, patch_size, patch_size};
        auto input_memory = memory({{input_dim}, dt::f32, tag::nchw}, eng);
        for (int i = 0; i<dataset.size(); i++){
                if(std::isnan(dataset[i])){
                        std::cout << "Found NAN!!!!!\n";
                }
        }
        write_to_dnnl_memory(dataset.data(), input_memory);

        std::cout << "I wrote the input data!\n";

        for (int i = 0; i<dataset_labels.size(); i++){
                if(std::isnan(dataset_labels[i])){
                        std::cout << "Found NAN in labels!!!!!\n";
                }
        }

        memory::dims labels_dim = {batch, 1};
        auto labels_memory = memory({{labels_dim}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(dataset_labels.data(), labels_memory);

        std::cout << "I wrote the label data!\n";

       

        // pnetcls: conv
        // {batch, 1, 32, 32} (x) {64, 1, 3, 3} -> {batch, 96, 55, 55}
        // strides: {4, 4}

        int conv1 = Conv2D(batch, patch_size, n_kernels, kernel_size, stride, padding, 1, 
               input_memory, net_fwd, net_fwd_args, eng);

        std::cout << "I created the first convolutional layer: " << net_fwd_args.size() << "!\n";

        int relu1 = Eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[conv1][DNNL_ARG_DST],
                            net_fwd, net_fwd_args, eng);

        // pnetcls: conv
        // {batch, 1, 32, 32} (x) {64, 1, 3, 3} -> {batch, 96, 55, 55}
        // strides: {4, 4}

        int conv2 = Conv2D(batch, patch_size, n_kernels, kernel_size, stride, padding, 1, 
               net_fwd_args[relu1][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);

        int relu2 = Eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[conv2][DNNL_ARG_DST],
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the second convolutional layer!\n";

        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        memory::dims fc1_src_dims = {batch, n_kernels, patch_size, patch_size};
        int fc1_output_size = 128;
        int fc1 = Dense(fc1_src_dims, fc1_output_size, 
                        net_fwd_args[relu2][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);

        int relu3 = Eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[fc1][DNNL_ARG_DST],
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the first dense layer!\n";

        // PnetCLS: Fully Connected 2
        // {batch, fc1_output_size} -> {batch, 1}

        memory::dims fc2_src_dims = {batch, fc1_output_size};
        int fc2_output_size = 1;
        int fc2 = Dense(fc2_src_dims, fc2_output_size, 
                        net_fwd_args[relu3][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);
                        
        int sigmoid1 = Eltwise(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, net_fwd_args[fc2][DNNL_ARG_DST],
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the second dense layer!\n";

        // L2 loss

        int loss = L2_Loss(net_fwd_args[sigmoid1][DNNL_ARG_DST], labels_memory, 
                          net_fwd, net_fwd_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Data)-------------------------------------

        std::cout << "Creating backward Loss" << "\n";
        int loss_back = L2_Loss_back(net_fwd_args, net_fwd_args[loss][DNNL_ARG_DST], net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the second Dense layer (back) using forward index: " << fc2 << "\n"; 
        int sigmoid1_back = Eltwise_back(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, net_fwd_args[sigmoid1], 
                                         net_bwd_data_args[loss_back][DNNL_ARG_DST], net_bwd_data, net_bwd_data_args, eng);
        int fc2_back_data = Dense_back_data(net_bwd_data_args[sigmoid1_back][DNNL_ARG_DIFF_SRC], net_fwd_args[fc2], net_bwd_data, net_bwd_data_args, eng);
        int relu3_back_data = Eltwise_back(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[relu3], 
                                         net_bwd_data_args[fc2_back_data][DNNL_ARG_DIFF_SRC], net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the first Dense layer (back) using forward index: " << fc1 << "\n"; 
        int fc1_back_data = Dense_back_data(net_bwd_data_args[relu3_back_data][DNNL_ARG_DIFF_SRC], net_fwd_args[fc1], net_bwd_data, net_bwd_data_args, eng);
        int relu2_back_data = Eltwise_back(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[relu2], 
                                         net_bwd_data_args[fc1_back_data][DNNL_ARG_DIFF_SRC], net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the second convolutional layer (back) using forward index: " << conv2 << "\n"; 
        int conv2_back_data = Conv2D_back_data(net_bwd_data_args[fc1_back_data][DNNL_ARG_DIFF_SRC], net_fwd_args[conv2], stride, padding, 1, net_bwd_data, net_bwd_data_args, eng);
        int relu1_back_data = Eltwise_back(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[relu1], 
                                         net_bwd_data_args[conv2_back_data][DNNL_ARG_DIFF_SRC], net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the first convolutional layer (back) using forward index: " << conv1 << "\n"; 
        int conv1_back_data = Conv2D_back_data(net_bwd_data_args[relu1_back_data][DNNL_ARG_DIFF_SRC] , net_fwd_args[conv1], stride, padding, 1, net_bwd_data, net_bwd_data_args, eng);
        
        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Weights)-------------------------------------
        std::cout << "Creating the second Dense layer (back) using forward index: " << fc2 << "\n"; 
        int fc2_back_weights = Dense_back_weights(net_bwd_data_args[loss_back][DNNL_ARG_DST], net_fwd_args[fc2], net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the first Dense layer (back) using forward index: " << fc1 << "\n"; 
        int fc1_back_weights = Dense_back_weights(net_bwd_data_args[fc2_back_data][DNNL_ARG_DIFF_SRC], net_fwd_args[fc1], net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the second convolutional layer (back) using forward index: " << conv2 << "\n"; 
        int conv2_back_weights = Conv2D_back_weights(net_bwd_data_args[fc1_back_data][DNNL_ARG_DIFF_SRC], net_fwd_args[conv2], stride, padding, 1, net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the first convolutional layer (back) using forward index: " << conv1 << "\n"; 
        int conv1_back_weights = Conv2D_back_weights(net_bwd_data_args[conv2_back_data][DNNL_ARG_DIFF_SRC] , net_fwd_args[conv1], stride, padding, 1, net_bwd_weights, net_bwd_weights_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Weights update -------------------------------------
        updateWeights_SGD(net_fwd_args[conv1][DNNL_ARG_WEIGHTS], 
                   net_bwd_weights_args[conv1_back_weights][DNNL_ARG_DIFF_WEIGHTS], 
                   learning_rate, net_sgd, net_sgd_args, eng);
        updateWeights_SGD(net_fwd_args[conv2][DNNL_ARG_WEIGHTS], 
                   net_bwd_weights_args[conv2_back_weights][DNNL_ARG_DIFF_WEIGHTS], 
                   learning_rate, net_sgd, net_sgd_args, eng);
        updateWeights_SGD(net_fwd_args[fc1][DNNL_ARG_WEIGHTS], 
                   net_bwd_weights_args[fc1_back_weights][DNNL_ARG_DIFF_WEIGHTS], 
                   learning_rate, net_sgd, net_sgd_args, eng);
        updateWeights_SGD(net_fwd_args[fc2][DNNL_ARG_WEIGHTS], 
                   net_bwd_weights_args[fc2_back_weights][DNNL_ARG_DIFF_WEIGHTS], 
                   learning_rate, net_sgd, net_sgd_args, eng);
 

        // didn't we forget anything?
        assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
        assert(net_bwd_data.size() == net_bwd_data_args.size() && "something is missing");
        assert(net_bwd_weights.size() == net_bwd_weights_args.size() && "something is missing");

        int n_iter = 10; // number of iterations for training

        // Prepare memory that will host loss
        std::vector<float> curr_loss(batch);

        for (int i = 0; i<curr_loss.size(); i++){
                curr_loss[i] = 65;
        }

        std::vector<float> weight_test(128);

        for (int i = 0; i<weight_test.size(); i++){
                weight_test[i] = 65;
        }

        std::vector<float> diff_src_test(128), diff_dst_test(128);

        for (int i = 0; i<weight_test.size(); i++){
                diff_src_test[i] = 65;
                diff_dst_test[i] = 65;
        }

        unsigned long batch_size = batch;

        const unsigned long loss_dim [] = {batch_size};

        read_from_dnnl_memory(curr_loss.data(), net_fwd_args[loss][DNNL_ARG_DST]);
        s.wait();
        print_vector2(curr_loss);


        // execute
        while (n_iter)
        {
                // forward

                std::cout << "Iteration # " << n_iter << "\n";

                std::cout << "Forward pass\n";

                for (size_t i = 0; i < net_fwd.size(); ++i)
                        net_fwd.at(i).execute(s, net_fwd_args.at(i));

                // Compute the gradients with respect to the outputs

                std::cout << "Backward data pass\n";
                for (size_t i = 0; i < net_bwd_data.size(); ++i)
                        net_bwd_data.at(i).execute(s, net_bwd_data_args.at(i));

                // Use the previous gradients to compute gradients with respect to the weights

                std::cout << "Backward weights pass\n";
                for (size_t i = 0; i < net_bwd_weights.size(); ++i)
                        net_bwd_weights.at(i).execute(s, net_bwd_weights_args.at(i));


                // Time to update the weights!
                std::cout << "Weights update\n";
                for (size_t i = 0; i < net_sgd.size(); ++i)
                        net_sgd.at(i).execute(s, net_sgd_args.at(i));

                if (n_iter % 1 == 0){  
                        s.wait();
                        read_from_dnnl_memory(curr_loss.data(), net_fwd_args[loss][DNNL_ARG_DST]);
                        s.wait();
                        read_from_dnnl_memory(diff_src_test.data(), net_bwd_weights_args[fc2_back_weights][DNNL_ARG_DIFF_WEIGHTS]);
                        s.wait();
                        read_from_dnnl_memory(diff_dst_test.data(), net_bwd_data_args[fc1_back_data][DNNL_ARG_DIFF_DST]);
                        s.wait();
                        print_vector2(curr_loss);
                        print_vector2(diff_src_test);
                        print_vector2(diff_dst_test);
                        
                        //std::string loss_filename = "./data/losses/iteration_" + std::to_string(n_iter) + ".npy";  
                        //npy::SaveArrayAsNumpy(loss_filename, false, 1, loss_dim, curr_loss);
                }

                --n_iter;
        }




}

int main(int argc, char **argv)
{
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
