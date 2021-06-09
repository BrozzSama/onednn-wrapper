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

#include "intel_utils.h"

#include "misc/include/npy.hpp"
#include "misc/include/util.h"
#include "layers/include/layers_fwd.h"
#include "layers/include/layers_bwd_data.h"
#include "layers/include/layers_bwd_weights.h"
#include "layers/include/losses.h"
#include "layers/include/weights_update.h"
#include "layers/include/primitive_wrappers.h"
#include "misc/include/data_loader.h"
#include "misc/include/json.hpp"

using namespace dnnl;

void simple_net(engine::kind engine_kind, int argc, char** argv)
{
        // Check if configuration file exists

        nlohmann::json config_file;

        if (argc != 3){
                std::cout << "No configuration file specified\n";
                exit(1);
        }

        // Read JSON configuration
        std::ifstream config_file_stream(argv[2]);
        if (config_file_stream.is_open())
        {
                config_file_stream >> config_file;
                config_file_stream.close();
        }
        else
                std::cout << "Unable to open file"; 

        // RNG for ALL purposes
        std::default_random_engine generator;

        // Open oneAPI engine
        auto eng = engine(engine_kind, 0);
        stream s(eng);

        // MNIST dataset (binary classification, only images corresponding to 0 and 1 were kept)
        const long samples = config_file["samples"];
        const long samples_val = config_file["samples_val"];
        const long batch = config_file["minibatch_size"];

        // Load dataset
        auto dataset_path = config_file["dataset_path"];
        auto labels_path = config_file["labels_path"];

        auto dataset_path_val = config_file["dataset_path_val"];
        auto labels_path_val = config_file["labels_path_val"];

        std::vector<long> dataset_shape = {samples, 3};      //Skin dataset
        std::vector<long> dataset_shape_val = {samples_val, 3};      //Skin dataset

        // Data loader 

        DataLoader skin_data(dataset_path, labels_path, samples, batch, dataset_shape, eng);
        std::cout << "Dataloader instantiated\n";

        DataLoader skin_data_val(dataset_path_val, labels_path_val, samples_val, samples_val, dataset_shape_val, eng);
        std::cout << "Dataloader instantiated\n";

        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Vector of primitives and their execute arguments
        std::vector<primitive> net_fwd, net_fwd_inf, net_bwd_data, net_bwd_weights, net_sgd;
        std::vector<std::unordered_map<int, memory>> net_fwd_args, net_fwd_inf_args, net_bwd_data_args, net_bwd_weights_args, net_sgd_args;

        const int n_features = dataset_shape[1];
        const float learning_rate = config_file["learning_rate"];

        // Load inputs inside engine
        memory::dims input_dim = {batch, n_features};
        memory::dims labels_dim = {batch, 1};
        auto input_memory = memory({{input_dim}, dt::f32, tag::nc}, eng);
        auto labels_memory = memory({{labels_dim}, dt::f32, tag::nc}, eng);

        memory::dims input_dim_val = {samples_val, n_features};
        memory::dims labels_dim_val = {samples_val, 1};
        auto input_memory_val = memory({{input_dim_val}, dt::f32, tag::nc}, eng);
        auto labels_memory_val = memory({{labels_dim_val}, dt::f32, tag::nc}, eng);

        std::cout << "Writing first batch to memory\n";
        skin_data.write_to_memory(input_memory, labels_memory);
        skin_data_val.write_to_memory(input_memory_val, labels_memory_val);

        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        memory::dims fc1_src_dims = {batch, n_features};
        int fc1_output_size = 5;
        Dense fc1(fc1_src_dims, fc1_output_size, 
                        input_memory, net_fwd, net_fwd_args, eng);

        Eltwise relu1(dnnl::algorithm::eltwise_relu, 0.f, 0.f, fc1.arg_dst,
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the first dense layer!\n";

        // PnetCLS: Fully Connected 2
        // {batch, fc1_output_size} -> {batch, 1}

        memory::dims fc2_src_dims = {batch, fc1_output_size};
        int fc2_output_size = 1;
        Dense fc2(fc2_src_dims, fc2_output_size, 
                        relu1.arg_dst, net_fwd, net_fwd_args, eng);
                        
        Eltwise sigmoid1 (dnnl::algorithm::eltwise_logistic, 0.f, 0.f, fc2.arg_dst,
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the second dense layer!\n";

        // BCE loss

        int loss = binaryCrossEntropyLoss(sigmoid1.arg_dst, labels_memory, net_fwd, net_fwd_args, eng);

        // Inference

        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        memory::dims fc1_src_dims_inf = {samples_val, n_features};
        Dense fc1_inf(fc1_src_dims_inf, fc1_output_size, 
                        input_memory_val, net_fwd_inf, net_fwd_inf_args, eng);
        Eltwise relu1_inf(dnnl::algorithm::eltwise_relu, 0.f, 0.f, fc1_inf.arg_dst,
                            net_fwd_inf, net_fwd_inf_args, eng);

        std::cout << "I created the first dense layer!\n";

        // PnetCLS: Fully Connected 2
        // {batch, fc1_output_size} -> {batch, 1}

        memory::dims fc2_src_dims_inf = {samples_val, fc1_output_size};
        Dense fc2_inf(fc2_src_dims_inf, fc2_output_size, 
                        relu1_inf.arg_dst, net_fwd_inf, net_fwd_inf_args, eng);
                        
        Eltwise sigmoid1_inf(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, fc2_inf.arg_dst,
                            net_fwd_inf, net_fwd_inf_args, eng);

        std::cout << "I created the second dense layer!\n";

        // BCE loss

        int loss_inf = binaryCrossEntropyLoss(sigmoid1_inf.arg_dst, labels_memory_val, net_fwd_inf, net_fwd_inf_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Data)-------------------------------------

        std::cout << "Creating backward Loss" << "\n";
        //int loss_back = L2_Loss_back(net_fwd_args[sigmoid1][DNNL_ARG_DST], labels_memory, net_bwd_data, net_bwd_data_args, eng);
        int loss_back = binaryCrossEntropyLoss_back(sigmoid1.arg_dst, labels_memory, net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the second Dense layer (back)\n"; 
        Eltwise_back sigmoid1_back_data(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, sigmoid1, 
                                         net_bwd_data_args[loss_back][DNNL_ARG_DST], net_bwd_data, net_bwd_data_args, eng);
        Dense_back_data fc2_back_data(sigmoid1_back_data.arg_diff_src, fc2, net_bwd_data, net_bwd_data_args, eng);
        Eltwise_back relu1_back_data(dnnl::algorithm::eltwise_relu, 0.f, 0.f, relu1, 
                                         fc2_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the first Dense layer (back)\n"; 
        Dense_back_data fc1_back_data(relu1_back_data.arg_diff_src, fc1, net_bwd_data, net_bwd_data_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Weights)-------------------------------------
        std::cout << "Creating the second Dense layer (back)\n"; 
        Dense_back_weights fc2_back_weights(sigmoid1_back_data.arg_diff_src, fc2, net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the first Dense layer (back)\n"; 
        Dense_back_weights fc1_back_weights(relu1_back_data.arg_diff_src, fc1, net_bwd_weights, net_bwd_weights_args, eng);
        //-----------------------------------------------------------------------
        //----------------- Weights update -------------------------------------

        std::cout << "Weight update FC1\n";
        updateWeights_SGD(fc1.arg_weights, 
                   fc1_back_weights.arg_diff_weights,
                   learning_rate, net_sgd, net_sgd_args, eng);
        // Copy weights to the inference network
        Reorder(fc1.arg_weights, fc1_inf.arg_weights,
                   net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Weight update FC2\n";
        updateWeights_SGD(fc2.arg_weights, 
                   fc2_back_weights.arg_diff_weights,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc2.arg_weights, fc2_inf.arg_weights,
                   net_bwd_weights, net_bwd_weights_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Bias update -------------------------------------
        std::cout << "Bias update FC1\n";
        updateWeights_SGD(fc1.arg_bias, 
                   fc1_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc1.arg_bias, fc1_inf.arg_bias,
                   net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Bias update FC2\n";
        updateWeights_SGD(fc2.arg_bias, 
                   fc2_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc2.arg_bias, fc2_inf.arg_bias,
                   net_bwd_weights, net_bwd_weights_args, eng);

        // didn't we forget anything?
        assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
        assert(net_bwd_data.size() == net_bwd_data_args.size() && "something is missing");
        assert(net_bwd_weights.size() == net_bwd_weights_args.size() && "something is missing");

        int max_iter = config_file["iterations"]; // number of iterations for training
        int step = config_file["step"];
        int n_iter = 0;

        // Prepare memory that will host loss
        std::vector<float> curr_loss_diff(batch);
        float curr_loss, curr_loss_inf;
        std::vector<float> loss_history((int)max_iter/step);
        std::vector<float> loss_inf_history((int)max_iter/step);

        // Prepare memory that will host weights and biases
        std::vector<float> weight_test(128);
        std::vector<float> weights_fc1_test(n_features * fc1_output_size), diff_weights_fc1_test(n_features * fc1_output_size), diff_dst_test(batch*fc1_output_size);
        std::vector<float> weights_fc2_test(fc1_output_size), diff_weights_fc2_test(fc1_output_size);
        std::vector<float> bias_fc1_test(fc1_output_size), bias_fc2_test(fc2_output_size);
        std::vector<float> diff_bias_fc1_test(fc1_output_size), diff_bias_fc2_test(fc2_output_size);
        
        // Prepare memory that will host src
        std::vector<float> src_test(batch * n_features), dst_test(batch * fc1_output_size);
        std::vector<float> src_test2(batch * fc1_output_size), dst_test2(batch);

        std::vector<float> labels_test(batch);
        
        // Prepare memory that will host final output
        std::vector<float> sigmoid_test2(batch);
        std::vector<float> output_val(samples_val);


        //unsigned long batch_size = batch;
        unsigned long batch_size = max_iter/step;
        const unsigned long loss_dim [] = {batch_size};
        const unsigned long output_val_dim [] = {config_file["samples_val"]};

        // execute
        while (n_iter < max_iter)
        {
                // forward
                std::cout << "Iteration # " << n_iter << "\n";

                std::cout << "Forward pass\n";

                for (size_t i = 0; i < net_fwd.size(); ++i)
                        net_fwd.at(i).execute(s, net_fwd_args.at(i));

                if (n_iter == 0){
                    read_from_dnnl_memory(weights_fc1_test.data(), fc1.arg_weights);
                    read_from_dnnl_memory(bias_fc1_test.data(), fc1.arg_bias);
                    read_from_dnnl_memory(weights_fc2_test.data(), fc2.arg_weights);
                    read_from_dnnl_memory(bias_fc2_test.data(), fc2.arg_bias);
                    s.wait();
                    std::cout << "FC1 Weights (initial):\n";
                    print_vector2(weights_fc1_test);
                    std::cout << "FC1 Bias: (initial)\n";
                    print_vector2(bias_fc1_test);
                    std::cout << "FC2 Weights: (initial)\n";
                    print_vector2(weights_fc2_test);
                    std::cout << "FC2 Bias: (initial)\n";
                    print_vector2(bias_fc2_test);
                }

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

                if (n_iter % step == 0){  
                        // Execute inference
                        for (size_t i = 0; i < net_fwd_inf.size(); ++i)
                                net_fwd_inf.at(i).execute(s, net_fwd_inf_args.at(i));

                        s.wait();
                        read_from_dnnl_memory(&curr_loss, net_fwd_args[loss][DNNL_ARG_DST]);
                        read_from_dnnl_memory(&curr_loss_inf, net_fwd_inf_args[loss_inf][DNNL_ARG_DST]);
                        read_from_dnnl_memory(curr_loss_diff.data(), net_bwd_data_args[loss_back][DNNL_ARG_DST]);
                        read_from_dnnl_memory(weights_fc1_test.data(), fc1.arg_weights);
                        read_from_dnnl_memory(bias_fc1_test.data(), fc1.arg_bias);
                        read_from_dnnl_memory(weights_fc2_test.data(), fc2.arg_weights);
                        read_from_dnnl_memory(bias_fc2_test.data(), fc2.arg_bias);
                        read_from_dnnl_memory(diff_weights_fc1_test.data(), fc1_back_weights.arg_diff_weights);
                        read_from_dnnl_memory(diff_weights_fc2_test.data(), fc2_back_weights.arg_diff_weights);
                        read_from_dnnl_memory(diff_bias_fc1_test.data(), fc1_back_weights.arg_diff_bias);
                        read_from_dnnl_memory(diff_bias_fc2_test.data(), fc2_back_weights.arg_diff_bias);
                        read_from_dnnl_memory(diff_dst_test.data(), fc2_back_data.arg_diff_dst);
                        read_from_dnnl_memory(labels_test.data(), labels_memory);
                        read_from_dnnl_memory(src_test.data(), fc1.arg_src);
                        read_from_dnnl_memory(dst_test.data(), fc1.arg_dst);
                        read_from_dnnl_memory(src_test2.data(), fc2.arg_src);
                        read_from_dnnl_memory(dst_test2.data(), fc2.arg_dst);
                        
                        read_from_dnnl_memory(sigmoid_test2.data(), sigmoid1.arg_dst);
                        
                        s.wait();

                        std::cout << "Loss: " << curr_loss << "\n";
                        std::cout << "Gradient of Loss:\n";
                        print_vector2(curr_loss_diff);
                        std::cout << "FC1 Weights:\n";
                        print_vector2(weights_fc1_test);
                        std::cout << "FC1 Bias:\n";
                        print_vector2(bias_fc1_test);
                        std::cout << "FC2 Weights:\n";
                        print_vector2(weights_fc2_test);
                        std::cout << "FC2 Bias:\n";
                        print_vector2(bias_fc2_test);
                        std::cout << "Gradient of FC1 bias:\n";
                        print_vector2(diff_bias_fc1_test);
                        std::cout << "Gradient of FC2 bias:\n";
                        print_vector2(diff_bias_fc2_test);
                        std::cout << "Gradient of FC1 weights:\n";
                        print_vector2(diff_weights_fc1_test);
                        std::cout << "Gradient of FC2 weights:\n";
                        print_vector2(diff_weights_fc2_test);
                        print_vector2(diff_dst_test);
                        std::cout << "FC1 SRC:\n";
                        print_vector2(labels_test, batch);
                        print_vector2(src_test, batch * dataset_shape[1]);
                        std::cout << "FC1 DST:\n";
                        print_vector2(dst_test);
                        std::cout << "FC2 SRC:\n";
                        print_vector2(src_test2);
                        std::cout << "FC2 DST:\n";
                        print_vector2(dst_test2);
                        std::cout << "Sigmoid DST:\n";
                        print_vector2(sigmoid_test2);
                        

                        loss_history[(int)n_iter/step] = curr_loss;
                        loss_inf_history[(int)n_iter/step] = curr_loss_inf;
                }

                // Change data
                skin_data.write_to_memory(input_memory, labels_memory);

                n_iter++;
        }

        read_from_dnnl_memory(output_val.data(), sigmoid1_inf.arg_dst);

        s.wait();

        std::string loss_filename = config_file["loss_filename"];  
        npy::SaveArrayAsNumpy(loss_filename, false, 1, loss_dim, loss_history);

        std::string loss_inf_filename = config_file["loss_inf_filename"];  
        npy::SaveArrayAsNumpy(loss_inf_filename, false, 1, loss_dim, loss_inf_history);

        std::string val_predicted_filename = config_file["val_predicted_filename"];  
        npy::SaveArrayAsNumpy(val_predicted_filename, false, 1, output_val_dim, output_val);
}

int main(int argc, char **argv)
{
        // Config file
        int extra_args = 1;
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv, extra_args), argc, argv);
}