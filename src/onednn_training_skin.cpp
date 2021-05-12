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
#include "../include/primitive_wrappers.hpp"
#include "../include/rapidcsv.hpp"

using namespace dnnl;

void simple_net(engine::kind engine_kind)
{
        // RNG for ALL purposes
        std::default_random_engine generator;

        // Admission dataset (regression)
        //unsigned long samples = 400;
        // Skin dataset (binary classification)
        unsigned long samples = 245057;
        // Load dataset
        //auto dataset_path = "data/features_admission.txt";
        auto dataset_path = "data/features_skin.txt";
        
        //std::vector<unsigned long> dataset_shape = {samples, 7};      //Admission dataset
        std::vector<unsigned long> dataset_shape = {samples, 3};      //Skin dataset
        
        std::vector<float> dataset(dataset_shape[0]*dataset_shape[1]);

        dataset_shape.clear();
        dataset.clear();

        data_loader(dataset_path, dataset);
                
        std::cout << "\n";
        
        // Load labels
        //auto labels_path = "data/label_admission.txt";
        auto labels_path = "data/label_skin.txt";
        
        std::vector<unsigned long> shape_labels = {samples};
        std::vector<float> dataset_labels(shape_labels[0]);

        dataset_labels.clear();

        //npy::LoadArrayFromNumpy(labels_path, shape_labels, fortran_order, dataset_labels);

        data_loader(labels_path, dataset_labels);

        using tag = memory::format_tag;
        using dt = memory::data_type;

        std::cout << "Starting engine...\n";

        auto eng = engine(engine_kind, 0);
        stream s(eng);

        // Vector of primitives and their execute arguments
        std::vector<primitive> net_fwd, net_bwd_data, net_bwd_weights, net_sgd;
        std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_data_args, net_bwd_weights_args, net_sgd_args;

        const int batch = dataset_shape[0];
        const int patch_size = dataset_shape[1];
        const int n_features = dataset_shape[1];
        const int stride = 1;
        const int kernel_size = 3;
        const int n_kernels = 64;
        const float learning_rate = 0.001;

        // Compute the padding to preserve the same dimension in input and output
        // const int padding = (shape[1] - 1) * stride - shape[1] + kernel_size;
        // padding /= 2;
        int padding = kernel_size - 1;
        padding /= 1;

        // Declare clipping parameters
        float clip_upper = 10000;
        float clip_lower = -10000;

        // Load inputs inside engine
        memory::dims input_dim = {batch, n_features};
        auto input_memory = memory({{input_dim}, dt::f32, tag::nc}, eng);
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


        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        memory::dims fc1_src_dims = {batch, n_features};
        int fc1_output_size = 10;
        int fc1 = Dense(fc1_src_dims, fc1_output_size, 
                        input_memory, net_fwd, net_fwd_args, eng);

        int relu1 = Eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[fc1][DNNL_ARG_DST],
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the first dense layer!\n";

        // PnetCLS: Fully Connected 2
        // {batch, fc1_output_size} -> {batch, 1}

        memory::dims fc2_src_dims = {batch, fc1_output_size};
        int fc2_output_size = 1;
        int fc2 = Dense(fc2_src_dims, fc2_output_size, 
                        net_fwd_args[relu1][DNNL_ARG_DST], net_fwd, net_fwd_args, eng);
                        
        int sigmoid1 = Eltwise(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, net_fwd_args[fc2][DNNL_ARG_DST],
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the second dense layer!\n";

        // L2 loss

        //int loss = L2_Loss(net_fwd_args[sigmoid1][DNNL_ARG_DST], labels_memory, net_fwd, net_fwd_args, eng);
        int loss = binaryCrossEntropyLoss(net_fwd_args[sigmoid1][DNNL_ARG_DST], labels_memory, net_fwd, net_fwd_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Data)-------------------------------------

        std::cout << "Creating backward Loss" << "\n";
        //int loss_back = L2_Loss_back(net_fwd_args, net_fwd_args[loss][DNNL_ARG_DST], net_bwd_data, net_bwd_data_args, eng);
        int loss_back = binaryCrossEntropyLoss_back(net_fwd_args[sigmoid1][DNNL_ARG_DST], labels_memory, batch, net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating clip\n";
        int clip_loss_back = Clip(net_bwd_data_args[loss_back][DNNL_ARG_DST], clip_upper, clip_lower,
                                  net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the second Dense layer (back) using forward index: " << fc2 << "\n"; 
        int sigmoid1_back = Eltwise_back(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, net_fwd_args[sigmoid1], 
                                         net_bwd_data_args[clip_loss_back][DNNL_ARG_DST], net_bwd_data, net_bwd_data_args, eng);
        int clip_sigmoid1_back = Clip(net_bwd_data_args[sigmoid1_back][DNNL_ARG_DIFF_SRC], clip_upper, clip_lower,
                                  net_bwd_data, net_bwd_data_args, eng);
        int fc2_back_data = Dense_back_data(net_bwd_data_args[clip_sigmoid1_back][DNNL_ARG_DST], net_fwd_args[fc2], net_bwd_data, net_bwd_data_args, eng);
        int clip_fc2_back_data = Clip(net_bwd_data_args[fc2_back_data][DNNL_ARG_DIFF_SRC], clip_upper, clip_lower,
                                  net_bwd_data, net_bwd_data_args, eng);
        int relu1_back_data = Eltwise_back(dnnl::algorithm::eltwise_relu, 0.f, 0.f, net_fwd_args[relu1], 
                                         net_bwd_data_args[clip_fc2_back_data][DNNL_ARG_DST], net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the first Dense layer (back) using forward index: " << fc1 << "\n"; 
        int fc1_back_data = Dense_back_data(net_bwd_data_args[relu1_back_data][DNNL_ARG_DIFF_SRC], net_fwd_args[fc1], net_bwd_data, net_bwd_data_args, eng);
        int clip_fc1_back_data = Clip(net_bwd_data_args[fc1_back_data][DNNL_ARG_DIFF_SRC], clip_upper, clip_lower,
                                  net_bwd_data, net_bwd_data_args, eng);
        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Weights)-------------------------------------
        std::cout << "Creating the second Dense layer (back) using forward index: " << fc2 << "\n"; 
        int fc2_back_weights = Dense_back_weights(net_bwd_data_args[clip_sigmoid1_back][DNNL_ARG_DST], net_fwd_args[fc2], net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating Clip\n";
        int clip_fc2_back_weights = Clip(net_bwd_weights_args[fc2_back_weights][DNNL_ARG_DIFF_WEIGHTS], clip_upper, clip_lower,
                                  net_bwd_weights, net_bwd_weights_args, eng);
        int clip_fc2_back_bias = Clip(net_bwd_weights_args[fc2_back_weights][DNNL_ARG_DIFF_BIAS], clip_upper, clip_lower,
                                  net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the first Dense layer (back) using forward index: " << fc1 << "\n"; 
        int fc1_back_weights = Dense_back_weights(net_bwd_data_args[relu1_back_data][DNNL_ARG_DIFF_SRC], net_fwd_args[fc1], net_bwd_weights, net_bwd_weights_args, eng);
        int clip_fc1_back_weights = Clip(net_bwd_weights_args[fc1_back_weights][DNNL_ARG_DIFF_WEIGHTS], clip_upper, clip_lower,
                                  net_bwd_weights, net_bwd_weights_args, eng);
        int clip_fc1_back_bias = Clip(net_bwd_weights_args[fc1_back_weights][DNNL_ARG_DIFF_BIAS], clip_upper, clip_lower,
                                  net_bwd_weights, net_bwd_weights_args, eng);
        //-----------------------------------------------------------------------
        //----------------- Weights update -------------------------------------

        std::cout << "Weight update FC1\n";
        updateWeights_SGD(net_fwd_args[fc1][DNNL_ARG_WEIGHTS], 
                   net_bwd_weights_args[clip_fc1_back_weights][DNNL_ARG_DST],
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Weight update FC2\n";
        updateWeights_SGD(net_fwd_args[fc2][DNNL_ARG_WEIGHTS], 
                   net_bwd_weights_args[clip_fc2_back_weights][DNNL_ARG_DST],
                   learning_rate, net_sgd, net_sgd_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Bias update -------------------------------------
        std::cout << "Bias update FC1\n";
        updateWeights_SGD(net_fwd_args[fc1][DNNL_ARG_BIAS], 
                   net_bwd_weights_args[clip_fc1_back_bias][DNNL_ARG_DST],
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Bias update FC2\n";
        updateWeights_SGD(net_fwd_args[fc2][DNNL_ARG_BIAS], 
                   net_bwd_weights_args[clip_fc2_back_bias][DNNL_ARG_DST],
                   learning_rate, net_sgd, net_sgd_args, eng);

        // didn't we forget anything?
        assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
        assert(net_bwd_data.size() == net_bwd_data_args.size() && "something is missing");
        assert(net_bwd_weights.size() == net_bwd_weights_args.size() && "something is missing");

        int max_iter = 500; // number of iterations for training
        int n_iter = 0;
        int step = 10;

        // Prepare memory that will host loss
        std::vector<float> curr_loss_diff(batch);
        float curr_loss;
        std::vector<float> loss_history((int)max_iter/step);

        // Prepare memory that will host weights and biases
        std::vector<float> weight_test(128);
        std::vector<float> weights_fc1_test(n_features * fc1_output_size), diff_weights_fc1_test(n_features * fc1_output_size), diff_dst_test(batch*fc1_output_size);
        std::vector<float> weights_fc2_test(fc1_output_size), diff_weights_fc2_test(fc1_output_size);
        std::vector<float> bias_fc1_test(fc1_output_size), bias_fc2_test(fc2_output_size);
        
        // Prepare memory that will host src
        std::vector<float> src_test(batch * n_features), dst_test(batch * fc1_output_size);
        std::vector<float> src_test2(batch * fc1_output_size), dst_test2(batch);
        
        // Prepare memory that will host final output
        std::vector<float> sigmoid_test2(batch);

        //unsigned long batch_size = batch;
        unsigned long batch_size = max_iter/step;
        const unsigned long loss_dim [] = {batch_size};

        read_from_dnnl_memory(&curr_loss, net_fwd_args[loss][DNNL_ARG_DST]);
        s.wait();
        //print_vector2(curr_loss);

        // execute
        while (n_iter < max_iter)
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

                if (n_iter % step == 0){  
                        s.wait();
                        read_from_dnnl_memory(&curr_loss, net_fwd_args[loss][DNNL_ARG_DST]);
                        read_from_dnnl_memory(curr_loss_diff.data(), net_bwd_data_args[clip_loss_back][DNNL_ARG_DST]);

                        read_from_dnnl_memory(weights_fc1_test.data(), net_fwd_args[fc1][DNNL_ARG_WEIGHTS]);
                        read_from_dnnl_memory(bias_fc1_test.data(), net_fwd_args[fc1][DNNL_ARG_BIAS]);
                        read_from_dnnl_memory(weights_fc2_test.data(), net_fwd_args[fc2][DNNL_ARG_WEIGHTS]);
                        read_from_dnnl_memory(bias_fc2_test.data(), net_fwd_args[fc2][DNNL_ARG_BIAS]);
                        read_from_dnnl_memory(diff_weights_fc1_test.data(), net_bwd_weights_args[clip_fc1_back_weights][DNNL_ARG_DST]);
                        read_from_dnnl_memory(diff_weights_fc2_test.data(), net_bwd_weights_args[clip_fc2_back_weights][DNNL_ARG_DST]);
                        read_from_dnnl_memory(diff_dst_test.data(), net_bwd_data_args[fc2_back_data][DNNL_ARG_DIFF_DST]);
                        
                        read_from_dnnl_memory(src_test.data(), net_fwd_args[fc1][DNNL_ARG_SRC]);
                        read_from_dnnl_memory(dst_test.data(), net_fwd_args[fc1][DNNL_ARG_DST]);

                        read_from_dnnl_memory(src_test2.data(), net_fwd_args[fc2][DNNL_ARG_SRC]);
                        read_from_dnnl_memory(dst_test2.data(), net_fwd_args[fc2][DNNL_ARG_DST]);
                        
                        read_from_dnnl_memory(sigmoid_test2.data(), net_fwd_args[sigmoid1][DNNL_ARG_DST]);
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
                        std::cout << "Gradient of FC1 weights:\n";
                        print_vector2(diff_weights_fc1_test);
                        std::cout << "Gradient of FC2 weights:\n";
                        print_vector2(diff_weights_fc2_test);
                        print_vector2(diff_dst_test);
                        std::cout << "FC1 SRC:\n";
                        print_vector2(src_test);
                        std::cout << "FC1 DST:\n";
                        print_vector2(dst_test);
                        std::cout << "FC2 SRC:\n";
                        print_vector2(src_test2);
                        std::cout << "FC2 DST:\n";
                        print_vector2(dst_test2);
                        std::cout << "Sigmoid DST:\n";
                        print_vector2(sigmoid_test2);

                        loss_history[(int)n_iter/step] = curr_loss;
                }

                n_iter++;
        }

        //std::string loss_filename = "./data/losses/iteration_" + std::to_string(n_iter) + ".npy";  
        std::string loss_filename = "./data/losses/loss_history.npy";  
        npy::SaveArrayAsNumpy(loss_filename, false, 1, loss_dim, loss_history);

        s.wait();
}

int main(int argc, char **argv)
{
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
