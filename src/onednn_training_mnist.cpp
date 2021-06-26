/// @copybrief onednn_training_mnist
///
/// @page onednn_training_mnist CNN example using oneDNN wrapper
/// This C++ example demonstrates how to build a simple CNN model
/// made of a Convolutional Layer, Maxpooling layer and a Fully connected layer
/// with one output
///
/// @include onednn_training_mnist.cpp

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

        auto eng = engine(engine_kind, 0);
        stream s(eng);

        // MNIST dataset (binary classification, only images corresponding to 0 and 1 were kept)
        //unsigned long samples = 245057;
        const long samples = config_file["samples"];
        const long batch = config_file["minibatch_size"];
        // Load dataset
        auto dataset_path = config_file["dataset_path"];
        auto labels_path = config_file["labels_path"];

        std::vector<long> dataset_shape = {samples, 28, 28};      //MNIST dataset

        // Data loader 

        DataLoader mnist_data(dataset_path, labels_path, samples, batch, dataset_shape, eng);

        std::cout << "Dataloader instantiated\n";

        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Vector of primitives and their execute arguments
        std::vector<primitive> net_fwd, net_bwd_data, net_bwd_weights, net_sgd;
        std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_data_args, net_bwd_weights_args, net_sgd_args;

        const int patch_size = dataset_shape[1];
        const int stride = 1;
        const int kernel_size = 5;
        const int n_kernels = 24;
        const float learning_rate = config_file["learning_rate"];

        // Compute the padding to preserve the same dimension in input and output
        // const int padding = (shape[1] - 1) * stride - shape[1] + kernel_size;
        // padding /= 2;
        int padding = kernel_size - 1;
        padding /= 1;

        // Declare clipping parameters
        float clip_upper = config_file["clip_upper"];
        float clip_lower = config_file["clip_lower"];

        // Initialize input and write first batch
        memory::dims input_dim = {batch, 1, patch_size, patch_size};
        auto input_memory = memory({{input_dim}, dt::f32, tag::nchw}, eng);

        memory::dims labels_dim = {batch, 1};
        auto labels_memory = memory({{labels_dim}, dt::f32, tag::nc}, eng);

        std::cout << "Writing first batch to memory\n";
        mnist_data.write_to_memory(input_memory, labels_memory);

        std::cout << "Loaded first batch \n";

        // Convolutional layer 1
        Conv2D conv1(batch, patch_size, n_kernels, kernel_size, stride, padding, 1, 
                           input_memory, net_fwd, net_fwd_args, eng);

        Eltwise relu0(dnnl::algorithm::eltwise_relu, 0.f, 0.f, conv1.arg_dst,
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the first convolutional layer!\n";

        // Max pooling
        int pool_stride = 1;
        int pool_kernel = 2;
        MaxPool2D maxpool1(pool_kernel, pool_stride, relu0.arg_dst, net_fwd, net_fwd_args, eng);
        std::cout << "I created the maxpooling layer!\n";

        // Convolutional layer 2
        int n_kernels2 = 64;
        Conv2D conv2(batch, maxpool1.arg_dst.get_desc().dims()[2], n_kernels2, kernel_size, stride, padding, 1, 
                           maxpool1.arg_dst, net_fwd, net_fwd_args, eng);

        Eltwise relu1(dnnl::algorithm::eltwise_relu, 0.f, 0.f, conv2.arg_dst,
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the first convolutional layer!\n";

        // Max pooling
        MaxPool2D maxpool2(pool_kernel, pool_stride, relu1.arg_dst, net_fwd, net_fwd_args, eng);
        std::cout << "I created the maxpooling layer!\n";

        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        int flatten_kernels = maxpool2.arg_dst.get_desc().dims()[1];
        int conv_o_h = maxpool2.arg_dst.get_desc().dims()[2];
        int conv_o_w = maxpool2.arg_dst.get_desc().dims()[3];
        memory::dims fc1_src_dims = {batch, flatten_kernels, conv_o_h, conv_o_w};
        int fc1_output_size = 256;
        Dense fc1(fc1_output_size, maxpool2.arg_dst, net_fwd, net_fwd_args, eng);

        Eltwise relu2(dnnl::algorithm::eltwise_relu, 0.f, 0.f, fc1.arg_dst,
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the first dense layer!\n";

        // PnetCLS: Fully Connected 2
        // {batch, fc1_output_size} -> {batch, 1}

        memory::dims fc2_src_dims = {batch, fc1_output_size};
        int fc2_output_size = 1;
        Dense fc2(fc2_output_size, relu2.arg_dst, net_fwd, net_fwd_args, eng);
                        
        Eltwise sigmoid1(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, fc2.arg_dst,
                            net_fwd, net_fwd_args, eng);

        std::cout << "I created the second dense layer!\n";
        binaryCrossEntropyLoss loss(sigmoid1.arg_dst, labels_memory, net_fwd, net_fwd_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Data)-------------------------------------

        std::cout << "Creating backward Loss" << "\n";
        binaryCrossEntropyLoss_back loss_back(sigmoid1.arg_dst, labels_memory, net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating the second Dense layer (back)\n"; 
        Eltwise_back sigmoid1_back(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, sigmoid1, 
                                         loss_back.arg_dst, net_bwd_data, net_bwd_data_args, eng);
        Dense_back_data fc2_back_data(sigmoid1_back.arg_diff_src, fc2, net_bwd_data, net_bwd_data_args, eng);

        Eltwise_back relu2_back_data(dnnl::algorithm::eltwise_relu, 0.f, 0.f, relu2, 
                                         fc2_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        Dense_back_data fc1_back_data(relu2_back_data.arg_diff_src, fc1, net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating maxpool (back)\n"; 
        MaxPool2D_back maxpool2_back_data(pool_kernel, pool_stride, maxpool1, fc1_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        Eltwise_back relu1_back_data(dnnl::algorithm::eltwise_relu, 0.f, 0.f, relu1, 
                                         maxpool2_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        Conv2D_back_data conv2_back_data(relu1_back_data.arg_diff_src, conv2, stride, padding, 1, 
                                               net_bwd_data, net_bwd_data_args, eng);
        std::cout << "Creating maxpool (back)\n"; 
        MaxPool2D_back maxpool1_back_data(pool_kernel, pool_stride, maxpool1, conv2_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        Eltwise_back relu0_back_data(dnnl::algorithm::eltwise_relu, 0.f, 0.f, relu0, 
                                         maxpool1_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        Conv2D_back_data conv1_back_data(relu0_back_data.arg_diff_src, conv1, stride, padding, 1, 
                                               net_bwd_data, net_bwd_data_args, eng);

        
        //-----------------------------------------------------------------------
        //----------------- Backpropagation Stream  (Weights)-------------------------------------
        std::cout << "Creating the second Dense layer (back weights)\n"; 
        Dense_back_weights fc2_back_weights(sigmoid1_back.arg_diff_src, fc2, net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the first Dense layer (back weights)\n"; 
        Dense_back_weights fc1_back_weights(relu2_back_data.arg_diff_src, fc1, net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the second convolutional layer (back weights)\n"; 
        Conv2D_back_weights conv2_back_weights(relu1_back_data.arg_diff_src , conv2, stride, padding, 1,
                                  net_bwd_weights, net_bwd_weights_args, eng);
        std::cout << "Creating the first convolutional layer (back weights)\n"; 
        Conv2D_back_weights conv1_back_weights(relu0_back_data.arg_diff_src , conv1, stride, padding, 1,
                                  net_bwd_weights, net_bwd_weights_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Weights update -------------------------------------

        std::cout << "Weight update conv1\n";
        updateWeights_SGD(conv1.arg_weights, 
                   conv1_back_weights.arg_diff_weights, 
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Weight update conv1\n";
        updateWeights_SGD(conv2.arg_weights, 
                   conv2_back_weights.arg_diff_weights, 
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Weight update FC1\n";
        updateWeights_SGD(fc1.arg_weights, 
                   fc1_back_weights.arg_diff_weights,
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Weight update FC2\n";
        updateWeights_SGD(fc2.arg_weights, 
                   fc2_back_weights.arg_diff_weights,
                   learning_rate, net_sgd, net_sgd_args, eng);

        //-----------------------------------------------------------------------
        //----------------- Bias update -------------------------------------
        std::cout << "Bias update conv1\n";
        updateWeights_SGD(conv1.arg_bias, 
                  conv1_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Bias update conv1\n";
        updateWeights_SGD(conv2.arg_bias, 
                  conv2_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Bias update FC1\n";
        updateWeights_SGD(fc1.arg_bias, 
                   fc1_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        std::cout << "Bias update FC2\n";
        updateWeights_SGD(fc2.arg_bias, 
                   fc2_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);

        // didn't we forget anything?
        assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
        assert(net_bwd_data.size() == net_bwd_data_args.size() && "something is missing");
        assert(net_bwd_weights.size() == net_bwd_weights_args.size() && "something is missing");

        int max_iter = config_file["iterations"]; // number of iterations for training
        int n_iter = 0;
        int step = config_file["step"];

        // Prepare memory that will host loss
        std::vector<float> curr_loss_diff(batch);
        float curr_loss;
        std::vector<float> loss_history((int)max_iter/step);
        
        //unsigned long batch_size = batch;
        unsigned long batch_size = max_iter/step;
        const unsigned long loss_dim [] = {batch_size};

        //s.wait();
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
                        read_from_dnnl_memory(&curr_loss, loss.arg_dst);                        
                        loss_history[(int)n_iter/step] = curr_loss;
                }

                // Change data
                mnist_data.write_to_memory(input_memory, labels_memory);

                n_iter++;
        }

        //std::string loss_filename = "./data/losses/iteration_" + std::to_string(n_iter) + ".npy";  
        std::string loss_filename = config_file["loss_filename"];  
        npy::SaveArrayAsNumpy(loss_filename, false, 1, loss_dim, loss_history);

        s.wait();
}

int main(int argc, char **argv)
{
        // Config file
        int extra_args = 1;
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv, extra_args), argc, argv);
}

