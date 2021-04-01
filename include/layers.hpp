#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include "util.hpp"

// Conv2D, only Glorot initializer implemented

int Conv2D(std::int batch_size, std::int patch_length,
           std::int n_kernels, std::int kernel_size,
           std::int stride_length, std::int padding_length,
           std::int dilation,
           dnnl::algorithm activation,
           dnnl::memory input,
           std::vector<primitive> net,
           std::vector<std::unordered_map<int, memory>> net_args,
           dnnl::engine eng)
{

    std::default_random_engine generator;

    // N channels = one since we have monochromatic images (WIP)
    memory::dims conv_src_tz = {batch_size, 1, patch_length, patch_length};
    memory::dims conv_weights_tz = {n_kernels, 1, kernel_size, kernel_size};
    memory::dims conv_bias_tz = {n_kernels};
    memory::dims conv_dst_tz = {batch_size, n_kernels, patch_length, patch_length};
    memory::dims conv_strides = {stride_length, stride_length};
    memory::dims conv_padding = {padding_length, padding_length};

    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));

    // Glorot initialization using the previously defined RNG
    int limit = sqrt(6 / (2 * kernel_size * kernel_size))
        std::uniform_real_distribution<double>
            distribution_conv1(-limit, limit);

    for (size_t i = 0; i < conv_weights.size(); ++i)
        conv_weights[i] = distribution_conv1(generator);
    for (size_t i = 0; i < conv_bias.size(); ++i)
        conv_bias[i] = distribution_conv1(generator);

    // Write initialized weights and biases on selected engine
    auto conv_user_weights_memory = memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv_weights.data(), conv_user_weights_memory);

    auto conv_user_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);

    // Create memory descriptor for input, weights, biases, destination
    auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

    // create a (dilated) convolution primitive descriptor
    auto conv_desc = dilated_convolution_forward::desc(prop_kind::forward,
                                                       algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                                       conv_bias_md, conv_dst_md, conv_strides, dilation, conv_padding,
                                                       conv_padding);

    // check if f32 convolution is supported on selected engine
    try
    {
        dilated_convolution_forward::primitive_desc(conv_desc, eng);
    }
    catch (error &e)
    {
        if (e.status == dnnl_unimplemented)
            throw example_allows_unimplemented{
                "No f32 convolution implementation is available for this "
                "platform.\n"
                "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    auto conv_pd = dilated_convolution_forward::primitive_desc(conv_desc, eng);

    // Check if the types are proper
    auto conv_src_memory = checkType(conv_pd.src_desc(), input, net, net_args, eng);
    auto conv_weights_memory = checkType(conv_pd.weights_desc(), conv_user_weights_memory, net, net_args, eng);
    auto conv_bias_memory = checkType(conv_pd.bias_desc(), conv_user_bias_memory, net, net_args, eng);

    // Create memory for output (no check needed)
    auto conv_dst_memory = memory(conv_pd.dst_desc(), eng);

    // Append primitive to network vector
    net.push_back(dilated_convolution_forward(conv_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_BIAS, conv_bias_memory},
                        {DNNL_ARG_DST, conv_dst_memory}});
    // Return index to locate the layer
    return net.size() - 1;
}

int Conv2D_back(
           std::unordered_map<int, memory> conv2d_fwd, 
           std::vector<primitive> net,
           std::vector<std::unordered_map<int, memory>> net_args,
           dnnl::engine eng)
{
    // Create memory area for backward pass (get types from conv2d_fwd)
    auto conv_user_diff_weights_memory = memory({{conv_weights_tz}, dt::f32, tag::nchw}, eng);
    auto conv_diff_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);

    // create memory descriptors for f32 convolution data
    auto conv_bwd_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
    auto conv_diff_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
    auto conv_diff_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);
    auto conv_diff_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::any);

    auto conv_bwd_desc = convolution_backward_weights::desc(algorithm::convolution_direct,
                                                                conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                                                                conv_diff_dst_md, conv_strides, conv_padding, conv_padding);
    
    auto conv_bwd_pd = convolution_backward_weights::primitive_desc(conv_bwd_desc, eng, conv_pd);

    auto conv_bwd_src_memory = checkType(conv_bwd_pd.src_desc(), conv_src_memory, net, net_args, eng);
    auto conv_diff_dst_memory = checkType(conv_bwd_pd.diff_dst_desc() ,net_args.back()[DNNL_ARG_TO], net, net_args, eng);
    net.push_back(convolution_backward_weights(conv_bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_bwd_src_memory},
                        {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});
    
    
}