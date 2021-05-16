#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

int Conv2D_back_weights(dnnl::memory diff_dst,
           std::unordered_map<int, dnnl::memory> conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{

    std::cout << "Allocating memory for backward convolution\n";
    // Create memory area for backward pass (get types from conv2d_fwd)
    auto conv_diff_weights_memory = dnnl::memory(conv2d_fwd[DNNL_ARG_WEIGHTS].get_desc(), eng);
    auto conv_diff_bias_memory = dnnl::memory(conv2d_fwd[DNNL_ARG_BIAS].get_desc(), eng);

    std::cout << "Obtaining memory descriptors for backward convolution\n";
    // create memory descriptors for f32 convolution data
    auto conv_bwd_src_md = conv2d_fwd[DNNL_ARG_SRC].get_desc();
    auto conv_diff_weights_md = conv2d_fwd[DNNL_ARG_WEIGHTS].get_desc();
    // Get dst descriptor to recreate forward primitive
    auto conv_fwd_dst_md = conv2d_fwd[DNNL_ARG_DST].get_desc();

    auto conv_diff_dst_md = diff_dst.get_desc();
    auto conv_diff_bias_md = conv2d_fwd[DNNL_ARG_BIAS].get_desc();

    std::cout << "SRC dims size: " << conv_bwd_src_md.dims().size() << "\n";
    std::cout << "Source vector md content: " << "\n";
    print_vector(conv_bwd_src_md.dims());
    std::cout << "Weights dims size: " << conv_diff_weights_md.dims().size() << "\n";
    std::cout << "Weights vector md content: " << "\n";
    print_vector(conv_diff_weights_md.dims());
    std::cout << "Dst dims size: " << conv_diff_dst_md.dims().size() << "\n";
    std::cout << "Dst vector md content: " << "\n";
    print_vector(conv_diff_dst_md.dims());
    std::cout << "Bias dims size: " << conv_diff_bias_md.dims().size() << "\n";
    std::cout << "Bias vector md content: " << "\n";
    print_vector(conv_diff_bias_md.dims());

    std::cout << "Setting dimensions\n";
    dnnl::memory::dims conv_strides = {stride_length, stride_length};
    dnnl::memory::dims conv_dilates = {dilation, dilation};
    dnnl::memory::dims conv_padding = {padding_length, padding_length};

    // Recreate forward descriptor since it is needed to create the backward primitive descriptor

    std::cout << "Recreating Convolutional layer primitive descriptor\n";
    auto conv_fwd_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
                                                       dnnl::algorithm::convolution_direct, conv_bwd_src_md, conv_diff_weights_md,
                                                       conv_diff_bias_md, conv_fwd_dst_md, conv_strides, conv_dilates, conv_padding,
                                                       conv_padding);
    std::cout << "Settings post-ops\n";

    auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(conv_fwd_desc, eng);


    auto conv_bwd_src_memory = dnnl::memory(conv_bwd_src_md, eng);

    std::cout << "Creating backwrard Convolutional layer primitive descriptor\n";
    auto conv_bwd_desc = dnnl::convolution_backward_weights::desc(dnnl::algorithm::convolution_direct,
                                                            conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                                                            conv_diff_dst_md, conv_strides, conv_dilates, conv_padding, conv_padding);
    
    auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(conv_bwd_desc, eng, conv_fwd_pd);

    conv_bwd_src_memory = checkType(conv_bwd_pd.src_desc(), conv2d_fwd[DNNL_ARG_SRC], net, net_args, eng);
    auto conv_diff_dst_memory = checkType(conv_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);
    net.push_back(dnnl::convolution_backward_weights(conv_bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_bwd_src_memory},
                        {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});
    
    // Return index to locate the layer
    return net.size() - 1;
}

int Dense_back_weights(dnnl::memory diff_dst,
           std::unordered_map<int, dnnl::memory> dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    // INPUT: diff_dst (ie. diff_src of previous layer), src OUTPUT: diff_weights, diff_bias

    // Create memory area for backward pass (get types from dense_fwd)
    auto fc_diff_weights_memory = dnnl::memory(dense_fwd[DNNL_ARG_WEIGHTS].get_desc(), eng);
    auto fc_diff_bias_memory = dnnl::memory(dense_fwd[DNNL_ARG_BIAS].get_desc(), eng);

    // create memory descriptors for f32 convolution data
    auto fc_bwd_src_md = dense_fwd[DNNL_ARG_SRC].get_desc();
    auto fc_diff_weights_md = dense_fwd[DNNL_ARG_WEIGHTS].get_desc();

    // This is only used to recreate fwd primitive
    auto fc_fwd_dst_md = dense_fwd[DNNL_ARG_DST].get_desc();
    auto fc_diff_dst_md = diff_dst.get_desc();
    auto fc_diff_bias_md = dense_fwd[DNNL_ARG_BIAS].get_desc();

    // Recreate forward descriptor (see conv2dback)

    auto fc_fwd_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, fc_bwd_src_md,
                                                fc_diff_weights_md, fc_diff_bias_md, fc_fwd_dst_md);
    
    std::cout << "Creating inner product weights gradient primitive\n";

    auto fc_fwd_pd = dnnl::inner_product_forward::primitive_desc(fc_fwd_desc, eng);

    auto fc_bwd_desc = dnnl::inner_product_backward_weights::desc(fc_bwd_src_md, fc_diff_weights_md, fc_diff_bias_md,
                                                            fc_diff_dst_md);

    std::cout << "Created inner product weights gradient primitive\n";
    
    auto fc_bwd_pd = dnnl::inner_product_backward_weights::primitive_desc(fc_bwd_desc, eng, fc_fwd_pd);

    std::cout << "Allocating source memory\n";
    auto fc_bwd_src_memory = dnnl::memory(fc_bwd_src_md, eng);
    std::cout << "Checking memory type src \n";
    fc_bwd_src_memory = checkType(fc_bwd_pd.src_desc(), dense_fwd[DNNL_ARG_SRC], net, net_args, eng);
    std::cout << "Checking memory type dst\n";
    std::cout << "The size of net_back is: " << net_args.size() << "\n";

    // Don't forget that this is the actual input
    auto fc_diff_dst_memory = checkType(fc_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);
        
    std::cout << "Adding backward\n";

    if(fc_diff_weights_memory.get_desc() != fc_bwd_pd.diff_weights_desc()){
        std::cout << "Formats are different\n";
    }

    std::cout << "Adding to net\n";
    net.push_back(dnnl::inner_product_backward_weights(fc_bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, fc_bwd_src_memory},
                        {DNNL_ARG_DIFF_DST, fc_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_DIFF_WEIGHTS, fc_diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, fc_diff_bias_memory}});
    
    // Return index to locate the layer
    return net.size() - 1;
}