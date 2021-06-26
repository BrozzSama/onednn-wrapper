#include "layers_bwd_weights.h"

Conv2D_back_weights::Conv2D_back_weights(dnnl::memory diff_dst,
           Conv2D conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{

    std::cout << "Allocating memory for backward convolution\n";
    // Create memory area for backward pass (get types from conv2d_fwd)
    auto conv_diff_weights_memory = dnnl::memory(conv2d_fwd.arg_weights.get_desc(), eng);
    auto conv_diff_bias_memory = dnnl::memory(conv2d_fwd.arg_bias.get_desc(), eng);

    std::cout << "Obtaining memory descriptors for backward convolution\n";
    // create memory descriptors for f32 convolution data
    auto conv_bwd_src_md = conv2d_fwd.arg_src.get_desc();
    auto conv_diff_weights_md = conv2d_fwd.arg_weights.get_desc();
    // Get dst descriptor to recreate forward primitive
    auto conv_fwd_dst_md = conv2d_fwd.arg_dst.get_desc();

    auto conv_diff_dst_md = diff_dst.get_desc();
    auto conv_diff_bias_md = conv2d_fwd.arg_bias.get_desc();

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

    conv_bwd_src_memory = checkType(conv_bwd_pd.src_desc(), conv2d_fwd.arg_src, net, net_args, eng);
    auto conv_diff_dst_memory = checkType(conv_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);

    arg_src = conv_bwd_src_memory;
    arg_diff_dst = conv_diff_dst_memory;
    arg_diff_weights = conv_diff_weights_memory;
    arg_diff_bias = conv_diff_bias_memory;

    net.push_back(dnnl::convolution_backward_weights(conv_bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_bwd_src_memory},
                        {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});
}

Dense_back_weights::Dense_back_weights(dnnl::memory diff_dst,
           Dense dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    // INPUT: diff_dst (ie. diff_src of previous layer), src OUTPUT: diff_weights, diff_bias

    // Create memory area for backward pass (get types from dense_fwd)
    auto fc_diff_weights_memory = dnnl::memory(dense_fwd.arg_weights.get_desc(), eng);
    auto fc_diff_bias_memory = dnnl::memory(dense_fwd.arg_bias.get_desc(), eng);


    // create memory descriptors for f32 convolution data
    auto fc_bwd_src_md = dense_fwd.arg_src.get_desc();
    auto fc_diff_weights_md = dense_fwd.arg_weights.get_desc();

    // This is only used to recreate fwd primitive
    auto fc_fwd_dst_md = dense_fwd.arg_dst.get_desc();
    auto fc_diff_dst_md = diff_dst.get_desc();
    auto fc_diff_bias_md = dense_fwd.arg_bias.get_desc();


    std::vector<float> diff_fc_weights(product(fc_diff_weights_md.dims()));
    std::vector<float> diff_fc_bias(product(fc_diff_bias_md.dims()));
    
    std::cout << "Initializing diff weights: \n";
    for (int i = 0; i<diff_fc_weights.size(); i++){
        diff_fc_weights[i] = 0;    
    }
    std::cout << "\n";

    std::cout << "Initializing diff bias: \n";
    for (int i = 0; i<diff_fc_bias.size(); i++){
        diff_fc_bias[i] = 0;    
    }
    std::cout << "\n";

    write_to_dnnl_memory(diff_fc_weights.data(), fc_diff_weights_memory);
    write_to_dnnl_memory(diff_fc_bias.data(), fc_diff_bias_memory);
    
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
    fc_bwd_src_memory = checkType(fc_bwd_pd.src_desc(), dense_fwd.arg_src, net, net_args, eng);
    std::cout << "Checking memory type dst\n";
    std::cout << "The size of net_back is: " << net_args.size() << "\n";

    // Don't forget that this is the actual input
    auto fc_diff_dst_memory = checkType(fc_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);
        
    std::cout << "Adding backward\n";

    if(fc_diff_weights_memory.get_desc() != fc_bwd_pd.diff_weights_desc()){
        std::cout << "Formats are different\n";
    }

    std::cout << "Adding to net\n";

    arg_src = fc_bwd_src_memory;
    arg_diff_dst = fc_diff_dst_memory;
    arg_diff_weights = fc_diff_weights_memory;
    arg_diff_bias = fc_diff_bias_memory;

    net.push_back(dnnl::inner_product_backward_weights(fc_bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, fc_bwd_src_memory},
                        {DNNL_ARG_DIFF_DST, fc_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_DIFF_WEIGHTS, fc_diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, fc_diff_bias_memory}});

}