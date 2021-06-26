#include "layers_bwd_data.h"

Conv2D_back_data::Conv2D_back_data(dnnl::memory diff_dst,
           Conv2D conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{

    auto conv_diff_src_md = conv2d_fwd.arg_src.get_desc();
    auto conv_diff_src_memory = dnnl::memory(conv_diff_src_md, eng);
    
    std::cout << "Allocating memory for backward convolution\n";
    // Create memory area for backward pass (get types from conv2d_fwd)
    auto conv_weights = conv2d_fwd.arg_weights;
    auto conv_weights_md = conv_weights.get_desc();
    
    auto conv_bias_md = conv2d_fwd.arg_bias.get_desc();
    
    std::cout << "Obtaining memory descriptors for backward convolution\n";
    // create memory descriptors for f32 convolution data
    auto conv_bwd_src_md = conv2d_fwd.arg_src.get_desc();
    // Get dst descriptor to recreate forward primitive
    auto conv_fwd_dst_md = conv2d_fwd.arg_dst.get_desc();

    auto conv_diff_dst_md = diff_dst.get_desc();
    
    std::cout << "SRC dims size: " << conv_bwd_src_md.dims().size() << "\n";
    std::cout << "Source vector md content: " << "\n";
    print_vector(conv_bwd_src_md.dims());
    std::cout << "Weights dims size: " << conv_weights_md.dims().size() << "\n";
    std::cout << "Weights vector md content: " << "\n";
    print_vector(conv_weights_md.dims());
    std::cout << "Dst dims size: " << conv_diff_dst_md.dims().size() << "\n";
    std::cout << "Dst vector md content: " << "\n";
    print_vector(conv_diff_dst_md.dims());
    std::cout << "Bias dims size: " << conv_bias_md.dims().size() << "\n";
    std::cout << "Bias vector md content: " << "\n";
    print_vector(conv_bias_md.dims());

    std::cout << "Setting dimensions\n";
    dnnl::memory::dims conv_strides = {stride_length, stride_length};
    dnnl::memory::dims conv_dilates = {dilation, dilation};
    dnnl::memory::dims conv_padding = {padding_length, padding_length};

    // Recreate forward descriptor since it is needed to create the backward primitive descriptor

    std::cout << "Recreating Convolutional layer primitive descriptor\n";
    auto conv_fwd_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
                                                       dnnl::algorithm::convolution_direct, conv_bwd_src_md, conv_weights_md,
                                                       conv_bias_md, conv_fwd_dst_md, conv_strides, conv_dilates, conv_padding,
                                                       conv_padding);

    std::cout << "Creating Convolutional layer primitive descriptor\n";

    auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(conv_fwd_desc, eng);
    
    std::cout << "Creating backward Convolutional layer primitive descriptor\n";
    auto conv_bwd_desc = dnnl::convolution_backward_data::desc(dnnl::algorithm::convolution_direct,
                                                            conv_diff_src_md, conv_weights_md,
                                                            conv_diff_dst_md, conv_strides, conv_dilates, conv_padding, conv_padding);
    
    auto conv_bwd_pd = dnnl::convolution_backward_data::primitive_desc(conv_bwd_desc, eng, conv_fwd_pd);

    std::cout << "Checking diff dst memory type\n";
    auto conv_diff_dst_memory = checkType(conv_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);

    arg_diff_src = conv_diff_src_memory;
    arg_diff_dst = conv_diff_dst_memory;
    arg_weights = conv_weights;

    net.push_back(dnnl::convolution_backward_data(conv_bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_SRC, conv_diff_src_memory},
                        {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_WEIGHTS, conv_weights}});
}


MaxPool2D_back::MaxPool2D_back(int kernel_size, int stride_length, 
           MaxPool2D maxpool_fwd,
           dnnl::memory diff_dst_mem,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    auto src_md = maxpool_fwd.arg_src.get_desc();

    long batch = src_md.dims()[0];
    long channels = src_md.dims()[1];
    long input_height = src_md.dims()[2];
    long input_width = src_md.dims()[3];
    long padding = 0;

    const dnnl::memory::dim output_height = (input_height - ((kernel_size - 1) * 1 + kernel_size) + padding + padding) / stride_length + 1;
    const dnnl::memory::dim output_width = (input_width - ((kernel_size - 1) * 1 + kernel_size) + padding + padding) / stride_length + 1;

    // Source (src) and destination (dst) tensors dimensions.
    dnnl::memory::dims src_dims = {batch, channels, input_height, input_width};
    dnnl::memory::dims dst_dims = {batch, channels, output_height, output_width};

    // Kernel dimensions.
    dnnl::memory::dims kernel_dims = {kernel_size, kernel_size};
    // Strides, padding dimensions.
    dnnl::memory::dims strides_dims = {stride_length, stride_length};
    dnnl::memory::dims padding_dims_l = {padding, padding};
    dnnl::memory::dims padding_dims_r = {padding, padding};
    dnnl::memory::dims dilation = {1, 1};

    auto diff_dst_md = maxpool_fwd.arg_dst.get_desc();
    auto diff_src_md = maxpool_fwd.arg_src.get_desc();
    auto diff_src_mem = dnnl::memory(diff_src_md, eng);
    std::cout << "Memory allocated\n";

    // Create descriptor.
    auto pooling_bwd_desc = dnnl::pooling_v2_backward::desc(dnnl::algorithm::pooling_max,
            diff_src_md, diff_dst_md, strides_dims, kernel_dims,
            dilation, padding_dims_l, padding_dims_r);
    auto pooling_fwd_desc = dnnl::pooling_v2_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max,
            diff_src_md, diff_dst_md, strides_dims, kernel_dims,
            dilation, padding_dims_l, padding_dims_r);
    auto pooling_fwd_pd = dnnl::pooling_v2_forward::primitive_desc(pooling_fwd_desc, eng);
    std::cout << "Created descriptor\n";
    auto pooling_pd = dnnl::pooling_v2_backward::primitive_desc(pooling_bwd_desc, eng, pooling_fwd_pd);
    std::cout << "Created primitive descriptor\n";
    
    arg_diff_src = diff_src_mem;
    arg_diff_dst = diff_dst_mem;

    net.push_back(dnnl::pooling_v2_backward(pooling_pd));
    net_args.push_back({{DNNL_ARG_DIFF_SRC, diff_src_mem},
                        {DNNL_ARG_DIFF_DST, diff_dst_mem},
                        {DNNL_ARG_WORKSPACE, maxpool_fwd.arg_workspace}
                        });
}

Dense_back_data::Dense_back_data(dnnl::memory diff_dst,
           Dense dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng){


    // INPUT: diff_dst, weights, bias OUTPUT: diff_src

    // Create memory area for backward pass (get types from dense_fwd)
    auto fc_diff_src_memory = dnnl::memory(dense_fwd.arg_src.get_desc(), eng);

    // Get inputs from the forward layer
    auto fc_weights = dense_fwd.arg_weights;
    auto fc_weights_md = fc_weights.get_desc();
    
    // This is only used to recreate fwd primitive
    auto fc_fwd_dst_md = dense_fwd.arg_dst.get_desc();
    auto fc_diff_dst_md = diff_dst.get_desc();
    auto fc_bias_md = dense_fwd.arg_bias.get_desc();
    auto fc_diff_src_md = fc_diff_src_memory.get_desc();


    // Initialize diff_src and diff_dst to zero
    std::vector<float> diff_fc_src(product(fc_diff_src_md.dims()));
    
    std::cout << "Initializing diff src: \n";
    for (int i = 0; i<diff_fc_src.size(); i++){
        diff_fc_src[i] = 0;    
    }
    std::cout << "\n";

    write_to_dnnl_memory(diff_fc_src.data(), fc_diff_src_memory);

    // Recreate forward descriptor (see conv2dback)

    std::cout << "Dimensions:\n";
    for(int i=0; i<fc_diff_src_md.dims().size(); i++)
        std::cout << fc_diff_src_md.dims()[i] << " ";
    std::cout << "\n";
    for(int i=0; i<fc_weights_md.dims().size(); i++)
        std::cout << fc_weights_md.dims()[i] << " ";
    std::cout << "\n";
    for(int i=0; i<fc_bias_md.dims().size(); i++)
        std::cout << fc_bias_md.dims()[i] << " ";
    std::cout << "\n";
    for(int i=0; i<fc_fwd_dst_md.dims().size(); i++)
        std::cout << fc_fwd_dst_md.dims()[i] << " ";
    std::cout << "\n";

    auto fc_fwd_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, fc_diff_src_md,
                                                fc_weights_md, fc_bias_md, fc_fwd_dst_md);

    auto fc_fwd_pd = dnnl::inner_product_forward::primitive_desc(fc_fwd_desc, eng);

    std::cout << "Creating inner product data gradient primitive\n";

    auto fc_bwd_desc = dnnl::inner_product_backward_data::desc(fc_diff_src_md, fc_weights_md, 
                                                            fc_diff_dst_md);

    std::cout << "Created inner product data gradient primitive\n";
    
    auto fc_bwd_pd = dnnl::inner_product_backward_data::primitive_desc(fc_bwd_desc, eng, fc_fwd_pd);

    std::cout << "Checking memory type dst\n";
    std::cout << "The size of net_back is: " << net_args.size() << "\n";

    // Don't forget that this is the actual input
    auto fc_diff_dst_memory = checkType(fc_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);
        
    std::cout << "Adding backward\n";

    // Set dnnl::memory pointers inside class
    arg_diff_src = fc_diff_src_memory;
    arg_diff_dst = fc_diff_dst_memory;
    arg_weights = fc_weights;

    net.push_back(dnnl::inner_product_backward_data(fc_bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_SRC, fc_diff_src_memory},
                        // fc_diff_dst_memory, not diff_dst since it might not have passed checkType
                        {DNNL_ARG_DIFF_DST, fc_diff_dst_memory},
                        // If something does not work check this, there might be some
                        // reordering needed done in a similar fashion to cnn_training_f32.cpp
                        {DNNL_ARG_WEIGHTS, fc_weights}});


}

// Only because eltwise has no weights!!!
Eltwise_back::Eltwise_back(dnnl::algorithm activation,
          float alpha,
          float beta,
          Eltwise eltwise_fwd, 
          dnnl::memory diff_dst,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng)
{

    auto diff_dst_md = diff_dst.get_desc();
    //auto diff_src_md = dnnl::memory::desc(diff_dst_md.dims(), dt::f32, tag::any);

    auto diff_src_md = diff_dst_md;

    auto diff_src_mem = dnnl::memory(diff_src_md, eng);

    auto src_mem = eltwise_fwd.arg_src;
    auto src_md = src_mem.get_desc();

    // Recreate forward descriptor for hint
    auto eltwise_fwd_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, activation,
                                                eltwise_fwd.arg_dst.get_desc(), alpha, beta);
    auto eltwise_fwd_pd = dnnl::eltwise_forward::primitive_desc(eltwise_fwd_desc, eng);

    // We use diff_dst_md as diff_data_md because it is an input and the cnn_trainin_f32.cpp examples
    // does the same thing, however there is no clear explanation in the documentation...
    // https://oneapi-src.github.io/oneDNN/structdnnl_1_1eltwise__backward_1_1desc.html

    auto eltwise_bwd_desc = dnnl::eltwise_backward::desc(activation, diff_dst_md, src_md, 
                                                alpha, beta);
                                        
    auto eltwise_bwd_pd = dnnl::eltwise_backward::primitive_desc(eltwise_bwd_desc, eng, eltwise_fwd_pd);

    arg_diff_dst = diff_dst;
    arg_src = src_mem;
    arg_diff_src = diff_src_mem;

    net.push_back(dnnl::eltwise_backward(eltwise_bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_SRC, src_mem},
                        {DNNL_ARG_DIFF_SRC, diff_src_mem}});

}
