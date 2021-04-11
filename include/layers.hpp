#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include "util.hpp"

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;


// Conv2D, only Glorot initializer implemented
int Conv2D(int batch_size, int patch_length,
           int n_kernels, int kernel_size,
           int stride_length, int padding_length,
           int dilation,
           dnnl::algorithm activation,
           dnnl::memory input,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    std::cout << "Creating convolutional layer!\n";
    // N channels = one since we have monochromatic images (WIP)
    //dnnl::memory::dims conv_src_tz = {batch_size, 1, patch_length, patch_length};
    dnnl::memory::dims conv_src_tz = input.get_desc().dims();
    // Get number of "channels" (concept of channel changes after input) from the input
    dnnl::memory::dims conv_weights_tz = {n_kernels, conv_src_tz[1], kernel_size, kernel_size};
    dnnl::memory::dims conv_bias_tz = {n_kernels};
    dnnl::memory::dims conv_dst_tz = {batch_size, n_kernels, patch_length, patch_length};
    dnnl::memory::dims conv_strides = {stride_length, stride_length};
    dnnl::memory::dims conv_dilates = {dilation, dilation};
    dnnl::memory::dims conv_padding = {padding_length, padding_length};

    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));

    // Write initialized weights and biases on selected engine
    auto conv_user_weights_memory = dnnl::memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
    auto conv_user_bias_memory = dnnl::memory({{conv_bias_tz}, dt::f32, tag::x}, eng);

    // Create memory descriptor for input, weights, biases, destination
    auto conv_src_md = dnnl::memory::desc({conv_src_tz}, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::f32, tag::any);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::f32, tag::any);

    // create a (dilated) convolution primitive descriptor
    // The method is overloaded, hence by simply having the correct number of parameters 
    // we are choosing a dilated convolution

    std::cout << "Creating primitive descriptor (ma quello desc non quello dopo) for convolution\n";

    std::cout << "SRC dims size: " << conv_src_md.dims().size() << "\n";
    std::cout << "Source vector md content: " << "\n";
    print_vector(conv_src_md.dims());
    std::cout << "Weights dims size: " << conv_weights_md.dims().size() << "\n";
    std::cout << "Weights vector md content: " << "\n";
    print_vector(conv_weights_md.dims());
    std::cout << "Dst dims size: " << conv_dst_md.dims().size() << "\n";
    std::cout << "Dst vector md content: " << "\n";
    print_vector(conv_dst_md.dims());
    std::cout << "Bias dims size: " << conv_bias_md.dims().size() << "\n";
    std::cout << "Bias vector md content: " << "\n";
    print_vector(conv_bias_md.dims());
    
    
    auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
                                                       dnnl::algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                                       conv_bias_md, conv_dst_md, conv_strides, conv_dilates, conv_padding,
                                                       conv_padding);

    std::cout << "Ho creato il primitive descriptor (ma quello desc non quello dopo) for convolution\n";

    // check if f32 convolution is supported on selected engine
    try
    {
        dnnl::convolution_forward::primitive_desc(conv_desc, eng);
    }
    catch (dnnl::error &e)
    {
        if (e.status == dnnl_unimplemented)
            throw example_allows_unimplemented{
                "No f32 convolution implementation is available for this "
                "platform.\n"
                "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    dnnl::post_ops conv_ops;
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    conv_ops.append_eltwise(scale, activation, alpha, beta);
    dnnl::primitive_attr conv_attr;
    conv_attr.set_post_ops(conv_ops);

    std::cout << "Creating primitive descriptor for convolution\n";

    auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, conv_attr, eng);

    // Check if the types are proper
    std::cout << "Testing types\n";
    auto conv_weights_memory = checkType(conv_pd.weights_desc(), conv_user_weights_memory, net, net_args, eng);
    std::cout << "Weights check OK!\n";
    auto conv_bias_memory = checkType(conv_pd.bias_desc(), conv_user_bias_memory, net, net_args, eng);
    std::cout << "Bias check ok!\n";
    //auto conv_src_memory = checkType(conv_pd.src_desc(), input, net, net_args, eng);
    auto conv_src_memory = input;
    std::cout << "Source check OK!\n";
    std::cout << "Types tested!\n";

    // Create memory for output (no check needed)
    auto conv_dst_memory = dnnl::memory(conv_pd.dst_desc(), eng);

    // Append primitive to network vector
    net.push_back(dnnl::convolution_forward(conv_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_BIAS, conv_bias_memory},
                        {DNNL_ARG_DST, conv_dst_memory}});
    std::cout << "Convolutional layer created, new net args size is: " << net_args.size() << "\n";
    // Return index to locate the layer
    return net.size() - 1;
}

int Conv2D_back(
           std::unordered_map<int, dnnl::memory> conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           dnnl::algorithm activation,
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
    auto conv_diff_dst_md = conv2d_fwd[DNNL_ARG_DST].get_desc();
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
                                                       conv_diff_bias_md, conv_diff_dst_md, conv_strides, conv_dilates, conv_padding,
                                                       conv_padding);
    std::cout << "Settings post-ops\n";
    dnnl::post_ops conv_fwd_ops;
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    conv_fwd_ops.append_eltwise(scale, activation, alpha, beta);
    dnnl::primitive_attr conv_fwd_attr;
    conv_fwd_attr.set_post_ops(conv_fwd_ops);

    std::cout << "Creating Convolutional layer primitive descriptor\n";
    auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(conv_fwd_desc, conv_fwd_attr, eng);

    auto conv_bwd_src_memory = dnnl::memory(conv_bwd_src_md, eng);

    std::cout << "Creating backwrard Convolutional layer primitive descriptor\n";
    auto conv_bwd_desc = dnnl::convolution_backward_weights::desc(dnnl::algorithm::convolution_direct,
                                                            conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                                                            conv_diff_dst_md, conv_strides, conv_dilates, conv_padding, conv_padding);
    
    auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(conv_bwd_desc, eng, conv_fwd_pd);

    conv_bwd_src_memory = checkType(conv_bwd_pd.src_desc(), conv2d_fwd[DNNL_ARG_SRC], net, net_args, eng);
    auto conv_diff_dst_memory = checkType(conv_bwd_pd.diff_dst_desc(), conv2d_fwd[DNNL_ARG_DST], net, net_args, eng);
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

int Dense(dnnl::memory::dims src_dims, 
          int fc_output_size,
          dnnl::algorithm activation,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng)
{

    // 0,1,2,3 are used to grab the dimension we need from the source dims vector
    dnnl::memory::dims weights_dims_fc;
    dnnl::memory::dims bias_dims_fc = {fc_output_size};
    dnnl::memory::dims dst_dims_fc = {src_dims[0], fc_output_size};  

    // check if we need to flatten ie. use the proper tag according to vector size
    // TODO. check what happens with dimension 3 ;)
    bool from_conv = (src_dims.size() > 3);
    
    std::cout << "Time to create some memory descriptors!\n";

    dnnl::memory::desc src_md_fc;
    
    if ( from_conv ){
        src_md_fc = dnnl::memory::desc(src_dims, dt::f32, tag::nchw);
        weights_dims_fc = {fc_output_size, src_dims[1], src_dims[2], src_dims[3]};
    }
    else {
        src_md_fc = dnnl::memory::desc(src_dims, dt::f32, tag::nc);
        weights_dims_fc = {fc_output_size, src_dims[1]};
    }

    std::cout << "Source MD OK!\n";

    auto bias_md_fc = dnnl::memory::desc(bias_dims_fc, dt::f32, tag::a);
    std::cout << "Bias MD OK!\n";
    auto dst_md_fc = dnnl::memory::desc(dst_dims_fc, dt::f32, tag::nc);
    std::cout << "DST MD OK!\n";

    std::cout << "time to allocate some memory!\n";
    auto src_mem_fc = dnnl::memory(src_md_fc, eng);
    std::cout << "Source allocated!\n";
    auto bias_mem_fc = dnnl::memory(bias_md_fc, eng);
    std::cout << "Bias allocated!\n";
    auto dst_mem_fc = dnnl::memory(dst_md_fc, eng);
    std::cout << "Destination allocated!\n";

    // No initialization, will be done in post-op routine
    std::vector<float> fc_weights(product(weights_dims_fc));
    std::vector<float> fc_bias(product(bias_dims_fc));

    // If something does not work check here (oihw?)
    dnnl::memory weights_mem_fc;
    if ( from_conv ){
        weights_mem_fc = dnnl::memory({weights_dims_fc, dt::f32, tag::oihw}, eng);
    }
    else {
        weights_mem_fc = dnnl::memory({weights_dims_fc, dt::f32, tag::oi}, eng);
    }

    auto weights_md_fc = dnnl::memory::desc(weights_dims_fc, dt::f32, tag::any);

    auto fc_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, src_md_fc,
                                                weights_md_fc, bias_md_fc, dst_md_fc);
    
    dnnl::post_ops fc_ops;
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    fc_ops.append_eltwise(scale, activation, alpha, beta);
    dnnl::primitive_attr fc_attr;
    fc_attr.set_post_ops(fc_ops);

    auto fc_pd = dnnl::inner_product_forward::primitive_desc(
        fc_desc, fc_attr, eng);

    // Check if the types are proper
    src_mem_fc = checkType(fc_pd.src_desc(), input, net, net_args, eng);
    weights_mem_fc = checkType(fc_pd.weights_desc(), weights_mem_fc, net, net_args, eng);
    bias_mem_fc = checkType(fc_pd.bias_desc(), bias_mem_fc, net, net_args, eng);

    // Create memory for output (no check needed)
    auto conv_dst_memory = dnnl::memory(fc_pd.dst_desc(), eng);

    // Append primitive to network vector
    net.push_back(dnnl::inner_product_forward(fc_pd));
    net_args.push_back({{DNNL_ARG_SRC, src_mem_fc},
                        {DNNL_ARG_WEIGHTS, weights_mem_fc},
                        {DNNL_ARG_BIAS, bias_mem_fc},
                        {DNNL_ARG_DST, dst_mem_fc}});
    // Return index to locate the layer
    return net.size() - 1;
}

int Dense_back(std::unordered_map<int, dnnl::memory> dense_fwd, 
           dnnl::algorithm activation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    // Create memory area for backward pass (get types from dense_fwd)
    auto fc_diff_weights_memory = dnnl::memory(dense_fwd[DNNL_ARG_WEIGHTS].get_desc(), eng);
    auto fc_diff_bias_memory = dnnl::memory(dense_fwd[DNNL_ARG_BIAS].get_desc(), eng);

    // create memory descriptors for f32 convolution data
    auto fc_bwd_src_md = dense_fwd[DNNL_ARG_SRC].get_desc();
    auto fc_diff_weights_md = dense_fwd[DNNL_ARG_WEIGHTS].get_desc();
    auto fc_diff_dst_md = dense_fwd[DNNL_ARG_DST].get_desc();
    auto fc_diff_bias_md = dense_fwd[DNNL_ARG_BIAS].get_desc();

    // Recreate forward descriptor (see conv2dback)

    auto fc_fwd_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, fc_bwd_src_md,
                                                fc_diff_weights_md, fc_diff_bias_md, fc_diff_dst_md);
    
    dnnl::post_ops fc_fwd_ops;
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    fc_fwd_ops.append_eltwise(scale, activation, alpha, beta);
    dnnl::primitive_attr fc_fwd_attr;
    fc_fwd_attr.set_post_ops(fc_fwd_ops);

    auto fc_fwd_pd = dnnl::inner_product_forward::primitive_desc(
        fc_fwd_desc, fc_fwd_attr, eng);

    std::cout << "Creating inner product weights gradient primitive\n";

    auto fc_bwd_desc = dnnl::inner_product_backward_weights::desc(fc_bwd_src_md, fc_diff_weights_md, fc_diff_bias_md,
                                                            fc_diff_dst_md);

    std::cout << "Created inner product weights gradient primitive\n";
    
    auto fc_bwd_pd = dnnl::inner_product_backward_weights::primitive_desc(fc_bwd_desc, eng, fc_fwd_pd);

    std::cout << "Allocating source memory\n";
    auto fc_bwd_src_memory = dnnl::memory(fc_bwd_src_md, eng);
    std::cout << "Checking memory type src \n";
    fc_bwd_src_memory = checkType(fc_bwd_pd.src_desc(), fc_bwd_src_memory, net, net_args, eng);
    std::cout << "Checking memory type dst\n";
    std::cout << "The size of net_back is: " << net_args.size() << "\n";
    auto fc_diff_dst_memory = checkType(fc_bwd_pd.diff_dst_desc(), dense_fwd[DNNL_ARG_DST], net, net_args, eng);
        
    std::cout << "Adding backward\n";
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

int binaryCrossEntropyLoss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{

    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_nc_tt = {1, y_hat.get_desc().dims()[0]};
    dnnl::memory::dims dim_scalar = {1};

    // Perform elementwise log on y_hat
    
    auto y_hat_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_log = memory(y_hat_log_md, eng);

    auto y_hat_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat.get_desc(), 0.f, 0.f);
    auto y_hat_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_log}});

    // Perform elementwise linear on y_hat with alpha = -1; beta = 1 ie. 1-y_hat

    auto y_hat_inv_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv = memory(y_hat_inv_md, eng);

    auto y_hat_inv_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_hat.get_desc(), -1.f, 1.f);
    auto y_hat_inv_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_inv}});

    // Perform log on previously obtained element ie. log(1-y_hat)

    auto y_hat_inv_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv_log = memory(y_hat_inv_log_md, eng);

    auto y_hat_inv_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat_inv.get_desc(), 0.f, 0.f);
    auto y_hat_inv_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_inv},
                        {DNNL_ARG_DST, y_hat_inv_log}});

    // Reorder y_true from nc to cn {batch, 1} -> {1, batch}

    auto y_true_reordered_md = dnnl::memory::desc(dim_nc_tt, dt::f32, tag::nc);
    auto y_true_reordered = memory(y_true_reordered_md, eng);

    net.push_back(dnnl::reorder(y_true, y_true_reordered));
    net_args.push_back({{DNNL_ARG_FROM, y_true},
                            {DNNL_ARG_TO, y_true_reordered}});

    // Perform elementwise linear on y_true_reordered with alpha = -1; beta = 1 ie. 1-y_true

    auto y_true_inv_reordered_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_true_inv_reordered = memory(y_true_inv_reordered_md, eng);

    auto y_true_inv_reordered_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -1.f, 1.f);
    auto y_true_inv_reordered_pd = dnnl::eltwise_forward::primitive_desc(y_true_inv_reordered_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_true_inv_reordered_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_reordered},
                        {DNNL_ARG_DST, y_true_inv_reordered}});

    // Perform matmul(y_true_reordered, log(y_hat))

    auto add1_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add1 = memory(add1_md, eng);

    auto matmul1_desc = dnnl::matmul::desc(y_true_reordered_md, y_hat_log_md, NULL, add1_md);
    auto matmul1_pd = dnnl::matmul::primitive_desc(matmul1_desc, eng);

    net.push_back(dnnl::matmul(matmul1_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_reordered},
                        {DNNL_ARG_WEIGHTS, y_hat_log_md},
                        {DNNL_ARG_DST, add1}});


    // Perform matmul(1-y_true_reordered, log(1-y_hat))

    auto add2_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add2 = memory(add2_md, eng);

    auto matmul2_desc = dnnl::matmul::desc(y_true_inv_reordered_md, y_hat_inv_log_md, NULL, add2_md);
    auto matmul2_pd = dnnl::matmul::primitive_desc(matmul2_desc, eng);

    net.push_back(dnnl::matmul(matmul2_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_inv_reordered},
                        {DNNL_ARG_WEIGHTS, y_hat_inv_log_md},
                        {DNNL_ARG_DST, add2}});

    // Sum obtained values (should be scalars)

    auto loss_sum_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss_sum = memory(loss_sum_md, eng);

    std::vector<dnnl::memory::desc> add_vector_md = {add1_md, add2_md};
    std::vector<dnnl::memory> add_vector = {add1, add2};
    std::vector<std::float> scales = {1.f, 1.f};

    auto loss_sum_pd = dnnl::sum::primitive_desc(scales, add_vector_md, eng);

    net.push_back(dnnl::sum(loss_sum_pd));

    std::unordered_map<int, memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sum});
    for (int i = 0; i<add_vector.size(); i++){
        sum_args.insert({DNN_ARG_SRC + i, add_vector[i]});
    }

    net_args.push_back(sum_args);
    

    // Divide by -N using elementwise linear

    auto loss_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss = memory(loss_md, eng);

    float batch_size_inv = 1/y_true.dims()[0];

    auto loss_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -batch_size_inv, 0.f);
    auto loss_pd = dnnl::eltwise_forward::primitive_desc(loss_desc, eng);

    net.push_back(dnnl::eltwise_forward(loss_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sum},
                        {DNNL_ARG_DST, loss}});   
    
    return net.size() - 1;

}

int binaryCrossEntropyLoss_back(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{

    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_nc_tt = {1, y_hat.get_desc().dims()[0]};
    dnnl::memory::dims dim_scalar = {1};

    // Perform elementwise log on y_hat
    
    auto y_hat_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_log = memory(y_hat_log_md, eng);

    auto y_hat_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat.get_desc(), 0.f, 0.f);
    auto y_hat_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_log}});

    // Perform elementwise linear on y_hat with alpha = -1; beta = 1 ie. 1-y_hat

    auto y_hat_inv_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv = memory(y_hat_inv_md, eng);

    auto y_hat_inv_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_hat.get_desc(), -1.f, 1.f);
    auto y_hat_inv_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_inv}});

    // Perform log on previously obtained element ie. log(1-y_hat)

    auto y_hat_inv_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv_log = memory(y_hat_inv_log_md, eng);

    auto y_hat_inv_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat_inv.get_desc(), 0.f, 0.f);
    auto y_hat_inv_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_inv},
                        {DNNL_ARG_DST, y_hat_inv_log}});

    // Reorder y_true from nc to cn {batch, 1} -> {1, batch}

    auto y_true_reordered_md = dnnl::memory::desc(dim_nc_tt, dt::f32, tag::nc);
    auto y_true_reordered = memory(y_true_reordered_md, eng);

    net.push_back(dnnl::reorder(y_true, y_true_reordered));
    net_args.push_back({{DNNL_ARG_FROM, y_true},
                            {DNNL_ARG_TO, y_true_reordered}});

    // Perform elementwise linear on y_true_reordered with alpha = -1; beta = 1 ie. 1-y_true

    auto y_true_inv_reordered_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_true_inv_reordered = memory(y_true_inv_reordered_md, eng);

    auto y_true_inv_reordered_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -1.f, 1.f);
    auto y_true_inv_reordered_pd = dnnl::eltwise_forward::primitive_desc(y_true_inv_reordered_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_true_inv_reordered_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_reordered},
                        {DNNL_ARG_DST, y_true_inv_reordered}});

    // Perform matmul(y_true_reordered, log(y_hat))

    auto add1_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add1 = memory(add1_md, eng);

    auto matmul1_desc = dnnl::matmul::desc(y_true_reordered_md, y_hat_log_md, NULL, add1_md);
    auto matmul1_pd = dnnl::matmul::primitive_desc(matmul1_desc, eng);

    net.push_back(dnnl::matmul(matmul1_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_reordered},
                        {DNNL_ARG_WEIGHTS, y_hat_log_md},
                        {DNNL_ARG_DST, add1}});


    // Perform matmul(1-y_true_reordered, log(1-y_hat))

    auto add2_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add2 = memory(add2_md, eng);

    auto matmul2_desc = dnnl::matmul::desc(y_true_inv_reordered_md, y_hat_inv_log_md, NULL, add2_md);
    auto matmul2_pd = dnnl::matmul::primitive_desc(matmul2_desc, eng);

    net.push_back(dnnl::matmul(matmul2_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_inv_reordered},
                        {DNNL_ARG_WEIGHTS, y_hat_inv_log_md},
                        {DNNL_ARG_DST, add2}});

    // Sum obtained values (should be scalars)

    auto loss_sum_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss_sum = memory(loss_sum_md, eng);

    std::vector<dnnl::memory::desc> add_vector_md = {add1_md, add2_md};
    std::vector<dnnl::memory> add_vector = {add1, add2};
    std::vector<std::float> scales = {1.f, 1.f};

    auto loss_sum_pd = dnnl::sum::primitive_desc(scales, add_vector_md, eng);

    net.push_back(dnnl::sum(loss_sum_pd));

    std::unordered_map<int, memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sum});
    for (int i = 0; i<add_vector.size(); i++){
        sum_args.insert({DNN_ARG_SRC + i, add_vector[i]});
    }

    net_args.push_back(sum_args);
    

    // Divide by -N using elementwise linear

    auto loss_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss = memory(loss_md, eng);

    float batch_size_inv = 1/y_true.dims()[0];

    auto loss_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -batch_size_inv, 0.f);
    auto loss_pd = dnnl::eltwise_forward::primitive_desc(loss_desc, eng);

    net.push_back(dnnl::eltwise_forward(loss_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sum},
                        {DNNL_ARG_DST, loss}});   
    
    return net.size() - 1;

}

/*

postOp(layer){

    do glorot initializatoin in user memory

    Do this for every weight/bias couple, otherwise you are not writing anything
    auto conv_user_src_memory = dnnl::memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_dnnl::memory(input.data(), conv_user_src_memory);

}

*/