#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

// Macro definitions to recover L2 loss layer in backward

#define L2_SUB 0
#define L2_LOSS 1

// Macro definition to recover Binary cross entropy loss layer in backward

#define BCE_LOG 0
#define BCE_INV 1
#define BCE_LOG_INV 2
#define BCE_TRUE_INV 3
#define BCE_IP 4
#define BCE_IP_INV 5
#define BCE_SUM 6
#define BCE_NORM 7


using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

// Conv2D, only Glorot initializer implemented
int Conv2D(int batch_size, int patch_length,
           int n_kernels, int kernel_size,
           int stride_length, int padding_length,
           int dilation,
           dnnl::memory input,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    // RNG for ALL purposes
    std::default_random_engine generator;
    std::normal_distribution<float> norm_dist(0.f,1.f);

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

    // Initialize weight and biases 
    for (int i = 0; i<conv_weights.size(); i++){
        conv_weights[i] = norm_dist(generator);
    }

    for (int i = 0; i<conv_bias.size(); i++){
        conv_bias[i] = norm_dist(generator);
    }


    // Write initialized weights and biases on selected engine
    auto conv_user_weights_memory = dnnl::memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
    auto conv_user_bias_memory = dnnl::memory({{conv_bias_tz}, dt::f32, tag::x}, eng);

    write_to_dnnl_memory(conv_weights.data(), conv_user_weights_memory);
    write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);

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

    
    std::cout << "Creating primitive descriptor for convolution\n";
    auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, eng);

    // Check if the types are proper
    std::cout << "Testing types\n";
    auto conv_weights_memory = checkType(conv_pd.weights_desc(), conv_user_weights_memory, net, net_args, eng);
    std::cout << "Weights check OK!\n";
    auto conv_bias_memory = checkType(conv_pd.bias_desc(), conv_user_bias_memory, net, net_args, eng);
    std::cout << "Bias check ok!\n";
    auto conv_src_memory = checkType(conv_pd.src_desc(), input, net, net_args, eng);
    //auto conv_src_memory = input;
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

int Dense(dnnl::memory::dims src_dims, 
          int fc_output_size,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng)
{

    // RNG for ALL purposes
    std::default_random_engine generator;
    std::normal_distribution<float> norm_dist(0.f,1.f);

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

    for (int i = 0; i<fc_weights.size(); i++){
        fc_weights[i] = norm_dist(generator);
    }

    for (int i = 0; i<fc_bias.size(); i++){
        fc_bias[i] = norm_dist(generator);
    }

    // If something does not work check here (oihw?)
    dnnl::memory weights_mem_fc;
    if ( from_conv ){
        weights_mem_fc = dnnl::memory({weights_dims_fc, dt::f32, tag::oihw}, eng);
    }
    else {
        weights_mem_fc = dnnl::memory({weights_dims_fc, dt::f32, tag::oi}, eng);
    }

    write_to_dnnl_memory(fc_bias.data(), bias_mem_fc);
    write_to_dnnl_memory(fc_weights.data(), weights_mem_fc);

    auto weights_md_fc = dnnl::memory::desc(weights_dims_fc, dt::f32, tag::any);

    auto fc_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, src_md_fc,
                                                weights_md_fc, bias_md_fc, dst_md_fc);

    auto fc_pd = dnnl::inner_product_forward::primitive_desc(fc_desc, eng);
    
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

int Eltwise(dnnl::algorithm activation,
          float alpha,
          float beta,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng)
{

    auto src_md = input.get_desc();

    /*dnnl::memory::dims ciao(src_md.dims().size());

    for (int i = 0; i<ciao.size(); i++){
        ciao[i] = src_md.dims()[i];
    }*/

    //auto dst_md = dnnl::memory::desc(ciao, dt::f32, tag::any);

    auto dst_mem = dnnl::memory(src_md, eng);
    auto dst_md =  dst_mem.get_desc();

    std::cout << "Memory allocated\n";

    auto eltwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, activation,
                                                dst_md, alpha, beta);
    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_desc, eng);

    net.push_back(dnnl::eltwise_forward(eltwise_pd));
    net_args.push_back({{DNNL_ARG_SRC, input},
                        {DNNL_ARG_DST, dst_mem}});

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