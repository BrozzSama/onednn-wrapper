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

using namespace dnnl;

void simple_net(engine::kind engine_kind)
{
        // RNG for ALL purposes
        std::default_random_engine generator;

        auto path = "data/skull_32_vessel.npy";
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<double> net_src;

        shape.clear();
        data.clear();

        npy::LoadArrayFromNumpy(path, shape, fortran_order, net_src);

        std::cout << "shape: ";
        for (size_t i = 0; i < shape.size(); i++)
                std::cout << shape[i] << ", ";
        std::cout << "\n";

        using tag = memory::format_tag;
        using dt = memory::data_type;

        auto eng = engine(engine_kind, 0);
        stream s(eng);

        // Vector of primitives and their execute arguments
        std::vector<primitive> net_fwd, net_bwd;
        std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;

        const int batch = shape[0];
        const int patch_size = shape[1];
        const int stride = 1;
        const int kernel_size = 3;
        // Compute the padding to preserve the same dimension in input and output
        const int padding = (shape[1] - 1) * stride - shape[1] + kernel_size;
        padding /= 2;

        // pnetcls: conv
        // {batch, 1, 32, 32} (x) {64, 1, 3, 3} -> {batch, 96, 55, 55}
        // strides: {4, 4}

        // N channels is one since we have monochromatic images
        memory::dims conv_src_tz = {batch, 1, patch_size, patch_size};
        memory::dims conv_weights_tz = {64, 1, kernel_size, kernel_size};
        memory::dims conv_bias_tz = {64};
        memory::dims conv_dst_tz = {batch, 64, patch_size, patch_size};
        memory::dims conv_strides = {stride, stride};
        memory::dims conv_padding = {padding, padding};

        // float data type is used for user data
        std::vector<float> conv_weights(product(conv_weights_tz));
        std::vector<float> conv_bias(product(conv_bias_tz));

        int limit = sqrt(6 / (2 * kernel_size * kernel_size))
            std::uniform_real_distribution<double>
                distribution_conv1(-limit, limit);

        // initializing non-zero values for weights and bias
        for (size_t i = 0; i < conv_weights.size(); ++i)
                conv_weights[i] = distribution_conv1(generator);
        for (size_t i = 0; i < conv_bias.size(); ++i)
                conv_bias[i] = distribution_conv1(generator);

        // create memory for user data
        auto conv_user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
        write_to_dnnl_memory(net_src.data(), conv_user_src_memory);

        auto conv_user_weights_memory = memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv_weights.data(), conv_user_weights_memory);

        auto conv_user_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);

        // create memory descriptors for bfloat16 convolution data w/ no specified
        // format tag(`any`)
        // tag `any` lets a primitive(convolution in this case)
        // chose the memory format preferred for best performance.
        auto conv_src_md = memory::desc({conv_src_tz}, dt::bf16, tag::any);
        auto conv_weights_md = memory::desc({conv_weights_tz}, dt::bf16, tag::any);
        auto conv_dst_md = memory::desc({conv_dst_tz}, dt::bf16, tag::any);
        // here bias data type is set to bf16.
        // additionally, f32 data type is supported for bf16 convolution.
        auto conv_bias_md = memory::desc({conv_bias_tz}, dt::bf16, tag::any);

        int dilation = 1;

        // create a convolution primitive descriptor
        auto conv_desc = dilated_convolution_forward::desc(prop_kind::forward,
                                                           algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                                           conv_bias_md, conv_dst_md, conv_strides, dilation, conv_padding,
                                                           conv_padding);

        // check if bf16 convolution is supported
        try
        {
                dilated_convolution_forward::primitive_desc(conv_desc, eng);
        }
        catch (error &e)
        {
                if (e.status == dnnl_unimplemented)
                        throw example_allows_unimplemented{
                            "No bf16 convolution implementation is available for this "
                            "platform.\n"
                            "Please refer to the developer guide for details."};

                // on any other error just re-throw
                throw;
        }

        auto conv_pd = dilated_convolution_forward::primitive_desc(conv_desc, eng);

        // create reorder primitives between user input and conv src if needed
        auto conv_src_memory = conv_user_src_memory;
        if (conv_pd.src_desc() != conv_user_src_memory.get_desc())
        {
                conv_src_memory = memory(conv_pd.src_desc(), eng);
                net_fwd.push_back(reorder(conv_user_src_memory, conv_src_memory));
                net_fwd_args.push_back({{DNNL_ARG_FROM, conv_user_src_memory},
                                        {DNNL_ARG_TO, conv_src_memory}});
        }

        auto conv_weights_memory = conv_user_weights_memory;
        if (conv_pd.weights_desc() != conv_user_weights_memory.get_desc())
        {
                conv_weights_memory = memory(conv_pd.weights_desc(), eng);
                net_fwd.push_back(
                    reorder(conv_user_weights_memory, conv_weights_memory));
                net_fwd_args.push_back({{DNNL_ARG_FROM, conv_user_weights_memory},
                                        {DNNL_ARG_TO, conv_weights_memory}});
        }

        // convert bias from f32 to bf16 as convolution descriptor is created with
        // bias data type as bf16.
        auto conv_bias_memory = conv_user_bias_memory;
        if (conv_pd.bias_desc() != conv_user_bias_memory.get_desc())
        {
                conv_bias_memory = memory(conv_pd.bias_desc(), eng);
                net_fwd.push_back(reorder(conv_user_bias_memory, conv_bias_memory));
                net_fwd_args.push_back({{DNNL_ARG_FROM, conv_user_bias_memory},
                                        {DNNL_ARG_TO, conv_bias_memory}});
        }

        // create memory for conv dst
        auto conv_dst_memory = memory(conv_pd.dst_desc(), eng);

        // finally create a convolution primitive
        net_fwd.push_back(dilated_convolution_forward(conv_pd));
        net_fwd_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
                                {DNNL_ARG_WEIGHTS, conv_weights_memory},
                                {DNNL_ARG_BIAS, conv_bias_memory},
                                {DNNL_ARG_DST, conv_dst_memory}});

        // PnetCLS: relu
        // {batch, 64, patch_size, patch_size} -> {batch, 64, patch_size, patch_size}
        memory::dims relu_data_tz = {batch, 64, patch_size, patch_size};
        const float negative_slope = 0.0f;

        // create relu primitive desc
        // keep memory format tag of source same as the format tag of convolution
        // output in order to avoid reorder
        auto relu_desc = eltwise_forward::desc(prop_kind::forward,
                                               algorithm::eltwise_relu, conv_pd.dst_desc(), negative_slope);
        auto relu_pd = eltwise_forward::primitive_desc(relu_desc, eng);

        // create relu dst memory
        auto relu_dst_memory = memory(relu_pd.dst_desc(), eng);

        // finally create a relu primitive
        net_fwd.push_back(eltwise_forward(relu_pd));
        net_fwd_args.push_back(
            {{DNNL_ARG_SRC, conv_dst_memory}, {DNNL_ARG_DST, relu_dst_memory}});

        // PnetCLS: Fully Connected 1
        // {batch, 64, patch_size, patch_size} -> {batch, fc1_output_size}

        // Create memory descriptors and memory objects for src and dst. In this
        // example, NCHW layout is assumed.

        std::int fc1_output_size = 128;
        memory::dims src_dims_fc1 = {batch, 64, patch_size, patch_size};
        memory::dims weights_dims_fc1 = {fc1_output_size, 64, patch_size, patch_size};
        memory::dims bias_dims_fc1 = {fc1_output_size};
        memory::dims dst_dims_fc1 = {batch, fc1_output_size};

        auto src_md_fc1 = memory::desc(src_dims_fc1, dt::f32, tag::nchw);
        auto bias_md_fc1 = memory::desc(bias_dims_fc1, dt::f32, tag::a);
        auto dst_md_fc1 = memory::desc(dst_dims_fc1, dt::f32, tag::nc);
        auto src_mem_fc1 = memory(src_md_fc1, eng);
        auto bias_mem_fc1 = memory(bias_md_fc1, eng);
        auto dst_mem_fc1 = memory(dst_md_fc1, eng);

        // float data type is used for user data
        std::vector<float> fc1_weights(product(weights_dims_fc1));
        std::vector<float> fc1_bias(product(bias_dims_fc1));

        int limit = sqrt(6 / (kernel_size * kernel_size + fc1_output_size))
            std::uniform_real_distribution<double>
                distribution_fc1(-limit, limit);

        // initializing non-zero values for weights and bias
        for (size_t i = 0; i < conv_weights.size(); ++i)
                fc1_weights[i] = distribution_fc1(generator);
        for (size_t i = 0; i < conv_bias.size(); ++i)
                fc1_bias[i] = distribution_fc1(generator);

        // Create memory object for user's layout for weights. In this example, OIHW
        // is assumed.
        auto user_weights_mem_fc1 = memory({weights_dims_fc1, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(fc1_bias.data(), bias_mem_fc1);
        write_to_dnnl_memory(fc1_weights.data(), user_weights_mem_fc1);

        auto weights_md_fc1 = memory::desc(weights_dims_fc1, dt::f32, tag::any);

        auto fc1_desc = inner_product_forward::desc(prop_kind::forward_training, src_md_fc1,
                                                    weights_md_fc1, bias_md_fc1, dst_md_fc1);

        // Create primitive post-ops (ReLU).
        const float scale = 1.0f;
        const float alpha = 0.f;
        const float beta = 0.f;

        post_ops fc1_ops;
        fc1_ops.append_eltwise(
            scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr fc1_attr;
        fc1_attr.set_post_ops(fc1_ops);

        auto fc1_pd = inner_product_forward::primitive_desc(
            fc1_desc, fc1_attr, engine);
        
        // Create the primitive.
        auto fc1_prim = inner_product_forward(fc1_pd);
        // Primitive arguments.
        std::unordered_map<int, memory> fc1_args;
        fc1_args.insert({DNNL_ARG_SRC, relu_dst_memory});
        fc1_args.insert({DNNL_ARG_WEIGHTS, user_weights_mem_fc1});
        fc1_args.insert({DNNL_ARG_BIAS, bias_mem_fc1});
        fc1_args.insert({DNNL_ARG_DST, dst_mem_fc1});

        // PnetCLS: Fully Connected 2
        // {batch, 64, patch_size, patch_size} -> {batch, fc2_output_size}

        // Create memory descriptors and memory objects for src and dst. In this
        // example, NCHW layout is assumed.

        std::int fc2_output_size = 1;
        memory::dims src_dims_fc2 = {batch, fc1_output_size};
        memory::dims weights_dims_fc2 = {fc2_output_size, fc1_output_size};
        memory::dims bias_dims_fc2 = {fc2_output_size};
        memory::dims dst_dims_fc2 = {batch, fc2_output_size};

        auto src_md_fc2 = memory::desc(src_dims_fc2, dt::f32, tag::nchw);
        auto bias_md_fc2 = memory::desc(bias_dims_fc2, dt::f32, tag::a);
        auto dst_md_fc2 = memory::desc(dst_dims_fc2, dt::f32, tag::nc);
        auto src_mem_fc2 = memory(src_md_fc2, eng);
        auto bias_mem_fc2 = memory(bias_md_fc2, eng);
        auto dst_mem_fc2 = memory(dst_md_fc2, eng);

        // float data type is used for user data
        std::vector<float> fc2_weights(product(weights_dims_fc2));
        std::vector<float> fc2_bias(product(bias_dims_fc2));

        int limit = sqrt(6 / (fc1_output_size + fc2_output_size))
        std::uniform_real_distribution<double> distribution_fc2(-limit, limit);

        // initializing non-zero values for weights and bias
        for (size_t i = 0; i < conv_weights.size(); ++i)
                fc2_weights[i] = distribution_fc2(generator);
        for (size_t i = 0; i < conv_bias.size(); ++i)
                fc2_bias[i] = distribution_fc2(generator);

        // Create memory object for user's layout for weights. In this example, OIHW
        // is assumed.
        auto user_weights_mem_fc2 = memory({weights_dims_fc2, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(fc2_bias.data(), bias_mem_fc2);
        write_to_dnnl_memory(fc2_weights.data(), user_weights_mem_fc2);

        auto weights_md_fc2 = memory::desc(weights_dims_fc2, dt::f32, tag::any);

        auto fc2_desc = inner_product_forward::desc(prop_kind::forward_training, src_md_fc2,
                                                    weights_md_fc2, bias_md_fc2, dst_md_fc2);

        // Create primitive post-ops (Sigmoid).
        //const float scale = 1.0f;
        //const float alpha = 0.f;
        //const float beta = 0.f;

        post_ops fc2_ops;
        fc2_ops.append_eltwise(
            scale, algorithm::eltwise_logistic, alpha, beta);
        primitive_attr fc2_attr;
        fc2_attr.set_post_ops(fc2_ops);

        auto fc2_pd = inner_product_forward::primitive_desc(
            fc2_desc, fc2_attr, engine);

        // Create the primitive.
        auto fc2_prim = inner_product_forward(fc2_pd);
        // Primitive arguments.
        std::unordered_map<int, memory> fc2_args;
        fc2_args.insert({DNNL_ARG_SRC, dst_mem_fc1});
        fc2_args.insert({DNNL_ARG_WEIGHTS, user_weights_mem_fc2});
        fc2_args.insert({DNNL_ARG_BIAS, bias_mem_fc2});
        fc2_args.insert({DNNL_ARG_DST, dst_mem_fc2});

        //-----------------------------------------------------------------------
        //----------------- Backward Stream -------------------------------------
        // ... user diff_data in float data type ...
        std::vector<float> net_diff_dst(batch * 96 * 27 * 27);
        for (size_t i = 0; i < net_diff_dst.size(); ++i)
                net_diff_dst[i] = sinf((float)i);

        // create memory for user diff dst data stored in float data type
        auto pool_user_diff_dst_memory = memory({{pool_dst_tz}, dt::f32, tag::nchw}, eng);
        write_to_dnnl_memory(net_diff_dst.data(), pool_user_diff_dst_memory);

        // Backward pooling
        // create memory descriptors for pooling
        auto pool_diff_src_md = memory::desc({lrn_data_tz}, dt::bf16, tag::any);
        auto pool_diff_dst_md = memory::desc({pool_dst_tz}, dt::bf16, tag::any);

        // create backward pooling descriptor
        auto pool_bwd_desc = pooling_backward::desc(algorithm::pooling_max,
                                                    pool_diff_src_md, pool_diff_dst_md, pool_strides, pool_kernel,
                                                    pool_padding, pool_padding);
        // backward primitive descriptor needs to hint forward descriptor
        auto pool_bwd_pd = pooling_backward::primitive_desc(pool_bwd_desc, eng, pool_pd);

        // create reorder primitive between user diff dst and pool diff dst
        // if required
        auto pool_diff_dst_memory = pool_user_diff_dst_memory;
        if (pool_dst_memory.get_desc() != pool_user_diff_dst_memory.get_desc())
        {
                pool_diff_dst_memory = memory(pool_dst_memory.get_desc(), eng);
                net_bwd.push_back(
                    reorder(pool_user_diff_dst_memory, pool_diff_dst_memory));
                net_bwd_args.push_back({{DNNL_ARG_FROM, pool_user_diff_dst_memory},
                                        {DNNL_ARG_TO, pool_diff_dst_memory}});
        }

        // create memory for pool diff src
        auto pool_diff_src_memory = memory(pool_bwd_pd.diff_src_desc(), eng);

        // finally create backward pooling primitive
        net_bwd.push_back(pooling_backward(pool_bwd_pd));
        net_bwd_args.push_back({{DNNL_ARG_DIFF_DST, pool_diff_dst_memory},
                                {DNNL_ARG_DIFF_SRC, pool_diff_src_memory},
                                {DNNL_ARG_WORKSPACE, pool_workspace_memory}});

        // Backward lrn
        auto lrn_diff_dst_md = memory::desc({lrn_data_tz}, dt::bf16, tag::any);

        // create backward lrn primitive descriptor
        auto lrn_bwd_desc = lrn_backward::desc(algorithm::lrn_across_channels,
                                               lrn_pd.src_desc(), lrn_diff_dst_md, local_size, alpha, beta, k);
        auto lrn_bwd_pd = lrn_backward::primitive_desc(lrn_bwd_desc, eng, lrn_pd);

        // create reorder primitive between pool diff src and lrn diff dst
        // if required
        auto lrn_diff_dst_memory = pool_diff_src_memory;
        if (lrn_diff_dst_memory.get_desc() != lrn_bwd_pd.diff_dst_desc())
        {
                lrn_diff_dst_memory = memory(lrn_bwd_pd.diff_dst_desc(), eng);
                net_bwd.push_back(reorder(pool_diff_src_memory, lrn_diff_dst_memory));
                net_bwd_args.push_back({{DNNL_ARG_FROM, pool_diff_src_memory},
                                        {DNNL_ARG_TO, lrn_diff_dst_memory}});
        }

        // create memory for lrn diff src
        auto lrn_diff_src_memory = memory(lrn_bwd_pd.diff_src_desc(), eng);

        // finally create a lrn backward primitive
        // backward lrn needs src: relu dst in this topology
        net_bwd.push_back(lrn_backward(lrn_bwd_pd));
        net_bwd_args.push_back({{DNNL_ARG_SRC, relu_dst_memory},
                                {DNNL_ARG_DIFF_DST, lrn_diff_dst_memory},
                                {DNNL_ARG_DIFF_SRC, lrn_diff_src_memory},
                                {DNNL_ARG_WORKSPACE, lrn_workspace_memory}});

        // Backward relu
        auto relu_diff_dst_md = memory::desc({relu_data_tz}, dt::bf16, tag::any);
        auto relu_src_md = conv_pd.dst_desc();

        // create backward relu primitive_descriptor
        auto relu_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
                                                    relu_diff_dst_md, relu_src_md, negative_slope);
        auto relu_bwd_pd = eltwise_backward::primitive_desc(relu_bwd_desc, eng, relu_pd);

        // create reorder primitive between lrn diff src and relu diff dst
        // if required
        auto relu_diff_dst_memory = lrn_diff_src_memory;
        if (relu_diff_dst_memory.get_desc() != relu_bwd_pd.diff_dst_desc())
        {
                relu_diff_dst_memory = memory(relu_bwd_pd.diff_dst_desc(), eng);
                net_bwd.push_back(reorder(lrn_diff_src_memory, relu_diff_dst_memory));
                net_bwd_args.push_back({{DNNL_ARG_FROM, lrn_diff_src_memory},
                                        {DNNL_ARG_TO, relu_diff_dst_memory}});
        }

        // create memory for relu diff src
        auto relu_diff_src_memory = memory(relu_bwd_pd.diff_src_desc(), eng);

        // finally create a backward relu primitive
        net_bwd.push_back(eltwise_backward(relu_bwd_pd));
        net_bwd_args.push_back({{DNNL_ARG_SRC, conv_dst_memory},
                                {DNNL_ARG_DIFF_DST, relu_diff_dst_memory},
                                {DNNL_ARG_DIFF_SRC, relu_diff_src_memory}});

        // Backward convolution with respect to weights
        // create user format diff weights and diff bias memory for float data type

        auto conv_user_diff_weights_memory = memory({{conv_weights_tz}, dt::f32, tag::nchw}, eng);
        auto conv_diff_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);

        // create memory descriptors for bfloat16 convolution data
        auto conv_bwd_src_md = memory::desc({conv_src_tz}, dt::bf16, tag::any);
        auto conv_diff_weights_md = memory::desc({conv_weights_tz}, dt::bf16, tag::any);
        auto conv_diff_dst_md = memory::desc({conv_dst_tz}, dt::bf16, tag::any);

        // use diff bias provided by the user
        auto conv_diff_bias_md = conv_diff_bias_memory.get_desc();

        // create backward convolution primitive descriptor
        auto conv_bwd_weights_desc = convolution_backward_weights::desc(algorithm::convolution_direct,
                                                                        conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                                                                        conv_diff_dst_md, conv_strides, conv_padding, conv_padding);
        auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, eng, conv_pd);

        // for best performance convolution backward might chose
        // different memory format for src and diff_dst
        // than the memory formats preferred by forward convolution
        // for src and dst respectively
        // create reorder primitives for src from forward convolution to the
        // format chosen by backward convolution
        auto conv_bwd_src_memory = conv_src_memory;
        if (conv_bwd_weights_pd.src_desc() != conv_src_memory.get_desc())
        {
                conv_bwd_src_memory = memory(conv_bwd_weights_pd.src_desc(), eng);
                net_bwd.push_back(reorder(conv_src_memory, conv_bwd_src_memory));
                net_bwd_args.push_back({{DNNL_ARG_FROM, conv_src_memory},
                                        {DNNL_ARG_TO, conv_bwd_src_memory}});
        }

        // create reorder primitives for diff_dst between diff_src from relu_bwd
        // and format preferred by conv_diff_weights
        auto conv_diff_dst_memory = relu_diff_src_memory;
        if (conv_bwd_weights_pd.diff_dst_desc() != relu_diff_src_memory.get_desc())
        {
                conv_diff_dst_memory = memory(conv_bwd_weights_pd.diff_dst_desc(), eng);
                net_bwd.push_back(reorder(relu_diff_src_memory, conv_diff_dst_memory));
                net_bwd_args.push_back({{DNNL_ARG_FROM, relu_diff_src_memory},
                                        {DNNL_ARG_TO, conv_diff_dst_memory}});
        }

        // create backward convolution primitive
        net_bwd.push_back(convolution_backward_weights(conv_bwd_weights_pd));
        net_bwd_args.push_back({{DNNL_ARG_SRC, conv_bwd_src_memory},
                                {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
                                // delay putting DIFF_WEIGHTS until reorder (if needed)
                                {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});

        // create reorder primitives between conv diff weights and user diff weights
        // if needed
        auto conv_diff_weights_memory = conv_user_diff_weights_memory;
        if (conv_bwd_weights_pd.diff_weights_desc() != conv_user_diff_weights_memory.get_desc())
        {
                conv_diff_weights_memory = memory(conv_bwd_weights_pd.diff_weights_desc(), eng);
                net_bwd_args.back().insert(
                    {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});

                net_bwd.push_back(reorder(
                    conv_diff_weights_memory, conv_user_diff_weights_memory));
                net_bwd_args.push_back({{DNNL_ARG_FROM, conv_diff_weights_memory},
                                        {DNNL_ARG_TO, conv_user_diff_weights_memory}});
        }
        else
        {
                net_bwd_args.back().insert(
                    {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});
        }

        // didn't we forget anything?
        assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
        assert(net_bwd.size() == net_bwd_args.size() && "something is missing");

        int n_iter = 50; // number of iterations for training
        // execute
        while (n_iter)
        {
                // forward

                std::cout << "Iteration # " << n_iter << "\n";

                for (size_t i = 0; i < net_fwd.size(); ++i)
                        net_fwd.at(i).execute(s, net_fwd_args.at(i));

                // update net_diff_dst
                // auto net_output = pool_user_dst_memory.get_data_handle();
                // ..user updates net_diff_dst using net_output...
                // some user defined func update_diff_dst(net_diff_dst.data(),
                // net_output)

                for (size_t i = 0; i < net_bwd.size(); ++i)
                        net_bwd.at(i).execute(s, net_bwd_args.at(i));
                // update weights and bias using diff weights and bias
                //
                // auto net_diff_weights
                //     = conv_user_diff_weights_memory.get_data_handle();
                // auto net_diff_bias = conv_diff_bias_memory.get_data_handle();
                //
                // ...user updates weights and bias using diff weights and bias...
                //
                // some user defined func update_weights(conv_weights.data(),
                // conv_bias.data(), net_diff_weights, net_diff_bias);

                --n_iter;
        }

        s.wait();
}

int main(int argc, char **argv)
{
        return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
