#include "oneapi/dnnl/dnnl.hpp"
#include "layers_fwd.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

#ifndef _LAYERS_BWD_WEIGHTS
#define _LAYERS_BWD_WEIGHTS

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

/**
 * @brief Primitive which provides backward weights pass for Conv2D
 * 
 */
class Conv2D_back_weights{
    public:
        dnnl::memory arg_src, arg_diff_dst;
        dnnl::memory arg_diff_weights, arg_diff_bias;
        /**
         * @brief Construct a new Conv2D_back_weights object
         * 
         * @param diff_dst Gradient of loss with respect to the output
         * @param conv2d_fwd Forward Conv2D object
         * @param stride_length Stride
         * @param padding_length Padding
         * @param dilation Dilation coefficient
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        Conv2D_back_weights(dnnl::memory diff_dst,
           Conv2D conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

/**
 * @brief Primitive which provides backward weights pass for the Dense
 * 
 */
class Dense_back_weights{
    public:
        dnnl::memory arg_src, arg_diff_dst;
        dnnl::memory arg_diff_weights, arg_diff_bias;
        /**
         * @brief Construct a new Dense_back_weights object
         * 
         * @param diff_dst Gradient of loss with respect to the output
         * @param dense_fwd Forward Dense object
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        Dense_back_weights(dnnl::memory diff_dst,
           Dense dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

#endif