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

class Conv2D_back_weights{
    public:
        dnnl::memory arg_src, arg_diff_dst;
        dnnl::memory arg_diff_weights, arg_diff_bias;
        Conv2D_back_weights(dnnl::memory diff_dst,
           Conv2D conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

class Dense_back_weights{
    public:
        dnnl::memory arg_src, arg_diff_dst;
        dnnl::memory arg_diff_weights, arg_diff_bias;
        Dense_back_weights(dnnl::memory diff_dst,
           Dense dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

#endif