#include "oneapi/dnnl/dnnl.hpp"
#include "layers_fwd.h"
#include "primitive_wrappers.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

#ifndef _LAYERS_BWD_DATA
#define _LAYERS_BWD_DATA


int Conv2D_back_data(dnnl::memory diff_dst,
           std::unordered_map<int, dnnl::memory> conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);

class Eltwise_back{
    public:
        dnnl::memory arg_diff_src, arg_diff_dst;
        dnnl::memory arg_src;
        Eltwise_back(dnnl::algorithm activation,
          float alpha,
          float beta,
          Eltwise eltwise_fwd, 
          dnnl::memory diff_dst,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng);
    private:
          
};

class Dense_back_data{
    public:
        dnnl::memory arg_diff_src, arg_diff_dst;
        dnnl::memory arg_weights;
        Dense_back_data(dnnl::memory diff_dst,
           Dense dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

#endif