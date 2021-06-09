#include "oneapi/dnnl/dnnl.hpp"
#include "intel_utils.h"
#include "util.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <time.h>

#ifndef _LAYERS_FWD
#define _LAYERS_FWD

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;


class Dense{
    public:
        dnnl::memory arg_src, arg_dst;
        dnnl::memory arg_bias, arg_weights;
        Dense(dnnl::memory::dims src_dims, 
          int fc_output_size,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng);
    private:
        
    
};

class Conv2D{
    public:
        dnnl::memory arg_src, arg_dst;
        dnnl::memory arg_bias, arg_weights;
        Conv2D(int batch_size, int patch_length,
           int n_kernels, int kernel_size,
           int stride_length, int padding_length,
           int dilation,
           dnnl::memory input,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
    
};

#endif 


