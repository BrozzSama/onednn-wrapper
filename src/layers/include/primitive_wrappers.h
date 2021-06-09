#include "oneapi/dnnl/dnnl.hpp"
#include "intel_utils.h"
#include "util.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <time.h>

#ifndef _PRIMITIVE_WRAPPERS
#define _PRIMITIVE_WRAPPERS

class Reorder{
    public:
        dnnl::memory arg_src, arg_dst;
        Reorder(dnnl::memory src_mem, dnnl::memory dst_mem,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};


class Eltwise{
    public:
        dnnl::memory arg_src, arg_dst;
        Eltwise(dnnl::algorithm activation,
          float alpha,
          float beta,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng);
    private:
          
};


#endif