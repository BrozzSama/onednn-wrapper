#include "oneapi/dnnl/dnnl.hpp"
#include "intel_utils.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

#ifndef _LOSSES
#define _LOSSES

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

class L2_Loss{
    public:
        dnnl::memory arg_dst;
        L2_Loss(dnnl::memory y_hat, dnnl::memory y_true, 
            std::vector<dnnl::primitive> &net,
            std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
            dnnl::engine eng);
    private:
          
};
                 
class L2_Loss_back{
    public:
        dnnl::memory arg_dst;
        L2_Loss_back(dnnl::memory y_hat, dnnl::memory y_true,
                 std::vector<dnnl::primitive> &net,
                 std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                 dnnl::engine eng);
    private:
          
};

class binaryCrossEntropyLoss{
    public:
        dnnl::memory arg_dst;
        binaryCrossEntropyLoss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng);
    private:
          
};

class binaryCrossEntropyLoss_back{
    public:
        dnnl::memory arg_dst;
        binaryCrossEntropyLoss_back(dnnl::memory y_hat, dnnl::memory y_true,
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng);
    private:
          
};


#endif