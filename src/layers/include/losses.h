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

int L2_Loss(dnnl::memory y_hat, dnnl::memory y_true, 
            std::vector<dnnl::primitive> &net,
            std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
            dnnl::engine eng);

// Net forward args is passed because a loss function contains more than one primitive
int L2_Loss_back(dnnl::memory y_hat, dnnl::memory y_true,
                 std::vector<dnnl::primitive> &net,
                 std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                 dnnl::engine eng);

int binaryCrossEntropyLoss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng);

int binaryCrossEntropyLoss_back(dnnl::memory y_hat, dnnl::memory y_true,
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng);

#endif