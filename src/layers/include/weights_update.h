#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

void updateWeights_SGD(dnnl::memory weights, 
                   dnnl::memory diff_weights,
                   float learning_rate,
                   std::vector<dnnl::primitive> &net,
                   std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                   dnnl::engine eng);