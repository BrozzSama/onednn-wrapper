#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
/**
 * @brief Primitive which updates the weights of SGD
 * 
 * @param weights dnnl::memory object with current weights
 * @param diff_weights dnnl::memory object with the gradient of the weights
 * @param learning_rate learning rate of SGD
 * @param net This is the vector of primitives to which we will append the FC layer primitive
 * @param net_args This is the associated map to which we will add the arguments of the primitive
 * @param eng oneAPI engine that will host the primitive
 */
void updateWeights_SGD(dnnl::memory weights, 
                   dnnl::memory diff_weights,
                   float learning_rate,
                   std::vector<dnnl::primitive> &net,
                   std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                   dnnl::engine eng);