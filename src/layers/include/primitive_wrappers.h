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

/**
 * @brief Reorder Primitive
 * 
 */
class Reorder{
    public:
        dnnl::memory arg_src, arg_dst;
        /**
         * @brief Construct a new Reorder object
         * 
         * @param src_mem Source dnnl::memory object
         * @param dst_mem Destination dnnl::memory object
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        Reorder(dnnl::memory src_mem, dnnl::memory dst_mem,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

/**
 * @brief Primitive which provides element-wise operations
 * 
 */
class Eltwise{
    public:
        dnnl::memory arg_src, arg_dst;
        /**
         * @brief Construct a new Eltwise object
         * 
         * @param activation dnnl::algorithm objects which defines the element-wise operation
         * @param alpha Alpha parameter (algorithm dependent)
         * @param beta Beta Paremeter (algorithm dependent)
         * @param input dnnl:memory object containing the input
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
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