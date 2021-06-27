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


/**
 * @brief Dense allows to create a fully connected layer forward primitive
 * 
 */

class Dense{
    public:
        dnnl::memory arg_src; //!< Source memory handler
        dnnl::memory arg_dst; //!< Destination memory handler
        dnnl::memory arg_bias; //!< Bias memory handler
        dnnl::memory arg_weights; //!< Weights memory handler
        /**
         * @brief Construct a new Dense object
         * 
         * @param fc_output_size this is the number of units inside the FC layer
         * @param input Input memory
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        Dense(int fc_output_size,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng);
    private:
        
    
};
/**
 * @brief Conv2D allows to create a forward convolution primitive
 * 
 */
class Conv2D{
    public:
        dnnl::memory arg_src; //!< Source memory handler
        dnnl::memory arg_dst; //!< Destination memory handler
        dnnl::memory arg_bias; //!< Bias memory handler
        dnnl::memory arg_weights; //!< Weights memory handler
        /**
         * @brief Construct a new Conv 2 D object
         * 
         * @param batch_size Size of the batch
         * @param patch_length Length of the H and W
         * @param n_kernels Number of kernels
         * @param kernel_size Size of the kernel
         * @param stride_length Stride
         * @param padding_length Padding 
         * @param dilation Dilation coefficient for the dilated convolution (0 for no dilation as per oneAPI specs)
         * @param input Input memory
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
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

/**
 * @brief Primitive which provides max pooling
 * 
 */

class MaxPool2D{
    public:
        dnnl::memory arg_src, arg_dst, arg_workspace;
        dnnl::pooling_v2_forward::primitive_desc *pooling_fwd_pd;
        /**
         * @brief Construct a new Max Pool 2 D object
         * 
         * @param kernel_size the size of the kernel
         * @param stride_length the length of the stride
         * @param input Input memory
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        MaxPool2D(int kernel_size, int stride_length, 
           dnnl::memory input,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
    
};

#endif 


