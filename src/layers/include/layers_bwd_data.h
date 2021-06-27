#include "oneapi/dnnl/dnnl.hpp"
#include "layers_fwd.h"
#include "primitive_wrappers.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

#ifndef _LAYERS_BWD_DATA
#define _LAYERS_BWD_DATA

/**
 * @brief Backward Data convolution
 * 
 */

class Conv2D_back_data{
    public:
        dnnl::memory arg_diff_src; //<! Gradient of the loss with respect to the input
        dnnl::memory arg_diff_dst; //<! Gradient of the loss with respect to the output
        dnnl::memory arg_weights; //<! Weights of the convolution primitive
        /**
         * @brief Construct a new Conv2D_back_data object
         * 
         * @param diff_dst Gradient of the loss with respect to the output (ie. the gradient coming from the previous layer)
         * @param conv2d_fwd The class containing the forward primitive
         * @param stride_length The stride
         * @param padding_length The padding
         * @param dilation The dilation
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
        Conv2D_back_data(dnnl::memory diff_dst,
           Conv2D conv2d_fwd,
           int stride_length, int padding_length,
           int dilation,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

/**
 * @brief Backward Data operation for 2D Max Pooling
 * 
 */
class MaxPool2D_back{
    public:
        dnnl::memory arg_diff_src, arg_diff_dst;
        /**
         * @brief Construct a new MaxPool2D_back object
         * 
         * @param kernel_size the size of the kernel
         * @param stride_length the stride length
         * @param maxpool_fwd the MaxPool2D forward class
         * @param diff_dst_mem The dnnl::memory object containing the gradient of the loss with respect to the output
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
        MaxPool2D_back(int kernel_size, int stride_length, 
           MaxPool2D maxpool_fwd,
           dnnl::memory diff_dst_mem,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
    
};
/**
 * @brief Eltwise backward primitive
 * 
 */
class Eltwise_back{
    public:
        dnnl::memory arg_diff_src, arg_src, arg_diff_dst;
        /**
         * @brief Construct a new Eltwise_back object
         * 
         * @param activation 
         * @param alpha 
         * @param beta 
         * @param eltwise_fwd 
         * @param diff_dst 
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
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

/**
 * @brief Dense layer backward data primitive
 * 
 */

class Dense_back_data{
    public:
        dnnl::memory arg_diff_src, arg_diff_dst;
        dnnl::memory arg_weights;
        /**
         * @brief Construct a new Dense_back_data object
         * 
         * @param diff_dst The dnnl::memory object containing the gradient of the loss with respect to the output
         * @param dense_fwd The Dense forward layer
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
        Dense_back_data(dnnl::memory diff_dst,
           Dense dense_fwd,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng);
    private:
          
};

#endif