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
/**
 * @brief L2-Loss class
 * 
 */

class L2_Loss{
    public:
        dnnl::memory arg_dst;
        /**
         * @brief Construct a new l2 loss object
         * 
         * @param y_hat Vector with prediction
         * @param y_true Vector with true labels
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        L2_Loss(dnnl::memory y_hat, dnnl::memory y_true, 
            std::vector<dnnl::primitive> &net,
            std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
            dnnl::engine eng);
    private:
          
};

/**
 * @brief Gradient of L2 Loss
 * 
 */
class L2_Loss_back{
    public:
        dnnl::memory arg_dst;
        /**
         * @brief Construct a new l2 loss back object
         * 
         * @param y_hat Vector with prediction
         * @param y_true Vector with true labels
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        L2_Loss_back(dnnl::memory y_hat, dnnl::memory y_true,
                 std::vector<dnnl::primitive> &net,
                 std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                 dnnl::engine eng);
    private:
          
};

/**
 * @brief Binary Cross Entropy Loss class
 * 
 */
class binaryCrossEntropyLoss{
    public:
        dnnl::memory arg_dst;
        /**
         * @brief Construct a new binary Cross Entropy Loss object
         * 
         * @param y_hat Vector with prediction
         * @param y_true Vector with true labels
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        binaryCrossEntropyLoss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng);
    private:
          
};

/**
 * @brief Gradient of binary cross entropy loss
 * 
 */

class binaryCrossEntropyLoss_back{
    public:
        dnnl::memory arg_dst;
        /**
         * @brief Construct a new binaryCrossEntropyLoss back object
         * 
         * @param y_hat Vector with prediction
         * @param y_true Vector with true labels
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
        binaryCrossEntropyLoss_back(dnnl::memory y_hat, dnnl::memory y_true,
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng);
    private:
          
};


#endif