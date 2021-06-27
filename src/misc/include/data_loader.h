#include "oneapi/dnnl/dnnl.hpp"
#ifndef _DATA_LOADER
#define _DATA_LOADER

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <functional>   // std::multiplies
#include <numeric>      // std::accumulate
#include "intel_utils.h"

/**
 * @brief DataLoader allows to create a dataloader object to implement minibatch stochastic gradient descent
 * 
 */

class DataLoader{
    public:
        int dataset_size; //!< Total number of samples
        int minibatch_size; //!< Minibatch size
        std::vector<float> dataset; //!< Vector containing the entire dataset
        std::vector<float> dataset_labels; //!< Vector containing the labels
        std::vector<float> curr_batch; //!< Vector containing the current batch that will be written to the engine
        std::vector<float> curr_batch_labels; //!< Vector containing the labels that will be written to the engine
        void write_to_memory(dnnl::memory dst_mem_features, dnnl::memory dst_mem_labels);  //!< Method that writes the curr batch to memory and moves the index forward
        /**
         * @brief Construct a new Data Loader object
         * 
         * This allows the creation of a class which implements minibatch stochastic gradient descent
         * @param features_path path to the text file containing the flattened features (in row-major order)
         * @param labels_path path to the labels corresponding to the features
         * @param _minibatch_size size of the minibatch (-1 for full batch)
         * @param dataset_shape shape of the single sample eg. {C} or {C, H, W}
         * @param _eng oneAPI engine
         */
        DataLoader(std::string features_path, std::string labels_path, int _minibatch_size, std::vector<long> dataset_shape, dnnl::engine _eng);
    private:
        int dataset_idx, sample_size;
        void load_from_file(std::string filename, std::vector<float> &data); //!< Helper method which loads the files
        dnnl::engine eng;
    
};

#endif