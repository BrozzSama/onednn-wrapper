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

class DataLoader{
    public:
        int dataset_size, minibatch_size;
        std::vector<float> dataset, dataset_labels;
        std::vector<float> curr_batch, curr_batch_labels;
        void load_from_file(std::string filename, std::vector<float> &data);
        void write_to_memory(dnnl::memory dst_mem_features, dnnl::memory dst_mem_labels);
        DataLoader(std::string features_path, std::string labels_path, int _dataset_size, int _minibatch_size, std::vector<long> dataset_shape, dnnl::engine _eng);
    private:
        int dataset_idx, sample_size;
        dnnl::engine eng;
    
};

#endif