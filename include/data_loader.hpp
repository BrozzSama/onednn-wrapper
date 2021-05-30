#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <functional>   // std::multiplies
#include <numeric>      // std::accumulate

class DataLoader{
    public:
        int dataset_size, minibatch_size;
        std::vector<float> dataset, dataset_labels;
        std::vector<float> curr_batch, curr_batch_labels;
        void load_from_file(std::string filename, std::vector<float> &data);
        void write_to_memory(dnnl::memory dst_mem_features, dnnl::memory dst_mem_labels);
        DataLoader(std::string features_path, std::string labels_path, int _dataset_size, int _minibatch_size, std::vector<unsigned long> dataset_shape, dnnl::engine _eng);
    private:
        int dataset_idx, sample_size;
        dnnl::engine eng;
    
};

DataLoader::DataLoader(std::string features_path, std::string labels_path, int _dataset_size, int _minibatch_size, std::vector<unsigned long> dataset_shape, dnnl::engine _eng){
    dataset_idx = 0;
    dataset_size = _dataset_size;
    minibatch_size = _minibatch_size;
    eng = _eng;

    // Load features
    int total_size = std::accumulate(std::begin(dataset_shape), std::end(dataset_shape), 1, std::multiplies<int>());
    dataset.reserve(total_size);
    dataset.clear();
    load_from_file(features_path, dataset);

    // Load Labels
    int n_samples = dataset_shape[0];
    dataset_labels.reserve(n_samples);
    dataset_labels.clear();
    load_from_file(labels_path, dataset_labels);

    // Prepare batch vector
    sample_size = std::accumulate(std::begin(dataset_shape)+1, std::end(dataset_shape), 1, std::multiplies<int>()); 
    int curr_batch_size = minibatch_size * sample_size;
    curr_batch.reserve(curr_batch_size);

    curr_batch_labels.reserve(minibatch_size);

}

void DataLoader::write_to_memory(dnnl::memory dst_mem_features, dnnl::memory dst_mem_labels){
    this->dataset_idx += this->minibatch_size;
    if (this->dataset_idx > this->dataset_size){
        this->dataset_idx = 0;
    }

    auto start = this->dataset.begin() + this->dataset_idx * this->sample_size;
    auto end = this->dataset.begin() + (this->dataset_idx + this->minibatch_size) * this->sample_size;
    std::copy(start, end, this->curr_batch.begin());

    std::cout << "Wrote features\n";

    write_to_dnnl_memory(this->curr_batch.data(), dst_mem_features);

    start = this->dataset_labels.begin() + this->dataset_idx;
    end = this->dataset_labels.begin() + (this->dataset_idx + this->minibatch_size);
    std::copy(start, end, this->curr_batch_labels.begin());
    std::cout << "Wrote labels\n";

    write_to_dnnl_memory(this->curr_batch_labels.data(), dst_mem_labels);
}

void DataLoader::load_from_file(std::string filename, std::vector<float> &data)
{
    std::ifstream myfile(filename);
    int i = 0;
    if (myfile.is_open())
    {
        while (myfile >> data[i++]){}

        myfile.close();
    }

    else
        std::cout << "Unable to open file";    
}