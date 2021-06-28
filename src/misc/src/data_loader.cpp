#include "../include/data_loader.h"

DataLoader::DataLoader(std::string features_path, std::string labels_path, int _minibatch_size, std::vector<int> dataset_shape, dnnl::engine _eng){
    dataset_idx = 0;
    eng = _eng;

    // Load features
    load_from_file(features_path, dataset);

    // Load Labels
    load_from_file(labels_path, dataset_labels);

    // Prepare batch vector
    sample_size = std::accumulate(std::begin(dataset_shape), std::end(dataset_shape), 1, std::multiplies<int>()); 
    dataset_size = dataset.size()/sample_size;
    std::cout << "Loaded a dataset of size: " << dataset_size << "\n";
    if (_minibatch_size == -1){
        minibatch_size = dataset_size;
    }
    else {
        minibatch_size = _minibatch_size;
    }
    int curr_batch_size = minibatch_size * sample_size;
    curr_batch.reserve(curr_batch_size);
    curr_batch_labels.reserve(minibatch_size);

}

void DataLoader::write_to_memory(dnnl::memory dst_mem_features, dnnl::memory dst_mem_labels){
    if (this->dataset_idx + this->minibatch_size >= this->dataset_size){
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
    this->dataset_idx += this->minibatch_size;
}

void DataLoader::load_from_file(std::string filename, std::vector<float> &data)
{
    std::ifstream myfile(filename);
    int i = 0;
    float curr;
    if (myfile.is_open())
    {
        while (myfile >> curr){
            data.push_back(curr);
        }

        myfile.close();
    }

    else
        std::cout << "Unable to open file";    
}