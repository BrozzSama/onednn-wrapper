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

    DataLoader(std::string features_path, std::string label_path, int dataset_size, int minibatch_size);
    load_from_file(std::string filename, std::vector<float> &data);
}

DataLoader::DataLoader(std::string features_path, std::string label_path, int _dataset_size, int _minibatch_size, std::Vector<float> dataset_shape){
    dataset_size = _dataset_size;
    minibatch_size = _minibatch_size;

    // Load features
    total_size = std::accumulate(std::begin(dataset_shape), std::end(dataset_shape), 1, std::multiplies<int>());
    std::vector<float> dataset(total_size);
    dataset.clear();
    load_from_file(features_path, dataset);

    n_samples = dataset_shape[0];
    std::vector<float> dataset_labels(n_samples);
    dataset_labels.clear();
    load_from_file(labels_path, dataset_labels);

}

DataLoader::load_from_file(std::string filename, std::vector<float> &data)
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