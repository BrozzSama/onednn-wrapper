#include "../include/weights_update.h"

void updateWeights_SGD(dnnl::memory weights, 
                   dnnl::memory diff_weights,
                   float learning_rate,
                   std::vector<dnnl::primitive> &net,
                   std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                   dnnl::engine eng)
{

    std::vector<dnnl::memory> sub_vector = {weights, diff_weights};
    std::vector<dnnl::memory::desc> sub_vector_md = {sub_vector[0].get_desc(), sub_vector[1].get_desc()};

    // Minibatch gradient descent needs normalization
    const long minibatch_size = sub_vector_md[0].dims()[0];
    std::vector<float> scales = {1.f, (learning_rate) * (-1.f)};

    auto weights_update_pd = dnnl::sum::primitive_desc(sub_vector_md[0], scales, sub_vector_md, eng);

    std::cout << "Created sum primitive" << "\n"; 

    net.push_back(dnnl::sum(weights_update_pd));

    std::unordered_map<int, dnnl::memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, sub_vector[0]});
    for (int i = 0; i<sub_vector.size(); i++){
        sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector[i]});
    }

    net_args.push_back(sum_args);

}