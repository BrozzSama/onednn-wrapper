#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

int Clip(dnnl::memory src_mem,
         float upper, float lower,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng)
{
    // Initialize output memory

    auto dst_md = src_mem.get_desc();        
    auto dst_mem = dnnl::memory(dst_md, eng);

    auto clip_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_clip,
                                                dst_md, lower, upper);
    auto clip_pd = dnnl::eltwise_forward::primitive_desc(clip_desc, eng);

    std::cout << "Adding clip\n";
    net.push_back(dnnl::eltwise_forward(clip_pd));
    std::cout << "Adding clip arguments\n";
    net_args.push_back({{DNNL_ARG_SRC, src_mem},
                        {DNNL_ARG_DST, dst_mem}});   
    
    return net.size() - 1;

}

int Reorder(dnnl::memory src_mem, dnnl::memory dst_mem,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng){
    // Initialize output memory

    net.push_back(dnnl::reorder(src_mem, dst_mem));
    net_args.push_back({{DNNL_ARG_FROM, src_mem},
                        {DNNL_ARG_TO, dst_mem}});

    return net.size() - 1;
}