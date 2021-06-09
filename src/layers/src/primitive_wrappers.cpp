#include "../include/primitive_wrappers.h"

Reorder::Reorder(dnnl::memory src_mem, dnnl::memory dst_mem,
           std::vector<dnnl::primitive> &net,
           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
           dnnl::engine eng){
    // Initialize output memory

    arg_src = src_mem;
    arg_dst = dst_mem;

    net.push_back(dnnl::reorder(src_mem, dst_mem));
    net_args.push_back({{DNNL_ARG_FROM, src_mem},
                        {DNNL_ARG_TO, dst_mem}});

}

Eltwise::Eltwise(dnnl::algorithm activation,
          float alpha,
          float beta,
          dnnl::memory input,
          std::vector<dnnl::primitive> &net,
          std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
          dnnl::engine eng)
{

    auto src_md = input.get_desc();

    auto dst_mem = dnnl::memory(src_md, eng);
    auto dst_md =  dst_mem.get_desc();

    std::cout << "Memory allocated\n";

    auto eltwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, activation,
                                                dst_md, alpha, beta);
    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_desc, eng);

    arg_src = input;
    arg_dst = dst_mem;

    net.push_back(dnnl::eltwise_forward(eltwise_pd));
    net_args.push_back({{DNNL_ARG_SRC, input},
                        {DNNL_ARG_DST, dst_mem}});
}