#include "oneapi/dnnl/dnnl.hpp"

// checkType syntax is:
// (md of primitive ie. the format supported by the primitive running on eng, memory with type that needs to be checked,
// network, network arguments, engine)
dnnl::memory checkType(dnnl::memory_desc md_true_type, dnnl::memory mem_to_check, 
    std::vector<primitive> net, 
    std::vector<std::unordered_map<int, memory>> net_args, 
    dnnl::engine eng)
{
    auto mem_reordered = mem_to_check;
    if (md_true_type != mem_to_check.get_desc())
    {
        auto mem_reordered = memory(md_true_type, eng);
        net_fwd.push_back(reorder(mem_to_check, mem_reordered));
        net_fwd_args.push_back({{DNNL_ARG_FROM, mem_to_check},
                                {DNNL_ARG_TO, mem_reordered}});
    }
    return mem_reordered;
}