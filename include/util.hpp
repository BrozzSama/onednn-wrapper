#include "oneapi/dnnl/dnnl.hpp"

// checkType syntax is:
// (md of primitive ie. the format supported by the primitive running on eng, memory with type that needs to be checked,
// network, network arguments, engine)
dnnl::memory checkType(dnnl::memory::desc md_true_type, dnnl::memory mem_to_check,
                       std::vector<dnnl::primitive> &net,
                       std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                       dnnl::engine eng)
{
    auto mem_reordered = mem_to_check;
    /*if (md_true_type != mem_to_check.get_desc())
    {
        std::cout << "Memory mismatch adding reorder primitive\n";
        auto mem_reordered = dnnl::memory(md_true_type, eng);
        net.push_back(dnnl::reorder(mem_to_check, mem_reordered));
        net_args.push_back({{DNNL_ARG_FROM, mem_to_check},
                                {DNNL_ARG_TO, mem_reordered}});
    }*/
    return mem_reordered;
}

void print_vector(std::vector<dnnl::memory::dim> const &input)
{
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input.at(i) << ' ';
    }
    std::cout << "\n";
}

void print_vector2(std::vector<float> input)
{
    int limit;
    if (input.size() > 50)
    {
        limit = 50;
    }
    else
    {
        limit = input.size();
    }
    for (int i = 0; i < limit; i++)
    {
        std::cout << input.at(i) << ' ';
    }
    std::cout << "\n";
}

void data_loader(std::string filename, std::vector<float> &data)
{
    std::ifstream myfile(filename);
    int i = 0;
    if (myfile.is_open())
    {
        while (myfile >> data[i++]){}
        
        std::cout << "Read from file: " <<data[2] << "\n";
        myfile.close();
    }

    else
        std::cout << "Unable to open file";

    
}