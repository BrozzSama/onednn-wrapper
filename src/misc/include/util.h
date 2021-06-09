#ifndef _UTILS
#define _UTILS

#include <iostream>
#include <fstream>
#include "oneapi/dnnl/dnnl.hpp"
dnnl::memory checkType(dnnl::memory::desc md_true_type, dnnl::memory mem_to_check,
                       std::vector<dnnl::primitive> &net,
                       std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                       dnnl::engine eng);

void print_vector(std::vector<dnnl::memory::dim> const &input);

void print_vector2(std::vector<float> input, int user_limit=100);

inline bool file_exists (const std::string& name);

#endif