#include "oneapi/dnnl/dnnl.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

int L2_Loss(dnnl::memory y_hat, dnnl::memory y_true, 
            std::vector<dnnl::primitive> &net,
            std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
            dnnl::engine eng)
{

    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_cn = {1, y_hat.get_desc().dims()[0]};
    dnnl::memory::dims dim_scalar = {1, 1};

    dnnl::memory::desc vector_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    dnnl::memory::desc vector_transpose_md = dnnl::memory::desc(dim_cn, dt::f32, tag::cn);
    dnnl::memory::desc scalar_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::nc);

    // 1) Sum y_hat - y_true 

    dnnl::memory loss_sub = dnnl::memory(vector_transpose_md, eng);

    // Create scale for subtraction
    std::vector<float> scales = {1.f, -1.f};

    std::vector<dnnl::memory::desc> sub_vector_md = {vector_md, vector_md};
    std::vector<dnnl::memory> sub_vector = {y_hat, y_true};

    auto loss_sub_pd = dnnl::sum::primitive_desc(scales, sub_vector_md, eng);

    net.push_back(dnnl::sum(loss_sub_pd));

    std::vector<float> init_values(product(dim_nc));
    for (int i = 0; i<init_values.size(); i++){
        init_values[i] = 65.f;
    }
    write_to_dnnl_memory((void*) init_values.data(), loss_sub);    

    std::unordered_map<int, dnnl::memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sub});
    for (int i = 0; i<sub_vector.size(); i++){
        sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector[i]});
    }

    net_args.push_back(sum_args);

    // 2) Inner product <(y_hat-y_true), (y_hat-true)>
    
    dnnl::memory loss = dnnl::memory(scalar_md, eng);

    auto ip_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, vector_transpose_md,
                                                vector_transpose_md, scalar_md);
    
    std::cout << "Inner product desc created" << "\n";

    // nullptr, since we have no attr (see oneDNN primitive_desc API reference)
    //auto ip_pd = dnnl::inner_product_forward::primitive_desc(ip_desc, nullptr, eng);
    auto ip_pd = dnnl::inner_product_forward::primitive_desc(ip_desc, eng);
    
    net.push_back(dnnl::inner_product_forward(ip_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sub},
                        {DNNL_ARG_WEIGHTS, loss_sub},
                        {DNNL_ARG_DST, loss}});
    
    return net.size() - 1;
}

// Net forward args is passed because a loss function contains more than one primitive
int L2_Loss_back(dnnl::memory y_hat, dnnl::memory y_true,
                 std::vector<dnnl::primitive> &net,
                 std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                 dnnl::engine eng)
{    
    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};

    dnnl::memory::desc vector_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);

    // 1) 2*(y_hat - y_true) 

    dnnl::memory loss_sub = dnnl::memory(vector_md, eng);

    // Create scale for subtraction
    std::vector<float> scales = {2.f, -2.f};

    std::vector<dnnl::memory::desc> sub_vector_md = {vector_md, vector_md};
    std::vector<dnnl::memory> sub_vector = {y_hat, y_true};

    auto loss_sub_pd = dnnl::sum::primitive_desc(scales, sub_vector_md, eng);

    net.push_back(dnnl::sum(loss_sub_pd));

    std::vector<float> init_values(product(dim_nc));
    for (int i = 0; i<init_values.size(); i++){
        init_values[i] = 65.f;
    }
    write_to_dnnl_memory((void*) init_values.data(), loss_sub);    

    std::unordered_map<int, dnnl::memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sub});
    for (int i = 0; i<sub_vector.size(); i++){
        sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector[i]});
    }

    net_args.push_back(sum_args);

    return net.size() - 1;
}

int binaryCrossEntropyLoss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{
    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_cn = {1, y_hat.get_desc().dims()[0]};
    dnnl::memory::dims dim_scalar = {1, 1};
    
    // memory descriptors used for all vectors
    std::cout << "Binary cross entropy loss: creating vector md\n";
    dnnl::memory::desc vector_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    std::cout << "Binary cross entropy loss: creating vector transpose md\n";
    dnnl::memory::desc vector_transpose_md = dnnl::memory::desc(dim_cn, dt::f32, tag::cn);
    // memory descriptor used for all scalars
    std::cout << "Binary cross entropy loss: creating scalar md\n";
    dnnl::memory::desc scalar_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::nc);

    // 1) Perform elementwise log on y_hat

    std::cout << "Binary cross entropy loss: creating log 1\n";

    auto y_hat_log = dnnl::memory(vector_transpose_md, eng);

    auto log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                vector_md, 0.f, 0.f);
    auto log_pd = dnnl::eltwise_forward::primitive_desc(log_desc, eng);

    net.push_back(dnnl::eltwise_forward(log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_log}});

    // 2) Perform elementwise linear on y_hat with alpha = -1; beta = 1 ie. 1-y_hat

    std::cout << "Binary cross entropy loss: creating eltwise linear 1\n";

    auto y_hat_inv = dnnl::memory(vector_transpose_md, eng);

    auto inv_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                vector_md, -1.f, 1.f);
    auto inv_pd = dnnl::eltwise_forward::primitive_desc(inv_desc, eng);

    net.push_back(dnnl::eltwise_forward(inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_inv}});

    // 3) Perform log on previously obtained element ie. log(1-y_hat)

    std::cout << "Binary cross entropy loss: creating log 2\n";

    auto y_hat_inv_log = dnnl::memory(vector_transpose_md, eng);

    net.push_back(dnnl::eltwise_forward(log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_inv},
                        {DNNL_ARG_DST, y_hat_inv_log}});

    // 4) Perform elementwise linear on y_true with alpha = -1; beta = 1 ie. 1-y_true

    std::cout << "Binary cross entropy loss: creating eltwise linear 2\n";

    auto y_true_inv = dnnl::memory(vector_transpose_md, eng);

    net.push_back(dnnl::eltwise_forward(inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true},
                        {DNNL_ARG_DST, y_true_inv}});

    // 5) Perform inner_product(y_true, log(y_hat))

    // Create y_true transpose by using an eltwise_linear which multiplies by 1

    std::cout << "Binary cross entropy loss: creating y_true_transpose\n";

    auto y_true_transpose = dnnl::memory(vector_transpose_md, eng);

    auto identity_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                vector_md, 1.f, 0.f);
    auto identity_pd = dnnl::eltwise_forward::primitive_desc(identity_desc, eng);

    net.push_back(dnnl::eltwise_forward(identity_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true},
                        {DNNL_ARG_DST, y_true_transpose}});  

    /*
    // Reorder y_true and log(y_hat) from nc to cn, otherwise inner product receives wrong dimensions

    std::cout << "Binary cross entropy loss: reordering y_true\n";

    auto y_true_transpose = dnnl::memory(vector_transpose_md, eng);

    auto reorder_pd = dnnl::reorder::primitive_desc(eng, vector_md, eng, vector_transpose_md);
    //auto reorder_prim = dnnl::reorder(reorder_pd);

    net.push_back(dnnl::reorder(reorder_pd));

    std::unordered_map<int, dnnl::memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, y_true});
    reorder_args.insert({DNNL_ARG_DST, y_true_transpose});

    net_args.push_back(reorder_args);

    //net.push_back(dnnl::reorder(y_true, y_true_transpose));
    //net_args.push_back({{DNNL_ARG_FROM, y_true}, {DNNL_ARG_TO, y_true_transpose}});

    std::cout << "Binary cross entropy loss: reordering y_hat_log\n";

    auto y_hat_log_transpose = dnnl::memory(vector_transpose_md, eng);
    net.push_back(dnnl::reorder(y_hat_log, y_hat_log_transpose));
    net_args.push_back({{DNNL_ARG_FROM, y_hat_log}, {DNNL_ARG_TO, y_hat_log_transpose}});
    */

    std::cout << "Binary cross entropy loss: creating inner product 1\n";

    auto add1 = dnnl::memory(scalar_md, eng);

    auto ip_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, vector_transpose_md,
                                                vector_transpose_md, scalar_md);

    // nullptr, since we have no attr (see oneDNN primitive_desc API reference)
    auto ip_pd = dnnl::inner_product_forward::primitive_desc(ip_desc, eng);
    
    net.push_back(dnnl::inner_product_forward(ip_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_transpose},
                        {DNNL_ARG_WEIGHTS, y_hat_log},
                        {DNNL_ARG_DST, add1}});

    // 6) Perform inner_product(1-y_true, log(1-y_hat))

    /*
    // Reorder 1-y_true and log(1-y_hat) from nc to cn, otherwise inner product receives wrong dimensions

    auto y_true_inv_transpose = dnnl::memory(vector_transpose_md, eng);
    net.push_back(dnnl::reorder(y_true_inv, y_true_inv_transpose));
    net_args.push_back({{DNNL_ARG_FROM, y_true_inv}, {DNNL_ARG_TO, y_true_inv_transpose}});

    auto y_hat_inv_log_transpose = dnnl::memory(vector_transpose_md, eng);
    net.push_back(dnnl::reorder(y_hat_inv_log, y_hat_inv_log_transpose));
    net_args.push_back({{DNNL_ARG_FROM, y_hat_inv_log}, {DNNL_ARG_TO, y_hat_inv_log_transpose}});
    */

    std::cout << "Binary cross entropy loss: creating inner product 2\n";

    auto add2 = dnnl::memory(scalar_md, eng);

    //auto matmul2_desc = dnnl::matmul::desc(y_true_inv_reordered_md, y_hat_inv_log_md, NULL, add2_md);
    //auto matmul2_pd = dnnl::matmul::primitive_desc(matmul2_desc, eng);

    net.push_back(dnnl::inner_product_forward(ip_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_inv},
                        {DNNL_ARG_WEIGHTS, y_hat_inv_log},
                        {DNNL_ARG_DST, add2}});

    // 7) Sum obtained values (should be scalars)

    std::cout << "Binary cross entropy loss: creating sum\n";

    auto loss_sum = dnnl::memory(scalar_md, eng);

    std::vector<dnnl::memory::desc> add_vector_md = {scalar_md, scalar_md};
    std::vector<dnnl::memory> add_vector = {add1, add2};
    std::vector<float> scales = {1.f, 1.f};

    auto loss_sum_pd = dnnl::sum::primitive_desc(scales, add_vector_md, eng);

    net.push_back(dnnl::sum(loss_sum_pd));

    std::unordered_map<int, dnnl::memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sum});
    for (int i = 0; i<add_vector.size(); i++){
        sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, add_vector[i]});
    }

    net_args.push_back(sum_args);
    
    // 8) Divide by -N using elementwise linear

    std::cout << "Binary cross entropy loss: creating eltwise linear 3\n";

    auto loss = dnnl::memory(scalar_md, eng);

    float batch_size_inv = 1/y_true.get_desc().dims()[0];

    //auto loss_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
    //                                            y_true_reordered.get_desc(), -batch_size_inv, 0.f);
    //auto loss_pd = dnnl::eltwise_forward::primitive_desc(loss_desc, eng);

    auto divide_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                scalar_md, -batch_size_inv, 0.f);
    auto divide_pd = dnnl::eltwise_forward::primitive_desc(divide_desc, eng);

    net.push_back(dnnl::eltwise_forward(divide_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sum},
                        {DNNL_ARG_DST, loss}});   

    return net.size() - 1;
}

int binaryCrossEntropyLoss_back(dnnl::memory y_hat, dnnl::memory y_true, int batch,
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{
    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_scalar = {1, 1};

    // memory descriptor used for all vectors
    dnnl::memory::desc vector_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);

    // 1) y_hat - y_true
    dnnl::memory sub_num = dnnl::memory(vector_md, eng);

    // Create scale for subtraction
    std::vector<float> scales_num = {1.f, -1.f};

    std::vector<dnnl::memory::desc> sub_vector_num_md = {y_hat.get_desc(), y_true.get_desc()};
    std::vector<dnnl::memory> sub_vector_num = {y_hat, y_true};

    auto sub_num_pd = dnnl::sum::primitive_desc(vector_md, scales_num, sub_vector_num_md, eng);

    std::cout << "Created sum primitive" << "\n"; 

    /*
    std::vector<float> init_values(product(dim_nc));

    for (int i = 0; i<init_values.size(); i++){
        init_values[i] = 65.f;
    }

    write_to_dnnl_memory((void*) init_values.data(), numerator_sub);
    */

    net.push_back(dnnl::sum(sub_num_pd));

    std::unordered_map<int, dnnl::memory> sum_args_num;

    sum_args_num.insert({DNNL_ARG_DST, sub_num});
    for (int i = 0; i<sub_vector_num.size(); i++){
        sum_args_num.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector_num[i]});
    }

    net_args.push_back(sum_args_num);

    // 2) y_hat^2

    auto y_hat_squared = dnnl::memory(vector_md, eng);

    auto power_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_pow,
                                                vector_md, 1.f, 2.f);
    auto power_pd = dnnl::eltwise_forward::primitive_desc(power_desc, eng);

    net.push_back(dnnl::eltwise_forward(power_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_squared}});

    // 3) N * y_hat * (1 - y_hat) => N * y_hat - N * y_hat^2

    dnnl::memory sub_den = dnnl::memory(vector_md, eng);

    // Create scale for subtraction
    std::vector<float> scales_den = {(float) batch, (float) -batch};

    std::vector<dnnl::memory::desc> sub_vector_den_md = {y_hat.get_desc(), vector_md};
    std::vector<dnnl::memory> sub_vector_den = {y_hat, y_hat_squared};

    auto sub_den_pd = dnnl::sum::primitive_desc(vector_md, scales_den, sub_vector_den_md, eng);

    std::cout << "Created sum primitive" << "\n"; 

    net.push_back(dnnl::sum(sub_den_pd));

    std::unordered_map<int, dnnl::memory> sum_args_den;

    sum_args_den.insert({DNNL_ARG_DST, sub_den});
    for (int i = 0; i<sub_vector_den.size(); i++){
        sum_args_den.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector_den[i]});
    }

    net_args.push_back(sum_args_den);


    // 4) log(y_hat - y_true) => log(sub_num)
   
    auto log_num = dnnl::memory(vector_md, eng);

    auto log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                vector_md, 0.f, 0.f);
    auto log_pd = dnnl::eltwise_forward::primitive_desc(log_desc, eng);

    net.push_back(dnnl::eltwise_forward(log_pd));
    net_args.push_back({{DNNL_ARG_SRC, sub_num},
                        {DNNL_ARG_DST, log_num}});

    // 5) log(N * y_hat - N * y_hat_squared) => log(sub_den)

    auto log_den = dnnl::memory(vector_md, eng);

    net.push_back(dnnl::eltwise_forward(log_pd));
    net_args.push_back({{DNNL_ARG_SRC, sub_den},
                        {DNNL_ARG_DST, log_den}});

    // 6) log(sub_num) - log(sub_den)

    dnnl::memory sub_logs = dnnl::memory(vector_md, eng);

    // Create scale for subtraction
    std::vector<float> scales_logs = {1.f, -1.f};

    std::vector<dnnl::memory::desc> sub_vector_logs_md = {vector_md, vector_md};
    std::vector<dnnl::memory> sub_vector_logs = {sub_num, sub_den};

    auto sub_logs_pd = dnnl::sum::primitive_desc(vector_md, scales_logs, sub_vector_logs_md, eng);

    std::cout << "Created sum primitive" << "\n"; 

    net.push_back(dnnl::sum(sub_logs_pd));

    std::unordered_map<int, dnnl::memory> sum_args_logs;

    sum_args_logs.insert({DNNL_ARG_DST, sub_logs});
    for (int i = 0; i<sub_vector_logs.size(); i++){
        sum_args_logs.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector_logs[i]});
    }

    net_args.push_back(sum_args_logs);

    // 7) exp(log(sub_num) - log(sub_den))

    auto loss_diff = dnnl::memory(vector_md, eng);

    auto exp_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_exp,
                                                vector_md, 0.f, 0.f);
    auto exp_pd = dnnl::eltwise_forward::primitive_desc(exp_desc, eng);

    net.push_back(dnnl::eltwise_forward(exp_pd));
    net_args.push_back({{DNNL_ARG_SRC, sub_logs},
                        {DNNL_ARG_DST, loss_diff}});

    return net.size() - 1;
}