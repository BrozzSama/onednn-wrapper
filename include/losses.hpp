

int L2_Loss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{

    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_scalar = {1};

    // 1) Sum y_hat - y_true 

    dnnl::memory::desc loss_sub_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    dnnl::memory loss_sub = dnnl::memory(loss_sub_md, eng);

    // Create scale for subtraction
    std::vector<float> scales = {1.f, -1.f};

    std::vector<dnnl::memory::desc> sub_vector_md = {y_hat.get_desc(), y_true.get_desc()};
    std::vector<dnnl::memory> sub_vector = {y_hat, y_true};

    auto loss_sub_pd = dnnl::sum::primitive_desc(scales, sub_vector_md, eng);

    net.push_back(dnnl::sum(loss_sub_pd));

    std::unordered_map<int, memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sub});
    for (int i = 0; i<sub_vector.size(); i++){
        sum_args.insert({DNN_ARG_SRC + i, sub_vector[i]});
    }

    net_args.push_back(sum_args);

    // 2) Inner product <(y_hat - y_true), (y_hat-true)>

    dnnl::memory::desc loss_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    dnnl::memory loss = dnnl::memory(loss_md, eng);

    auto ip1_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, sub_vector_md,
                                                sub_vector_md, NULL, loss_md);

    // nullptr, since we have no attr (see oneDNN primitive_desc API reference)
    auto ip1_pd = dnnl::inner_product_forward::primitive_desc(
        ip1_desc, nullptr, eng);

    net.push_back(dnnl::inner_product_forward(ip1_pd));
    net_args.push_back({{DNNL_ARG_SRC, sub_vector_md},
                        {DNNL_ARG_WEIGHTS, sub_vector_md},
                        {DNNL_ARG_DST, loss}});
    
    return net.size() - 1;

}

int L2_Loss_back(std::vector<dnnl::primitive> net_fwd, 
                 int finish,
                 std::vector<dnnl::primitive> &net,
                 std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                 dnnl::engine eng)
{
    // we want to start from the first layer of the L2 loss when dealing with the offset
    int start = finish - L2_LOSS;

    auto loss_sub = net_fwd[start + L2_SUB];
    dnnl::memory::dims dim_nc = {loss_sub.get_desc().dims()[0], 1};

    // Multiply by 2 using eltwise_linear

    auto loss_diff_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto loss_diff = memory(loss_diff_md, eng);

    auto linear_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                loss_diff_md, 2.f, 0.f);
    auto linear_pd = dnnl::eltwise_forward::primitive_desc(linear_desc, eng);

    net.push_back(dnnl::eltwise_forward(linear_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sub},
                        {DNNL_ARG_DST, loss_diff}});

    return net.size() - 1;
}


/*
int binaryCrossEntropyLoss(dnnl::memory y_hat, dnnl::memory y_true, 
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{

    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_nc_tt = {1, y_hat.get_desc().dims()[0]};
    dnnl::memory::dims dim_scalar = {1};

    // 1) Perform elementwise log on y_hat
    
    auto y_hat_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_log = memory(y_hat_log_md, eng);

    auto y_hat_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat.get_desc(), 0.f, 0.f);
    auto y_hat_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_log}});

    // 2) Perform elementwise linear on y_hat with alpha = -1; beta = 1 ie. 1-y_hat

    auto y_hat_inv_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv = memory(y_hat_inv_md, eng);

    auto y_hat_inv_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_hat.get_desc(), -1.f, 1.f);
    auto y_hat_inv_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_inv}});

    // 3) Perform log on previously obtained element ie. log(1-y_hat)

    auto y_hat_inv_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv_log = memory(y_hat_inv_log_md, eng);

    auto y_hat_inv_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat_inv.get_desc(), 0.f, 0.f);
    auto y_hat_inv_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_inv},
                        {DNNL_ARG_DST, y_hat_inv_log}});


    // 4) Perform elementwise linear on y_true with alpha = -1; beta = 1 ie. 1-y_true

    auto y_true_inv_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_true_inv = memory(y_true_md, eng);

    auto y_true_inv_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true.get_desc(), -1.f, 1.f);
    auto y_true_inv_pd = dnnl::eltwise_forward::primitive_desc(y_true_inv_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_true_inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true},
                        {DNNL_ARG_DST, y_true_inv}});

    // 5) Perform inner_product(y_true, log(y_hat))

    auto add1_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add1 = memory(add1_md, eng);

    auto ip1_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, y_true_md,
                                                y_hat_log_md, NULL, add1_md);

    // nullptr, since we have no attr (see oneDNN primitive_desc API reference)
    auto ip1_pd = dnnl::inner_product_forward::primitive_desc(
        ip1_desc, nullptr, eng);

    net.push_back(dnnl::inner_product_forward(ip1_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true},
                        {DNNL_ARG_WEIGHTS, y_hat_log},
                        {DNNL_ARG_DST, add1}});


    // 6) Perform inner_product(1-y_true, log(1-y_hat))

    auto add2_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add2 = memory(add2_md, eng);

    auto matmul2_desc = dnnl::matmul::desc(y_true_inv_reordered_md, y_hat_inv_log_md, NULL, add2_md);
    auto matmul2_pd = dnnl::matmul::primitive_desc(matmul2_desc, eng);

    net.push_back(dnnl::matmul(matmul2_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_inv_reordered},
                        {DNNL_ARG_WEIGHTS, y_hat_inv_log_md},
                        {DNNL_ARG_DST, add2}});

    // 7) Sum obtained values (should be scalars)

    auto loss_sum_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss_sum = memory(loss_sum_md, eng);

    std::vector<dnnl::memory::desc> add_vector_md = {add1_md, add2_md};
    std::vector<dnnl::memory> add_vector = {add1, add2};
    std::vector<std::float> scales = {1.f, 1.f};

    auto loss_sum_pd = dnnl::sum::primitive_desc(scales, add_vector_md, eng);

    net.push_back(dnnl::sum(loss_sum_pd));

    std::unordered_map<int, memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sum});
    for (int i = 0; i<add_vector.size(); i++){
        sum_args.insert({DNN_ARG_SRC + i, add_vector[i]});
    }

    net_args.push_back(sum_args);
    

    // 8) Divide by -N using elementwise linear

    auto loss_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss = memory(loss_md, eng);

    float batch_size_inv = 1/y_true.dims()[0];

    auto loss_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -batch_size_inv, 0.f);
    auto loss_pd = dnnl::eltwise_forward::primitive_desc(loss_desc, eng);

    net.push_back(dnnl::eltwise_forward(loss_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sum},
                        {DNNL_ARG_DST, loss}});   
    
    return net.size() - 1;

}

int binaryCrossEntropyLoss_back(std::vector<dnnl::primitive> net_fwd, 
                           int finish,
                           std::vector<dnnl::primitive> &net,
                           std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                           dnnl::engine eng)
{
    // we want to start from the first layer of the BCE when dealing with the offset
    int start = finish - BCE_NORM;

    dnnl::memory loss = net_fwd[start+BCE_NORM][DNNL_ARG_DST];
    dnnl::memory loss_sum = net_fwd[start+BCE_NORM][DNNL_ARG_SRC];

    dnnl::memory y_true = net_fwd[start+BCE_REORDER][DNNL_ARG_FROM];
    dnnl::memory y_true_reordered = net_fwd[start+BCE_REORDER][DNNL_ARG_TO];

    dnnl::memory y_hat = net_fwd[start+BCE_LOG][DNNL_ARG_SRC];
    dnnl::memory y_hat_log = net_fwd[start+BCE_LOG][DNNL_ARG_DST];

    dnnl::memory 

    dnnl::memory::dims dim_nc = {y_hat.get_desc().dims()[0], 1};
    dnnl::memory::dims dim_nc_tt = {1, y_hat.get_desc().dims()[0]};
    dnnl::memory::dims dim_scalar = {1};

    // Backward flow starts from last element of forward flow

    // Divide by -N using elementwise linear

    auto loss_diff_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss_diff = memory(loss_md, eng);

    float batch_size_inv = 1/y_true.dims()[0];

    auto loss_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -batch_size_inv, 0.f);
    auto loss_pd = dnnl::eltwise_forward::primitive_desc(loss_desc, eng);

    // Allocate memory

    auto loss_sum_diff = memory(loss_sum.get_desc(), eng);

    // Diff

    auto loss_bwd_desc = dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_linear,
                                                loss_diff_md, loss_sum.get_desc(), -batch_size_inv, 0.f);
    auto loss_bwd_pd = dnnl::eltwise_backward::primitive_desc(loss_bwd_desc, eng,
                                                                   loss_pd);
    
    
    net.push_back(dnnl::eltwise_backward(y_hat_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, loss_sum},
                        {DNNL_ARG_DIFF_SRC, loss_sum_diff},
                        {DNNL_ARG_DST, loss},
                        {DNNL_ARG_DIFF_DST, loss_diff},
                        });

    // Sum obtained values (should be scalars)

    auto loss_sum_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto loss_sum = memory(loss_sum_md, eng);

    std::vector<dnnl::memory::desc> add_vector_md = {add1_md, add2_md};
    std::vector<dnnl::memory> add_vector = {add1, add2};
    std::vector<std::float> scales = {1.f, 1.f};

    auto loss_sum_pd = dnnl::sum::primitive_desc(scales, add_vector_md, eng);

    net.push_back(dnnl::sum(loss_sum_pd));

    std::unordered_map<int, memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, loss_sum});
    for (int i = 0; i<add_vector.size(); i++){
        sum_args.insert({DNN_ARG_SRC + i, add_vector[i]});
    }

    net_args.push_back(sum_args);
    
    // Perform elementwise linear on y_hat with alpha = -1; beta = 1 ie. 1-y_hat

    auto y_hat_inv_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv = memory(y_hat_inv_md, eng);

    auto y_hat_inv_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_hat.get_desc(), -1.f, 1.f);
    auto y_hat_inv_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DST, y_hat_inv}});

    // Perform log on previously obtained element ie. log(1-y_hat)

    auto y_hat_inv_log_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_inv_log = memory(y_hat_inv_log_md, eng);

    auto y_hat_inv_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat_inv.get_desc(), 0.f, 0.f);
    auto y_hat_inv_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_inv_log_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_hat_inv_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_inv},
                        {DNNL_ARG_DST, y_hat_inv_log}});

    // Reorder y_true from nc to cn {batch, 1} -> {1, batch}

    auto y_true_reordered_md = dnnl::memory::desc(dim_nc_tt, dt::f32, tag::nc);
    auto y_true_reordered = memory(y_true_reordered_md, eng);

    net.push_back(dnnl::reorder(y_true, y_true_reordered));
    net_args.push_back({{DNNL_ARG_FROM, y_true},
                            {DNNL_ARG_TO, y_true_reordered}});

    // Perform elementwise linear on y_true_reordered with alpha = -1; beta = 1 ie. 1-y_true

    auto y_true_inv_reordered_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_true_inv_reordered = memory(y_true_inv_reordered_md, eng);

    auto y_true_inv_reordered_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear,
                                                y_true_reordered.get_desc(), -1.f, 1.f);
    auto y_true_inv_reordered_pd = dnnl::eltwise_forward::primitive_desc(y_true_inv_reordered_desc, eng);

    net.push_back(dnnl::eltwise_forward(y_true_inv_reordered_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_reordered},
                        {DNNL_ARG_DST, y_true_inv_reordered}});

    // Perform matmul(y_true_reordered, log(y_hat))

    auto add1_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add1 = memory(add1_md, eng);

    auto matmul1_desc = dnnl::matmul::desc(y_true_reordered_md, y_hat_log_md, NULL, add1_md);
    auto matmul1_pd = dnnl::matmul::primitive_desc(matmul1_desc, eng);

    net.push_back(dnnl::matmul(matmul1_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_reordered},
                        {DNNL_ARG{DNNL_ARG_DST, add1}});_WEIGHTS, y_hat_log_md},
                        


    // Perform matmul(1-y_true_reordered, log(1-y_hat))

    auto add2_md = dnnl::memory::desc(dim_scalar, dt::f32, tag::a);
    auto add2 = memory(add2_md, eng);

    auto matmul2_desc = dnnl::matmul::desc(y_true_inv_reordered_md, y_hat_inv_log_md, NULL, add2_md);
    auto matmul2_pd = dnnl::matmul::primitive_desc(matmul2_desc, eng);

    net.push_back(dnnl::matmul(matmul2_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_true_inv_reordered},
                        {DNNL_ARG_WEIGHTS, y_hat_inv_log_md},
                        {DNNL_ARG_DST, add2}});

    

    // If you are here you are done
  

    // Perform elementwise log on y_hat
    
    auto y_hat_log_diff_md = dnnl::memory::desc(dim_nc, dt::f32, tag::nc);
    auto y_hat_log_diff = memory(y_hat_log_diff_md, eng);

    // Allocate memory

    auto y_hat_diff = memory(y_hat.get_desc(), eng);

    // Recreate forward descriptor
    auto y_hat_log_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                                                y_hat.get_desc(), 0.f, 0.f);
    auto y_hat_log_pd = dnnl::eltwise_forward::primitive_desc(y_hat_log_desc, eng);

    // Diff

    auto y_hat_log_bwd_desc = dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_log,
                                                y_hat_log_diff_md, y_hat.get_desc(), 0.f, 0.f);
    auto y_hat_log_bwd_pd = dnnl::eltwise_backward::primitive_desc(y_hat_log_bwd_desc, eng,
                                                                   y_hat_log_pd);
    
    
    net.push_back(dnnl::eltwise_backward(y_hat_log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat},
                        {DNNL_ARG_DIFF_SRC, y_hat_diff},
                        {DNNL_ARG_DST, y_hat_log},
                        {DNNL_ARG_DIFF_DST, y_hat_log_diff},
                        });

    
    return net.size() - 1;

}

*/
