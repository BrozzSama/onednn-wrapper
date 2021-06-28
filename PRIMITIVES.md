# Primitives

This page provides examples of how to use the implemented primitives and the general procedure to implement a new one. 

## General usage

The primitive wrapper itself is quite simple, it is a class that exposes as public members the dnnl::memory objects associated with a primitive. For example, in a Dense layer we have:
- DNNL_ARG_SRC which is mapped to the public member Dense::arg_src
- DNNL_ARG_DST which is mapped to the public member Dense::arg_dst
- DNNL_ARG_WEIGHTS which is mapped to the public member Dense::arg_weights
- DNNL_ARG_BIAS which is mapped to the public member Dense::arg_bias

Hence, this means that once we instantiate a Dense class we can simply call .arg_X to access any memory handler we need.

To instantiate a primitive it is sufficient to instantiate a class of the proper type, for example:

    Dense fc1(fc1_output_size, input_memory, net_fwd, net_fwd_args, eng);


## General implementation

To implement a primitive coming from the oneDNN toolkit the first step is to read the documentation in order to understand what members we will have and what inputs we need to instantiate the class. Our specific design choice takes uses 5 header files to instantiate the primitives (or a combination of primitives for example in the case of a Loss Function):
- layers_fwd.h which contains the wrappers for the forward operations: Dense, Convolution...
- primitive_wrappers.h which contains simple primitives such as Reorder, Eltwise...
- losses.h which contains the loss functions as well as their gradients
- layers_bwd_data.h which contains the wrapper for the backward data operations corresponding to the primitives declared in layers_fwd.h and primitive_wrappers.cpp
- layers_bwd_weights.h which contains the wrapper for the backward weights operations corresponding to the primitives declared in layers_fwd.h and primitive_wrappers.cpp

Given the previous description, it is clear that depending on the primitive we are implementing we might need to implement the corresponding backward operation. We will now be looking at the example of the Dense layer.

### EXAMPLE: Implementing a Dense Layer 

The Dense layer is the most complete example that allows us to explain the inner working of the oneDNN wrapper. First, we will implement the forward operation by declaring the prototype in the layers_fwd.h file as follows 

    class Dense{
        public:
            dnnl::memory arg_src;
            dnnl::memory arg_dst; 
            dnnl::memory arg_bias; 
            dnnl::memory arg_weights; 
            Dense(int fc_output_size,
                dnnl::memory input,
                std::vector<dnnl::primitive> &net,
                std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
                dnnl::engine eng);
        private:
            
    };

The dnnl::memory variable instantiates an object that can be then associated to a memory location, the constructor Dense::Dense takes as input all the information that we need to create the primitive.
The actual forward operation is declared inside layers_fwd.cpp in a way that's very similar to the examples in oneDNN. 

The first thing we do is obtain the dimensions from the input parameters

    dnnl::memory::dims bias_dims_fc = {fc_output_size};
    dnnl::memory::desc src_md_fc = input.get_desc(); 
    dnnl::memory::dims src_dims_fc = src_md_fc.dims();

Here we have:
- bias_dims_fc which is simply a scalar that contains the output dimension
- src_md_fc which is the memory::desc associated with the input
- src_dims_fc which is the memory::desc::dims associated with the input. This vector contains the input dimension and can be indexed to obtain what we need.

The weights dimensions require a bit more work, in fact the depend on the dimensions of our input, which can be 2D in the case of a simple network with only Dense layers, or 3D if coming from a convolutional layer. To address this use the following logic which exploits the src_dims_fc vector:

    bool from_conv = (src_dims_fc.size() > 3); 
    if ( from_conv ){
        weights_dims_fc = {fc_output_size, src_dims_fc[1], src_dims_fc[2], src_dims_fc[3]};
    }
    else {
        weights_dims_fc = {fc_output_size, src_dims_fc[1]};
    }

Now that we have all the input dimensions we can create all the necessary dnnl::memory::desc. The dnnl::memory::desc object can then be passed to the dnnl::memory constructor to allocate the memory inside the chosen engine:

    auto bias_md_fc = dnnl::memory::desc(bias_dims_fc, dt::f32, tag::a);
    auto dst_md_fc = dnnl::memory::desc(dst_dims_fc, dt::f32, tag::nc);
    dnnl::memory::desc weights_md_fc;
    if ( from_conv ){
        weights_md_fc = dnnl::memory::desc(weights_dims_fc, dt::f32, tag::oihw)
    }
    else {
        weights_md_fc = dnnl::memory::desc(weights_dims_fc, dt::f32, tag::oi);
    }
    auto src_mem_fc = dnnl::memory(src_md_fc, eng);
    auto bias_mem_fc = dnnl::memory(bias_md_fc, eng);
    auto weights_mem_fc = dnnl::memory(weights_md_fc, eng);
    auto dst_mem_fc = dnnl::memory(dst_md_fc, eng);

The weights and biases are initialized using an RNG inside the local memory and are then written to the engine

    std::vector<float> fc_weights(product(weights_dims_fc));
    std::vector<float> fc_bias(product(bias_dims_fc));

    for (int i = 0; i<fc_weights.size(); i++){
        fc_weights[i] = norm_dist(generator);
    }
    
    for (int i = 0; i<fc_bias.size(); i++){
        fc_bias[i] = norm_dist(generator);
        //std::cout << fc_bias[i] << " ";
    }

    write_to_dnnl_memory(fc_bias.data(), bias_mem_fc);
    write_to_dnnl_memory(fc_weights.data(), weights_mem_fc);

The only interesting thing about this piece of code is the use of the product() function to obtain the "flattened" dimension and the write_to_dnnl_memory utility, which allows writing the float numbers in memory.

The primitive is created in a manner analogous to the memory, first, we create the dnnl::primitive::desc, then we instantiate the dnnl::primitive object

    auto fc_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, src_md_fc,
                                                weights_md_fc, bias_md_fc, dst_md_fc);
    auto fc_pd = dnnl::inner_product_forward::primitive_desc(fc_desc, eng);


We then check all the types to see if they are compatible with the instantiated primitive, and if not create a Reorder:

    src_mem_fc = checkType(fc_pd.src_desc(), input, net, net_args, eng);
    weights_mem_fc = checkType(fc_pd.weights_desc(), weights_mem_fc, net, net_args, eng);
    bias_mem_fc = checkType(fc_pd.bias_desc(), bias_mem_fc, net, net_args, eng);

The checkType has a different role in each case:
- In src_mem_fc we check if the input we are providing to the fully connected layer is compatible with the input expected by the FC;
- In weights_mem_fc and bias_mem_fc we are simply checking if we instantiated the memory correctly, this helps a lot when debugging issues since it allows us to check if we correctly implemented our primitive, ideally, we should see no reorders for weights and biases;

Finally, we expose the dnnl::memory objects by associating the pointers and append the primitive to the pipeline

    arg_src = src_mem_fc;
    arg_dst = dst_mem_fc;
    arg_weights = weights_mem_fc;
    arg_bias = bias_mem_fc;

    net.push_back(dnnl::inner_product_forward(fc_pd));
    net_args.push_back({{DNNL_ARG_SRC, src_mem_fc},
                        {DNNL_ARG_WEIGHTS, weights_mem_fc},
                        {DNNL_ARG_BIAS, bias_mem_fc},
                        {DNNL_ARG_DST, dst_mem_fc}});

