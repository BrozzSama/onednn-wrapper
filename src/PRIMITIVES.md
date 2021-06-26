# Primitives

This page provides examples on how to use the implemented primitives and the general procedure to implement a new one. 

## General usage

The primitive wrapper itself is quite simple, it is a class that exposes as public member the dnnl::memory objects associated to a primitive. For example, in a Dense layer we have:
- DNNL_ARG_SRC which is mapped to the public member Dense::arg_src
- DNNL_ARG_DST which is mapped to the public member Dense::arg_dst
- DNNL_ARG_WEIGHTS which is mapped to the public member Dense::arg_weights
- DNNL_ARG_BIAS which is mapped to the public member Dense::arg_bias

Hence, this means that once we instantiate a Dense class we can simpy call .arg_X to access any memory handler we need.

## General implementation

To implement a primitive coming from the oneDNN toolkit the first step is to read the Documentation in order to understand what members we will have and what inputs we need to instantiate the class. Our specific design choice takes uses 5 header files to instantiate the primitives (or a combination of primitives for example in the case of a Loss Function):
- layers_fwd.h which contains the wrappers for the forward operations: Dense, Convolution...
- primitive_wrappers.cpp which contains simple primitives such as: Reorder, Eltwise...
- losses.h which contains the loss functions as well as their gradients
- layers_bwd_data.h which contains the wrapper for the backward data operations corresponding to the primitives declared in layers_fwd.h and primitive_wrappers.cpp
- layers_bwd_weights.h which contains the wrapper for the backward weights operations corresponding to the primitives declared in layers_fwd.h and primitive_wrappers.cpp

Given the previous description it is clear that depending on the primitive we are implementing we might need to implement the corresponding backward operation. We will now looking at the example of the Dense layer.

### Implementing a Dense Layer 

The Dense layer is the most complete example that allows to explain the inner working of the oneDNN wrapper. First we will implement the forward operation by declaring the prototype in the layers_fwd.h file as follows 

    class Dense{
        public:
            dnnl::memory arg_src;
            dnnl::memory arg_dst; 
            dnnl::memory arg_bias; 
            dnnl::memory arg_weights; 
            Dense(dnnl::memory::dims src_dims, 
            int fc_output_size,
            dnnl::memory input,
            std::vector<dnnl::primitive> &net,
            std::vector<std::unordered_map<int, dnnl::memory>> &net_args,
            dnnl::engine eng);
        private:
            
    };

The dnnl::memory variable instantiate an object that can be then associated to a memory location, the constructor Dense::Dense takes as input all the information that we need to create the primitive.
The actual forward operation is declared inside layers_fwd.cpp in a way that's very similar to the examples in oneDNN. 

The first thing we do is obtain the dimensions from the input parameters

    dnnl::memory::dims weights_dims_fc;
    dnnl::memory::dims bias_dims_fc = {fc_output_size};
    dnnl::memory::dims dst_dims_fc = {src_dims[0], fc_output_size};  

Using the src_dims vector we are able to 



