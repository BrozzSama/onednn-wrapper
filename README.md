oneDNN Wrapper 
===============

oneDNN wrapper is a project written in C++ which "wraps" oneDNN primitive into simpler and more usable classes, in order to implement Deep Neural Networks in a way that is more similar to TensorFlow or Pytorch. 

Getting Started
===============

The following tutorial will show the basic building blocks to build a fully connected network. A full working example based on the skin segmentation dataset (https://archive.ics.uci.edu/ml/datasets/skin+segmentation) is available in onednn_training_skin.cpp .

First we need to allocate all the required memory spaces for our model. In the example we consider a simple fully connected network with two layers, one with 5 hidden neurons and one with a single output neuron. To train any network we need to prepare three different pipelines:
- A forward pipeline, which simply provides inference (ie. the forward pass)
- A backward data pipeline, which computes the gradient of each input of the network with respect to the loss
- A backward weights pipeline, which starting from the gradient of the input and the output computes the gradient of the weights with respect to the loss
- An update weights pipeline, which performs Gradient Descent on the weights

## Creating a pipeline

To create a primitives pipeline we use the procedure described in the example onednn_training_skin.cpp, and in the oneDNN documentation. It is sufficient to allocate a vector of dnnl::primitive and a memory map

    std::vector<dnnl::primitive> net_fwd
    std::vector<std::unordered_map<int, memory>> net_fwd_args

The dnnl:: suffix can be omitted when we are in the dnnl namespace.

### Forward Pass

To create a primitive you instantiate a class. For every class the procedure is quite similar, you simply have to include different arguments depending on the primitive you are using. In this example we will explain how to create a primitive for a dense layer; all the information regarding the other primitives is available in the specific pages.

In a fully connected layer we will have to allocate:

- The weights 2D vector of size {OUTPUT, INPUT}, since we will have to do the dot product wT x
- The bias vector of size {OUTPUT}
- The output 2D vector of size {BATCH, OUTPUT}

This is done automatically by the wrapper and it is sufficient to create a class of type Dense with the proper arguments.

    Dense fc1(fc1_output_size, input_memory, net_fwd, net_fwd_args, eng);

Let's unpack this line a bit:
- fc1_src_dims provides the input dimensions as a dnnl::memory::dims vector
- fc1_output_size provides the output dimensions as a dnnl::memory::dims vector
- input_memory provides the dnnl::memory object containing the input
- net_fwd is the vector of dnnl::primitives which contain the full pipeline. The Dense::Dense class constructor will automatically append the correct primitive when instantiated
- net_fwd_args is the unordered map which provides the arguments for each primitives, again this is done automatically by the wrapper
- eng is the dnnl::engine that we are using

### Backward Data Pass

In a similar fashion to the Forward Pass we create the backward pass for the Dense layer:

    Dense_back_data fc1_back_data(relu1_back_data.arg_diff_src, fc1, net_bwd_data, net_bwd_data_args, eng);

Here we have:

- relu1_back_data.arg_diff_src which is the input argument ie. the gradient of the source with respect to the loss
- fc1 which is the original class containing the forward primitive
- net_bwd_data which is the vector of dnnl::primitives relative to the backward data pipeline
- net_bwd_data_args which is the memory map associated to net_bwd_data
- eng is the dnnl::engine that we are using

### Backward Weights Pass

Last but not least we have our backward weights pass. Here we will use all the gradients computed in the backward data pipeline to compute the gradients with respect to the weights. 
Again, considering the Dense layer we instantiate the backward weights pass as follows:

    Dense_back_weights fc1_back_weights(relu1_back_data.arg_diff_src, fc1, net_bwd_weights, net_bwd_weights_args, eng);

- relu1_back_data.arg_diff_src which is the input argument ie. the gradient of the source with respect to the loss
- fc1 which is the original class containing the forward primitive
- net_bwd_weights which is the vector of dnnl::primitives relative to the backward weights pipeline
- net_bwd_weights_args which is the memory map associated to net_bwd_data
- eng is the dnnl::engine that we are net_bwd_weights

## Data Loading

Data is loaded inside the oneAPI memory using the DataLoader class. It reads the file in flattended txt format, which means that the vectors are written in row-major order in a text file. An example on how to generate these files starting from a csv is available in dataset_utils. 

To instantiate a data loader the syntax is the following
    DataLoader skin_data(dataset_path, labels_path, batch, dataset_shape, eng);

- dataset_path is the path to the file containing the features
- labels_path is the path to the file containing the labels
- batch is the minibatch size
- dataset_shape is the shape of each feature, for example it can be a vector of the form {N, C, H} for an image or {C} for a simple feature vector that has length C
- eng is the oneAPI engine

Once the data loader class is instantiated we can simply use

    skin_data.write_to_memory(input_memory, labels_memory)

to load a batch of features and label inside the dnnl::memory objects: input_memory and labels_memory. The useful thing about this method is that it can be called as many times as we need, everytime we need a new mini-batch.

### Inference

To have inference we can instantiate an extra dataloader with a validation set. To have batch size as large as the sample size we can put -1.

    DataLoader skin_data(dataset_path, labels_path, -1, dataset_shape, eng)