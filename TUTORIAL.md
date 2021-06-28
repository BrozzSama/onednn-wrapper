# Fully connected layer tutorial

This tutorial will cover all the aspects of training a simple neural network using oneDNN. It is divided into three parts and follows the structure of the example onednn_training_skin.cpp.

- Pipeline creation: which explains how to create the forward and backward streams, as well as how to update the weights
- Output generation: this covers how to retrieve data from a oneAPI engine

## Forward pass

The forward pass is built as follows:

        int fc1_output_size = 5;
        Dense fc1(fc1_output_size, input_memory, net_fwd, net_fwd_args, eng);
        Eltwise relu1(dnnl::algorithm::eltwise_relu, 0.f, 0.f, fc1.arg_dst,
                            net_fwd, net_fwd_args, eng);
        int fc2_output_size = 1;
        Dense fc2(fc2_output_size, relu1.arg_dst, net_fwd, net_fwd_args, eng);    
        Eltwise sigmoid1 (dnnl::algorithm::eltwise_logistic, 0.f, 0.f, fc2.arg_dst,
                            net_fwd, net_fwd_args, eng);
        binaryCrossEntropyLoss loss(sigmoid1.arg_dst, labels_memory, net_fwd, net_fwd_args, eng);

The pipeline is quite simple, first, we have a Dense layer that takes as input the input_memory onto which we wrote the data using the DataLoader class. This first layer is activated by a relu function which is an Eltwise primitive. The output of the relu1 Eltwise is then passed to the second fully connected layers by using the Eltwise::arg_dst pubic member, which exposes the memory location of the output; again we use Eltwise to provide the activation function, in this case, we chose sigmoid. Finally, we have the binaryCrossEntropyLoss which computed the loss starting from the probabilities of the sigmoid and the labels_memory.

### Forward inference

To have inference on validation we use the exact same scheme as before with different variable names.

## Backward data pass

The backward data pass is a bit more complex to understand due to the fact that it forces us to see the pipeline the other way around. 

        binaryCrossEntropyLoss_back loss_back(sigmoid1.arg_dst, labels_memory, net_bwd_data, net_bwd_data_args, eng);
        Eltwise_back sigmoid1_back_data(dnnl::algorithm::eltwise_logistic, 0.f, 0.f, sigmoid1, 
                                         loss_back.arg_dst, net_bwd_data, net_bwd_data_args, eng);
        Dense_back_data fc2_back_data(sigmoid1_back_data.arg_diff_src, fc2, net_bwd_data, net_bwd_data_args, eng);
        Eltwise_back relu1_back_data(dnnl::algorithm::eltwise_relu, 0.f, 0.f, relu1, 
                                         fc2_back_data.arg_diff_src, net_bwd_data, net_bwd_data_args, eng);
        Dense_back_data fc1_back_data(relu1_back_data.arg_diff_src, fc1, net_bwd_data, net_bwd_data_args, eng);

First, we compute the gradient of the cross-entropy loss by providing the labels and the output of the sigmoid. This computation is passed as Eltwise_back#arg_diff_dst, since the Eltwise_back operation will compute the Eltwise_back::arg_diff_src from it. The same operation is done with Dense_back_data in the case of fc2, and so on.

## Backward weights pass

The backward weights pass can be put in any order we want since they are independent computations, in this specific scenario we simply preserve the order from the backward data pass.

    Dense_back_weights fc2_back_weights(sigmoid1_back_data.arg_diff_src, fc2, net_bwd_weights, net_bwd_weights_args, eng);
    Dense_back_weights fc1_back_weights(relu1_back_data.arg_diff_src, fc1, net_bwd_weights, net_bwd_weights_args, eng);

## Weights update

To update the gradient through SGD we can use the updateWeights_SGD class, which simply subtracts the gradient multiplied by the learning rate to each weight and bias.

        updateWeights_SGD(fc1.arg_weights, 
                   fc1_back_weights.arg_diff_weights,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc1.arg_weights, fc1_inf.arg_weights,
                   net_bwd_weights, net_bwd_weights_args, eng);
        updateWeights_SGD(fc2.arg_weights, 
                   fc2_back_weights.arg_diff_weights,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc2.arg_weights, fc2_inf.arg_weights,
                   net_bwd_weights, net_bwd_weights_args, eng);
        updateWeights_SGD(fc1.arg_bias, 
                   fc1_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc1.arg_bias, fc1_inf.arg_bias,
                   net_bwd_weights, net_bwd_weights_args, eng);
        updateWeights_SGD(fc2.arg_bias, 
                   fc2_back_weights.arg_diff_bias,
                   learning_rate, net_sgd, net_sgd_args, eng);
        Reorder(fc2.arg_bias, fc2_inf.arg_bias,
                   net_bwd_weights, net_bwd_weights_args, eng);

The syntax is pretty intuitive, the only tricky part to understand here is the use of the Reorder primitive. In this specific scenario, we have a Reorder for each weight and bias vector because we want to propagate the changes made in training to the validation pipeline.

## Training Loop

Finally, once the pipelines are ready, it is time to train our network. We simply have a loop that iterates the training n_iter times. 

    while (n_iter < max_iter)
        {

                for (size_t i = 0; i < net_fwd.size(); ++i)
                        net_fwd.at(i).execute(s, net_fwd_args.at(i));

                for (size_t i = 0; i < net_bwd_data.size(); ++i)
                        net_bwd_data.at(i).execute(s, net_bwd_data_args.at(i));
                for (size_t i = 0; i < net_bwd_weights.size(); ++i)
                        net_bwd_weights.at(i).execute(s, net_bwd_weights_args.at(i));
                for (size_t i = 0; i < net_sgd.size(); ++i)
                        net_sgd.at(i).execute(s, net_sgd_args.at(i));

                if (n_iter % step == 0){  
                        for (size_t i = 0; i < net_fwd_inf.size(); ++i)
                                net_fwd_inf.at(i).execute(s, net_fwd_inf_args.at(i));

                }

                skin_data.write_to_memory(input_memory, labels_memory);

                n_iter++;
        }

the dnnl::primitive::execute method of each primitive executed the primitive for each pipeline. Once an iteration is fully complete we use DataLoader::write_to_memory to change the batch.

## Changing parameters

Through the included nholman::json library it is possible to parametrize your network and avoid recompiling the code every time there is a minor change. Currently the onednn_training_skin.cpp example supports:

- iterations: which controls the number of iterations
- dataset_path: which provides the path to the feature file
- labels_path: which provides the path to the labels
- dataset_path_val: which provides the path to the feature file for validation
- labels_path_val: which provides the path to the labels file for validation
- loss_filename: which provides the path for the loss vector that will be produced in the output
- loss_inf_filename: which provides the path for the validation loss vector that will be produced in the output
- val_predicted_filename: which provides the predictions on the validation set
- minibatch_size: which provides the batch size
- step: which provides after how many iterations the data should be printed
- learning_rate: which provides the learning rate for SGD

## Debugging

Unfortunately debugging is not available on GPU. To print the output of a vector inside a oneAPI engine we can use the read_from_dnnl_memory() function from intel_utils.h that moves data from oneAPI engine inside the RAM. To print a vector one can use the print_vector utility. Note that once read_from_dnnl_memory() is called, one must also use the wait() method on the engine, in order to wait for all the queued operations to finish.

## Performance measurements

By setting the DNNL_VERBOSE=1 flag in the run script as follows:

        DNNL_VERBOSE=1 ./dpcpp/onednn-training-$1-cpp gpu $2

it is possible to enable verbose output. What this means is that oneAPI will provide additional information about all of the operations that are being run on the engine. This is particularly useful since it allows to check if all the memory types are proper and additionally it allows to compare performance since it provides the execution time. It must be noted that the output is essentially a non-standard CSV format, therefore after some cleaning up it can be analyzed using pandas. To do so it is sufficient to strip all the additional output by means of the grep command
        cat run_cpu_xxx | grep dnnl_verbose > my_clean_output.log
Once this is done to have a file that is readable by pandas it is sufficient to remove the first few lines that only tell us what engines are available:
        dnnl_verbose,info,oneDNN v2.2.0 (commit 7489a2ea4c14c94f5bbe7d5fc774f722444cdd3f)
        dnnl_verbose,info,cpu,runtime:DPC++
        dnnl_verbose,info,cpu,isa:Intel AVX-512 with Intel DL Boost
        dnnl_verbose,info,gpu,runtime:DPC++
        dnnl_verbose,info,cpu,engine,0,backend:OpenCL,name:Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz,driver_version:2021.11.3
        dnnl_verbose,info,gpu,engine,0,backend:Level Zero,name:Intel(R) Iris(R) Xe MAX Graphics [0x4905],driver_version:1.0.19310
        dnnl_verbose,info,gpu,engine,1,backend:Level Zero,name:Intel(R) Iris(R) Xe MAX Graphics [0x4905],driver_version:1.0.19310
        dnnl_verbose,info,gpu,engine,2,backend:Level Zero,name:Intel(R) Iris(R) Xe MAX Graphics [0x4905],driver_version:1.0.19310
And then edit the header by modifying:

        dnnl_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time

into:
        dnnl_verbose,operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time

Now using pandas we can import the data frame and do analysis. Some examples are available in the analysis.ipynb Jupyter Notebook.


