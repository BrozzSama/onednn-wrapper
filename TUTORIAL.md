# Fully connected layer tutorial

This tutorial will cover all the aspects of training a simple neural network using oneDNN. It is divided in three parts and follows the structure of the example onednn_training_skin.cpp.

- Data loading: which covers the basics of loading data in memory using the DataLoader Class
- Pipeline creation: which explains how to create the forward and backward streams, as well as how to update the weights
- Output generation: which covers how to retrieve data from a oneAPI engine