# Issues and To-do list

- Currently training a CNN is not possible, it is unclear if it is due to our implementation or due to the  instability of the oneDNN toolkit. What has been noticed is that generally we have a weird periodic behaviour of the loss and many outlier values at different point in the layers.
- The data loader, although working, could be further improved by:
    - Choosing a more optimized format that also provides the shape in the header
    - Making sure that when we are at the end of the dataset and we are switching batch we are doing so in a "circular fashion" ie. in a dataset of size 80 samples with batch 15, once we get to 75 it would be better if we were able to create a batch that has 75-80 and 1-10.

