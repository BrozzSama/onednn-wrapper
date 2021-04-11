import numpy as np
import os

data_path = "data/"
dataset_path = "shuffled_data/"
empty_patches_path =  os.path.join(data_path, "skull_32_empty.npy")
vessel_patches_path = os.path.join(data_path, "skull_32_vessel.npy")

full_dataset_path = os.path.join(dataset_path, "full_dataset.npy")
label_dataset_path = os.path.join(dataset_path, "labels.npy")

# NHW format
empty_patches = np.load(empty_patches_path)
vessel_patches = np.load(vessel_patches_path)

# Create ground truth
empty_patches_label = np.zeros(empty_patches.shape[0])
vessel_patches_label = np.ones(vessel_patches.shape[0])

full_dataset = np.concatenate((empty_patches, vessel_patches), axis=0)
full_labels = np.concatenate((empty_patches_label, vessel_patches_label))

shuffle_idx = np.random.permutation(range(full_dataset.shape[0]))

full_dataset_shuffled = full_dataset[shuffle_idx, :, :]
full_labels_shuffled = full_labels[shuffle_idx]

np.save(full_dataset_path, full_dataset_shuffled)
np.save(label_dataset_path, full_labels_shuffled)


