import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

data_path = "data/Admission_Predict.csv"

df = pd.read_csv(data_path, delimiter=",")

# NH format
myScaler = MinMaxScaler()
features = df.iloc[:, 1:-1].to_numpy(copy=True, dtype="float64")
features = myScaler.fit_transform(features)
label = df.iloc[:, -1].to_numpy(copy=True, dtype="float64").reshape([-1,1])

shuffle_idx = np.random.permutation(range(features.shape[0]))

features = features[shuffle_idx, :]
label = label[shuffle_idx]

feature_file = open("data/features_admission.txt", "w")
for row in features:
    np.savetxt(feature_file, row, fmt="%f")

feature_file.close()

label_file = open("data/label_admission.txt", "w")
for row in label:
    np.savetxt(label_file, row, fmt="%f")

label_file.close()

np.save("data/features_admission.npy", features)
np.save("data/label_admission.npy", label)


