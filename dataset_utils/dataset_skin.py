import numpy as np
import pandas as pd
import os

data_path = "data/SkinSegmentation.txt"

df = pd.read_csv(data_path, delimiter="\t")

# NHW format
features = df.iloc[:, 0:-1].to_numpy(copy=True, dtype="float64")
label = df.iloc[:, -1].to_numpy(copy=True, dtype="float64").reshape([-1,1])

np.save("data/features_skin.npy", features)
np.save("data/features_label.npy", label)


