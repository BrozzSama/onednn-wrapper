import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_path = "data/SkinSegmentation.txt"

df = pd.read_csv(data_path, delimiter=",")

# NH format
myScaler = MinMaxScaler()
features = df.iloc[:, 0:-1].to_numpy(copy=True, dtype="float64")
features = myScaler.fit_transform(features)
label = df.iloc[:, -1].to_numpy(copy=True, dtype="float64").reshape([-1,1])

features, features_val, labels, labels_val = train_test_split(features, label, test_size = 0.25, random_state = 42)

feature_file = open("data/features_skin.txt", "w")
for row in features:
    np.savetxt(feature_file, row, fmt="%f")

feature_file.close()

label_file = open("data/label_skin.txt", "w")
for row in labels:
    np.savetxt(label_file, row, fmt="%f")

label_file.close()

feature_file_val = open("data/features_skin_val.txt", "w")
for row in features_val:
    np.savetxt(feature_file_val, row, fmt="%f")

feature_file_val.close()

label_file_val = open("data/label_skin_val.txt", "w")
for row in labels_val:
    np.savetxt(label_file_val, row, fmt="%f")

label_file_val.close()




