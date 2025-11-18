import os

dataset_path = "dataset/encoded.csv"
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
print(dataset_name)