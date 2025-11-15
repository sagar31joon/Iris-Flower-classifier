import numpy as np
import pandas as pd
from datasets import load_dataset

#iris_dataset = load_dataset("sasageooo/iris-flower-classification") #loading dataset from hugging face

#iris_dataset['train'].to_csv("iris_dataset.csv") #saving dataset as CSV file

df = pd.read_csv("dataset/iris_raw.csv")
#print(df)
print("\nRaw dataset : ")
print(df.head()) 
#print(df.info()) 
#print(df.describe()) 

from sklearn.preprocessing import LabelEncoder #label encoding
le = LabelEncoder()
df['species'] = le.fit_transform(df['species']) #encoding species column
print("\nEncoded dataset : ")
print(df.head())
print (df.shape) #rows x column numbers

df.to_csv("dataset/iris_processed.csv", index=False) #creating processed dataset file for model
