#Decision Tree model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("dataset/iris_processed.csv") #loading data
print(df.head())

data_value = df.values #slicing dataset
x = data_value[:, 0:4]
y = data_value[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #splitting dataset 
print("Train split X shape", x_train.shape)
print("Test split X shape", x_test.shape)

with open("models/scaler.pkl", "rb") as f: #load scaler model
    scaler = pickle.load(f)
print("Scaler model 'scaler.pkl' loaded ")

x_train = scaler.transform(x_train) #scaling the data values
x_test = scaler.transform(x_test)

model_DT = DecisionTreeClassifier() #initialising model
model_DT.fit(x_train, y_train) #training model

predict = model_DT.predict(x_test) #testing
accuracy = accuracy_score(y_test, predict)
for i in range(len(predict)): #loop for results
    print(y_test[i], predict[i])
print("Decision Tree Classifier accuracy : ", accuracy*100)

with open("models/model_DT.pkl", "wb") as f:
    pickle.dump(model_DT, f)
print ("Decision Tree Classifier model saved as 'model_DT.pkl'")
