# SVM model with scaler object
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



df = pd.read_csv("dataset/iris_processed.csv") #loading processed dataset
print(df.head())

data_value = df.values #slicing dataset
x = data_value[:, 0:4]
y = data_value[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #train and test split

with open("models/scaler.pkl", "rb") as f: #loading scaler model
    scaler = pickle.load(f)
print("Scaler model 'scaler.pkl' loaded")

x_train = scaler.transform(x_train) #scaling the data values
x_test = scaler.transform(x_test)

model_svc = SVC() #initialising SVC model
model_svc.fit(x_train, y_train) #training model

predict = model_svc.predict(x_test) # testing
accuracy = accuracy_score(y_test, predict)
for i in range(len(predict)): #loop for checking results
    print(y_test[i], predict[i])
print("Support Vector Classifier model accuracy : ", accuracy*100) #printing accuracy

with open("models/model_SVC.pkl", "wb") as f: #saving model as model_SVC.pkl
    pickle.dump(model_svc, f)
print("Support Vector Classifier saved as 'model_SVC.pkl'")










