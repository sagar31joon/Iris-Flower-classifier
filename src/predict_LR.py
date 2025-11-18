#prediction file for Logistic Regression model
import numpy as np
import pickle

with open("models/scaler.pkl", "rb") as f: #loading scaler model
    scaler = pickle.load(f)

with open("models/model_LR.pkl", "rb") as f: #loading Logistic Regression model
    model_LR = pickle.load(f)

#user inputs : 
sepal_length = float(input("Enter Sepal length : "))
sepal_width = float(input("Enter Sepal width : "))
petal_length = float(input("Enter Petal length : "))
petal_width = float(input("Enter Petal width : "))

#converting to array
user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#scaling the input
user_data_scaled = scaler.transform(user_data)

#prediction
prediction = model_LR.predict(user_data_scaled)[0]

#Decoding the output (0,1,2 to species name)
species_maping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
print("Model used : Logistic Regression")
print("Predicted Species : ", species_maping[prediction])