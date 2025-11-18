#Prediction file for Decision Tree Classifier model
import numpy as np
import pickle

with open("models/scaler.pkl", "rb") as f: #loading scaler
    scaler = pickle.load(f)

with open("models/model_DT.pkl", "rb") as f: #loading DT model
    model_DT = pickle.load(f)

#User inputs
sepal_length = float(input("Enter Sepal length : "))
sepal_width = float(input("Enter Sepal width : "))
petal_length = float(input("Enter P1etal length : "))
petal_width = float(input("Enter Petal width : "))

#converting to array
user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#scaling
user_data_scaled = scaler.transform(user_data)

#prediction
prediction = model_DT.predict(user_data_scaled)[0]

#decoding
species_maping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
print("Model used : Decision Tree Classifier")
print("Predicted Species : ", species_maping[prediction])