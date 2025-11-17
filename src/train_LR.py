#Logistic Refression model and sclaler model initialiser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset/iris_processed.csv") #loading processed dataset file
print(df.head())

data_value = df.values #slicing dataset
x = data_value[:, 0:4]
y = data_value[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #test and train split
print("Train split X shape:", x_train.shape)
print("Test split X shape :", x_test.shape)

scaler = StandardScaler() #scaling
x_train = scaler.fit_transform(x_train) #standardise and transform data
x_test = scaler.transform(x_test) # only transform data

print("\nScaled TRAIN:")
print(x_train)

print("\nScaled TEST:")
print(x_test)

with open("models/scaler.pkl", "wb") as f: #saving scaler model for reuse
    pickle.dump(scaler, f)
print("Scaler model saved as 'scaler.pkl'")

model_LR = LogisticRegression(max_iter=500) #initialising model
model_LR.fit(x_train, y_train) #training model

predict = model_LR.predict(x_test) #testing x_test 
for i in range(len(predict)): #loop for showing results
    print(y_test[i], predict[i])

#Evaluation
accuracy = accuracy_score(y_test, predict) #Accuracy
cm = confusion_matrix(y_test, predict) #Confusion Matrix
report = classification_report(y_test, predict, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) #Classification report
print("\nLR Accuracy :", accuracy*100)
print("\nLR Confusion Matrix : \n", cm)
print("\nLR Classification Report : \n", report)



model_save = input("Do you want to save this LR model ? (Y/N)")
match model_save:
    case ("Y" | "y"):
        with open("models/model_LR.pkl", "wb") as f: #saving the LR model as model_LR.pkl
            pickle.dump(model_LR, f)
        print("Logistic Regression model saved as 'model_LR.pkl'")
        print("Model saved as 'model_LR.pkl")
    case ("N" | "n"):
        print("Very well")












