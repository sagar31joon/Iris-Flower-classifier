Welcome to the Iris_flower_classification_model
To run the program, write "python3 main.py" in the terminal and hit "ENTER".


iris_project/
│
├── dataset/
│   ├── iris_raw.csv
│   └── iris_processed.csv
│
├── models/
│   ├── model_LR.pkl
│   ├── model_DT.pkl
│   ├── model_SVM.pkl
│   └── scaler.pkl
│
├── src/
│   ├── prepare_data.py
│   ├── train_LR.py
│   ├── train_decision_tree.py
│   ├── train_SVM.py
│   ├── predict_LR.py
│   ├── predict_decision_tree.py
│   ├── predict_SVM.py
│
├── samples/
│   ├──Visualisation_iris_raw.py
│
└── main.py


* Scaler (sc) used for :
    1. Preventing model to become biased (on long range features and ignore small range features or visa-versa).
    2. Acts as value transformer for raw user inputs to scaled values (on which model is trained upon).