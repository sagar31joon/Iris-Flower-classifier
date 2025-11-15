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
│   ├── prepare_data.py (done)
│   ├── train_LR.py (done)
│   ├── train_decision_tree.py (done)
│   ├── train_SVM.py 
│   ├── predict_LR.py
│   ├── predict_decision_tree.py
│   ├── predict_SVM.py
│
├── samples/
│
└── main.py


* Scaler (sc) used for :
    1. Preventing model to become biased (on long range features and ignore small range features or visa-versa).
    2. Acts as value transformer for raw user inputs to scaled values (on which model is trained upon).