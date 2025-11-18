
# IRIS-ML-PIPELINE

A fully modular, end-to-end Machine Learning pipeline for classifying **Iris flower species** using three classical ML algorithms:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Support Vector Classifier (SVC)**

This project includes **full preprocessing**, **training**, **evaluation**, **model comparison**, and a **CLI-based prediction system**.  
All modules are cleanly separated for reusability and easy future expansion.

---

## 📁 Project Structure

```
Iris-ML-Pipeline/
│
├── dataset/
│   ├── iris_raw.csv
│   ├── encoded.csv
│   ├── x_train.npy
│   ├── x_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── dataset_name.txt
│   └── model_comparison_iris_raw.csv
│
├── models/
│   ├── scaler.pkl
│   ├── model_LR.pkl
│   ├── model_DT.pkl
│   └── model_SVC.pkl
│
├── src/
│   ├── prepare_data.py
│   ├── train_LR.py
│   ├── train_decision_tree.py
│   ├── train_SVM.py
│   ├── predict_LR.py
│   ├── predict_decision_tree.py
│   ├── predict_SVM.py
│   └── model_compare.py
│
├── samples/
│   └── Visualisation_iris_raw.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Features

### ✔ Complete Preprocessing Pipeline  
- Reads raw CSV (`iris_raw.csv`)
- Label-encodes species
- Train/test split (configurable)
- Feature scaling via `StandardScaler`
- Saves:
  - Encoded dataset
  - Scaled `.npy` arrays
  - `scaler.pkl`
  - Dataset name text file

---

### ✔ Three ML Models with Evaluation  
Each model script computes:

- **Training time**
- **Batch inference time**
- **Single-sample inference time**
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

All trained models are saved in `/models/` as:

```
model_LR.pkl
model_DT.pkl
model_SVC.pkl
```

---

### ✔ Model Comparison Script  
`model_compare.py`:

- Loads all 3 trained models
- Evaluates on the same test set
- Prints structured comparison table
- Saves CSV as:

```
model_comparison_<dataset_name>.csv
```

Example:
```
model_comparison_iris_raw.csv
```

---

### ✔ CLI-Based Prediction System  
Run `main.py`:

```
1. Logistic Regression
2. Decision Tree Classifier
3. Support Vector Classifier
4. Exit
```

The chosen script:

- Takes 4 numeric inputs  
- Scales using `scaler.pkl`  
- Predicts species  
- Decodes class number → species name  

---

### ✔ (Optional) Dataset Visualization  
`Visualisation_iris_raw.py` performs simple EDA:

- Head of dataset  
- Info & statistics  
- Seaborn pairplot showing feature relationships  

Run:
```
python3 samples/Visualisation_iris_raw.py
```

---

## 🧠 Models Used

### 1. Logistic Regression  
Lightweight baseline classifier.

### 2. Decision Tree Classifier  
Non-linear, interpretable classifier.

### 3. Support Vector Classifier (SVC)  
Margin-based classifier suitable for smaller datasets.

---

## 🔧 Technologies Used
- Python 3.12  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Pickle  

---

## 🏃‍♂️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
```bash
python3 src/prepare_data.py
```
This will:
- Encode dataset  
- Scale features  
- Save training & testing arrays  
- Save scaler  
- Save dataset name  

### 3. Train all models
```bash
python3 src/train_LR.py
python3 src/train_decision_tree.py
python3 src/train_SVM.py
```

### 4. Run model comparison (optional)
```bash
python3 src/model_compare.py
```

### 5. Start CLI prediction
```bash
python3 main.py
```

---

## 🔍 Example CLI Output

```
🌸 IRIS FLOWER CLASSIFICATION SYSTEM 🌸
---------------------------------------

Choose a model:
1. Logistic Regression
2. Decision Tree Classifier
3. Support Vector Classifier
4. Exit
```

---

## 📌 Notes
- Re-running training scripts will overwrite previous models (if user chooses to confirm the save).  
- Comparison CSV names are generated dynamically based on original dataset name.  
- You can plug in any dataset with the same structure (4 features + label).  
- Scripts are modular and can be reused for other ML projects.  

---

## 📝 License  
This project is made for educational and experimental purposes.  
Feel free use it.

