# Brain Tumor Classification: Traditional ML vs. RNN

This repository implements a complete machine learning pipeline to classify brain tumors from a dataset of CT/MRI image features. The project explores the performance of classical algorithms (Logistic Regression, KNN, Decision Trees, Random Forest) against a deep learning approach (Recurrent Neural Network - RNN).

## 🧠 Project Overview

The objective is to accurately identify the presence of a brain tumor based on extracted image characteristics. The workflow includes comprehensive exploratory data analysis (EDA), rigorous preprocessing, and multi-model benchmarking.

### **1. Exploratory Data Analysis (EDA)**

* **Class Distribution:** Visualizes the balance between tumor and non-tumor cases using pie charts.
* **Feature Analysis:** Generates histograms for all numeric attributes to understand the data's spread and variance.
* **Data Integrity:** Conducts automated checks for missing values and duplicates.

### **2. Preprocessing & Engineering**

* **Encoding:** Transforms categorical metadata into numerical format using `LabelEncoder`.
* **Feature Selection:** Removes irrelevant columns (like Image IDs) to focus on diagnostic features.
* **Normalization:** Scales data using `StandardScaler` to ensure optimal performance for distance-based and gradient-based algorithms.
* **Reshaping:** Specifically prepares data for the RNN by transforming it into the required 3D temporal format.

---

## 🤖 Model Benchmarking

The project evaluates five distinct modeling strategies. Below is a summary of the performance metrics achieved on the test set:

| Model | Accuracy | Precision | Sensitivity | F1 Score | AUROC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 97.43% | 98.95% | 95.17% | 0.970 | 0.993 |
| **KNN** | 98.14% | 99.58% | 96.18% | 0.978 | 0.994 |
| **Decision Tree** | 98.23% | 98.57% | 97.38% | 0.980 | 0.981 |
| **Random Forest** | **98.67%** | **99.59%** | **97.38%** | **0.985** | **0.996** |
| **RNN (Deep Learning)** | 98.14% | 99.38% | 96.38% | 0.978 | **0.997** |

---

## 🛠️ Tech Stack

* **Language:** Python
* **ML Frameworks:** Scikit-Learn (Logistic Regression, KNN, DT, RF)
* **Deep Learning:** TensorFlow / Keras (SimpleRNN)
* **Visualization:** Matplotlib, Seaborn
* **Data Handling:** Pandas, Numpy

## 🚀 How to Run

1. **Dataset:** Place the `Brain Tumor.csv` file in the project root.
2. **Environment:** Ensure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn imbalanced-learn

```

3. **Execution:** Run the `SBC_TUBES_traditional_ml.ipynb` notebook. The script will automatically generate:
* Confusion Matrices for each model.
* ROC-AUC curves for performance visualization.
* Comparative performance metrics printed in the console.

---

## 📝 Findings

While the **RNN** provided the highest **AUROC** (indicating excellent separability), the **Random Forest** model achieved the highest overall **Accuracy** and **F1 Score**, making it a robust and interpretable choice for this specific clinical dataset.
