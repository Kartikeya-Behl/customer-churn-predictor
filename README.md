# 🛒 E-Commerce Customer Churn Prediction Pipeline

An end-to-end, production-ready machine learning pipeline built with **XGBoost** to identify high-risk customer churn and optimize retention strategies.

---

## 📌 Executive Summary

Customer acquisition costs significantly outpace retention costs. This project implements a regularized XGBoost classification pipeline to forecast customer churn on e-commerce platforms. By optimizing the classification decision boundary ($0.50 \rightarrow 0.60$), the model achieves a **33% reduction in false positives** (minimizing wasted retention discounts) while maintaining a **95% recall rate** on actual churners.

---

## 🛠️ Architecture & Technical Highlights

- **Data Preprocessing & Encoding:** Standard scaling for numerical features and One-Hot Encoding for categorical variables using `ColumnTransformer` to prevent data leakage.
- **Regularization & Overfitting Mitigation:**
  - Constrained tree depth (`max_depth=5`) and applied row/feature subsampling (`subsample=0.7`, `colsample_bytree=0.8`).
  - Implemented split penalty parameter ($\gamma = 0.5$) to prevent leaf overfitting, closing the gap between training and 5-fold cross-validation scores.
- **Threshold Tuning for Business Value:** Custom decision thresholding to optimize the Precision-Recall trade-off based on customer retention economics.
- **Pipeline Serialization:** Fully serialized model, preprocessor, and metadata artifacts via `joblib` for seamless inference.

---

## 📊 Model Performance

| Metric                                | Default Threshold (0.50) | Optimized Threshold (0.60) |
| :------------------------------------ | :----------------------: | :------------------------: |
| **Churn Precision**                   |           0.84           |          **0.88**          |
| **Churn Recall**                      |           0.99           |          **0.95**          |
| **F1-Score (Churn)**                  |           0.91           |          **0.92**          |
| **False Positives (Wasted Budget)**   |            36            |       **24 (-33%)**        |
| **False Negatives (Missed Churners)** |            2             |           **9**            |

---

## 📁 Repository Structure

```
.
├── main.ipynb # Exploratory Data Analysis, Model Training & Validation
├── main.py # Production-style batch inference pipeline
├── churn_preprocessor.joblib # Fitted Feature Transformer
├── churn_xgb_model.joblib # Trained XGBoost Model
├── model_metadata.joblib # Decision Threshold Metadata
├── E-Commerce_Churn_Data.csv # Dataset
├── requirements.txt # Environment Dependencies
└── README.md # Project Documentation
```

## Quickstart & Usage:

### 1. Installation

Clone the repository and install required dependencies:
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt

### 2. Run Batch Inference

Execute the inference script to make predictions on sample customer data using the serialized pipeline artifacts:
python main.py

### 3. Notebook Workflow

To inspect exploratory data analysis, hyperparameter tuning grids, and performance curves, open main.ipynb:
jupyter notebook main.ipynb
