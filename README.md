# 👗 Fashion Product Sales Forecasting & Purchase Prediction

A end-to-end Machine Learning pipeline and interactive web application designed to predict whether a customer will purchase a garment based on product attributes, trial dynamics, and pricing. Built with **Python**, **XGBoost**, **scikit-learn**, **imbalanced-learn**, and **Streamlit**.

---

## 📌 Project Overview

In fashion e-commerce and retail, predicting customer purchase intent based on product attributes and fitting trial interactions is critical for inventory management, dynamic pricing, and personalized recommendations.

This project implements a production-grade machine learning system that:
1. Preprocesses and cleans raw retail fashion interaction data.
2. Standardizes categorical features and handles class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
3. Trains an optimized **XGBoost** classifier.
4. Performs **decision threshold tuning** to maximize the F1-Score and Recall.
5. Evaluates model performance using ROC curves, Precision-Recall curves, confusion matrices, and feature dominance checks.
6. Serves the trained model via a **Streamlit Web Application** for real-time predictions.

---

## 🛠️ Key Features

- **Modular Architecture**: Clean separation of concerns across dedicated Python modules (`data_loader`, `preprocessing`, `smote_handler`, `model`, `train`, `evaluate`, `utils`).
- **Class Imbalance Management**: Applies SMOTE strictly on training partitions to prevent data leakage while resolving target skewness.
- **Decision Threshold Optimization**: Tunes the probability classification threshold to improve positive class detection (Recall & F1-Score).
- **Comprehensive Visualizations**: Automatically generates and saves high-resolution ROC curves, Precision-Recall curves, Confusion Matrices, and Feature Importance charts.
- **Feature Dominance Guardrails**: Includes automated checks to detect over-dominant features (>40% importance threshold).
- **Interactive Web Dashboard**: Streamlit interface with interactive sliders, select boxes, confidence indicators, and detailed metrics.

---

## 📁 Repository Structure

```
d:/garment_purchase_prediction final/
├── app.py                         # Streamlit web application interface
├── main.py                        # Orchestrator script for the complete ML pipeline
├── data_loader.py                 # Data loading and initial inspection module
├── preprocessing.py              # Data cleaning, normalization, and one-hot encoding
├── smote_handler.py               # Imbalanced data handling using SMOTE
├── model.py                       # XGBoost model definition and hyperparameter settings
├── train.py                       # Model training and prediction generator
├── evaluate.py                    # Evaluation metrics, threshold tuning, and plotting
├── utils.py                       # Helper utilities (stratified split, model saving/loading)
├── fashion_purchase_dataset.csv   # Dataset containing product attributes and purchase target
├── requirements.txt               # Dependencies required to run the project
└── outputs/                       # Directory storing generated plots and trained model file
    ├── purchase_model.joblib      # Serialized trained XGBoost model
    ├── confusion_matrix.png       # Confusion matrix visualization
    ├── roc_curve.png              # Receiver Operating Characteristic (ROC) curve
    ├── precision_recall_curve.png # Precision-Recall curve
    └── feature_importance.png     # Top feature importances bar chart
```

---

## 📊 Dataset & Features

The model utilizes product characteristics and trial interactions to predict purchase decisions (`Purchased`: 0 or 1).

| Feature Name | Type | Description / Range |
| :--- | :--- | :--- |
| **Size** | Categorical | Garment size (`S`, `M`, `L`, `XL`) |
| **Sleeve_Type** | Categorical | Sleeve design (`Full`, `Half`, `Sleeveless`) |
| **Color** | Categorical | Garment color (`Black`, `Blue`, `White`, `Red`, `Green`, `Yellow`) |
| **Collar_Type** | Categorical | Collar style (`Round`, `Polo`, `V-Neck`, `Mandarin`) |
| **Fit** | Categorical | Garment fit (`Slim`, `Regular`, `Loose`) |
| **Pattern** | Categorical | Fabric pattern (`Solid`, `Printed`, `Striped`, `Checked`) |
| **Price** | Numerical | Garment price in ₹ (Range: 200 - 3,000) |
| **Trial_Count** | Numerical | Number of fitting trial attempts by customer (Range: 0 - 10) |
| **Purchased** | Target (Binary) | `1` if purchased, `0` if not purchased |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Required packages include:* `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`, and `streamlit`.

---

## 🚀 Usage Instructions

### 1. Run the Machine Learning Pipeline
To execute data loading, preprocessing, model training, threshold tuning, and evaluation:

```bash
python main.py
```

Upon execution, `main.py` will:
- Process `fashion_purchase_dataset.csv`.
- Train the XGBoost model.
- Save the trained model to `outputs/purchase_model.joblib`.
- Output metric reports and save plots to `outputs/`.

### 2. Launch the Streamlit Web Application
To start the interactive web user interface:

```bash
streamlit run app.py
```

Then open your browser and navigate to:
```
http://localhost:8501
```

---

## 📈 Model Performance & Evaluation

The pipeline evaluates model predictions across standard probability thresholds as well as optimized decision boundaries:

- **Default Threshold (0.50)**:
  - F1-Score: `0.4384`
- **Optimal Threshold (0.28)**:
  - F1-Score: `0.5540`
  - Recall: `0.9170`
  - ROC AUC: `0.6294`

### Output Artifacts
Visual outputs saved in `outputs/`:
- `confusion_matrix.png`: Breakdown of true vs. false predictions.
- `roc_curve.png`: Trade-off between True Positive Rate and False Positive Rate.
- `precision_recall_curve.png`: Performance across imbalance threshold levels.
- `feature_importance.png`: Top features influencing customer purchase decisions (e.g., `Size_m`, `Trial_Count`, `Fit_slim`).

---

## 💻 Tech Stack

- **Language**: Python
- **Machine Learning**: XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Data Processing & Analytics**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Model Storage**: Joblib
- **Web Framework**: Streamlit
