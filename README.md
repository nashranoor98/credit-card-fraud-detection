# credit-card-fraud-detection

Case Study 2: Credit Card Fraud Detection — Apply XGBoost on a heavily imbalanced transactions dataset. Use SMOTE for oversampling, tune decision thresholds, and interpret results with feature importance scores.

# Credit Card Fraud Detection using XGBoost and SMOTE

A machine learning pipeline designed to detect fraudulent online transactions using the **IEEE-CIS Fraud Detection** dataset.

## Project Overview

The project handles class imbalance using **SMOTE**, trains an **XGBoost Classifier**, and tunes the decision threshold to study fraud detection performance.

### Key Highlights

* **Dataset:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
* **Data:** `train_transaction.csv` and `train_identity.csv`
* **Handling Imbalance:** SMOTE applied on the training data.
* **Modeling:** XGBoost Classifier.
* **Threshold Tuning:** Different probability thresholds are compared.
* **Evaluation:** ROC-AUC, precision, recall, F1-score and confusion matrix.
* **Interpretation:** Feature importance scores are used to understand important features.

---

## Project Structure

```text
├── CaseStudy2.ipynb
├── README.md
└── .gitignore
```

---

## Getting Started

Download the IEEE-CIS dataset from Kaggle and run `CaseStudy2.ipynb` in Jupyter Notebook or Google Colab.

> This project is an academic machine learning case study and is intended for educational purposes.
