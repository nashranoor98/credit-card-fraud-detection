# credit-card-fraud-detection

Case study 2 : Credit Card Fraud Detection — Apply XGBoost on a heavily imbalanced transactions dataset (e.g. Kaggle IEEE-CIS). Use SMOTE for oversampling, tune decision thresholds, and interpret results with feature importance scores.

# Credit Card Fraud Detection using XGBoost and SMOTE

A complete machine learning pipeline designed to detect fraudulent transactions using the **IEEE-CIS Fraud Detection** dataset. This project uses **SMOTE**, an **XGBoost Classifier**, and decision threshold tuning.

## Project Overview

In fraud detection, transaction datasets are highly imbalanced. This project demonstrates how to handle the imbalance and train a model to identify fraudulent transactions.

### Key Highlights

* **Dataset:** Official [Kaggle IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) containing **590,540 transactions** and **394 transaction columns**.
* **Exploratory Data Analysis (EDA):** Examined the class distribution between legitimate and fraudulent transactions.
* **Handling Imbalance:** Applied **SMOTE** on the training split.
* **Modeling:** Trained an **XGBClassifier** to capture non-linear patterns.
* **Threshold Tuning:** Compared classification thresholds to balance fraud detection and false alarms.

## Performance Metrics

Evaluation on the held-out validation set from the working sample:

- **ROC-AUC Score:** `0.9117`
- **Selected Threshold:** `0.40`
- **Precision:** `0.7639`
- **Recall:** `0.7697`
- **F1-Score:** `0.7668`
- **Accuracy:** `0.8631`
- **Confusion Matrix:** `[[9017, 983], [952, 3181]]`

(TN 9017 | FP 983 | FN 952 | TP 3181)

## Project Structure

```text
├── CaseStudy2.ipynb
└── README.md
```

## Getting Started & Replication

Download the IEEE-CIS Fraud Detection training files from Kaggle and run `CaseStudy2.ipynb` in Google Colab or Jupyter Notebook.

> The notebook uses a balanced working sample of the IEEE-CIS training data so that SMOTE remains practical on memory-limited machines.

> This project is an academic machine learning case study intended for educational purposes.
