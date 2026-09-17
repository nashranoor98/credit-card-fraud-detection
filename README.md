# Credit Card Fraud Detection

**Case Study 2:** Apply **XGBoost** on the heavily imbalanced **IEEE-CIS Fraud Detection** dataset. Use **SMOTE** for oversampling, tune decision thresholds, and interpret results with feature importance scores.

## Project Overview

This project implements an end-to-end fraud detection pipeline using the IEEE-CIS transaction and identity training data. The workflow addresses class imbalance, trains an XGBoost classifier, tunes the classification threshold, and analyzes important features.

### Key Highlights

- **Dataset:** Kaggle IEEE-CIS Fraud Detection
- **Data:** `train_transaction.csv` + `train_identity.csv`
- **Imbalance handling:** SMOTE applied only to the training data
- **Model:** XGBoost Classifier
- **Threshold tuning:** Evaluate multiple probability thresholds using fraud-focused metrics
- **Interpretability:** XGBoost feature importance scores

## Project Structure

```text
├── CaseStudy2.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## Dataset Setup

The raw IEEE-CIS files are intentionally not committed because of their size. Download the competition data from Kaggle and place `train_transaction.csv` and `train_identity.csv` inside `data/`.

## Run

```bash
pip install -r requirements.txt
jupyter notebook CaseStudy2.ipynb
```

## Methodology

1. Load transaction and identity training data.
2. Merge the datasets using `TransactionID`.
3. Prepare numeric features and handle missing values.
4. Split into stratified training and validation sets.
5. Apply SMOTE only to the training portion.
6. Train XGBoost.
7. Evaluate ROC-AUC and precision/recall/F1.
8. Tune the probability threshold on validation data.
9. Plot the confusion matrix and top feature importances.

## Note

This is an educational machine-learning case study. The selected threshold should be interpreted as an experimental model operating point rather than a production fraud-policy decision.
