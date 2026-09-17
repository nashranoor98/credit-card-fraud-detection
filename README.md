# credit-card-fraud-detection

**Case Study 2: Credit Card Fraud Detection** — Apply **XGBoost** on the heavily imbalanced **IEEE-CIS Fraud Detection** dataset. Use **SMOTE** for oversampling, tune decision thresholds, and interpret results with feature importance scores.

# Credit Card Fraud Detection using XGBoost

An end-to-end machine learning pipeline for detecting fraudulent online transactions using the **IEEE-CIS Fraud Detection** dataset. The project combines transaction-level and identity information, handles severe class imbalance with SMOTE, trains an XGBoost classifier, tunes the probability threshold, and analyzes feature importance.

## Project Overview

Online fraud detection is a highly imbalanced binary-classification problem where fraudulent transactions represent only a small portion of all transactions. Because of this imbalance, accuracy alone is not an appropriate primary evaluation measure.

This project focuses on identifying fraudulent transactions while evaluating the trade-off between precision and recall at different decision thresholds.

### Key Highlights

- **Dataset:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Training data:** `train_transaction.csv` + `train_identity.csv`
- **Target:** `isFraud` (`0` = legitimate transaction, `1` = fraudulent transaction)
- **Dataset size:** `train_transaction.csv` contains **590,540 transactions and 394 columns**; `train_identity.csv` contains additional identity information linked through `TransactionID`.
- **Data integration:** Transaction and identity records are merged using `TransactionID`.
- **Imbalance handling:** **SMOTE** is applied only to the training data to avoid data leakage into validation data.
- **Model:** **XGBoost Classifier**
- **Threshold tuning:** Multiple probability thresholds are evaluated using precision, recall, and F1-score.
- **Evaluation:** ROC-AUC, precision, recall, F1-score, classification report, and confusion matrix.
- **Interpretability:** XGBoost feature importance scores are used to identify influential numeric features.

---

## Dataset

The project uses the official **IEEE-CIS Fraud Detection** competition dataset provided through Kaggle:

**Kaggle Dataset:** https://www.kaggle.com/competitions/ieee-fraud-detection/data

The competition data is divided into transaction and identity tables. The two training tables are joined using `TransactionID`. Kaggle notes that identity information is available only for a subset of transactions. citeturn0search0

### Files used in this project

| File | Purpose |
|---|---|
| `train_transaction.csv` | Main transaction-level training data containing the `isFraud` target |
| `train_identity.csv` | Additional identity/device information for a subset of transactions |

The Kaggle competition also provides test files and a sample submission file, but **this academic case study uses only the two training files**. citeturn0search0

### Target variable

The target column is:

```text
isFraud
```

where:

- `0` = legitimate transaction
- `1` = fraudulent transaction

The training data contains **590,540 transactions**, with fraud representing only a small minority of observations. This makes class imbalance a central part of the modeling problem.

---

## Methodology

```text
IEEE-CIS Training Data
        ↓
Load Transaction + Identity Data
        ↓
Merge using TransactionID
        ↓
Select Numeric Features
        ↓
Handle Missing Values
        ↓
Stratified Train / Validation Split
        ↓
SMOTE on Training Data Only
        ↓
XGBoost Classifier
        ↓
Fraud Probability Prediction
        ↓
Threshold Tuning
        ↓
ROC-AUC + Precision / Recall / F1
        ↓
Confusion Matrix + Feature Importance
```

### Step-by-step

1. Load `train_transaction.csv` and `train_identity.csv`.
2. Merge the two tables using `TransactionID`.
3. Prepare numeric predictors and handle missing values.
4. Create a stratified training/validation split.
5. Apply **SMOTE only to the training portion**.
6. Train the XGBoost classifier.
7. Generate fraud probabilities on the untouched validation data.
8. Evaluate ROC-AUC and classification metrics.
9. Test multiple probability thresholds to study the precision/recall/F1 trade-off.
10. Visualize the confusion matrix and the most important model features.

---

## Why SMOTE?

Fraudulent transactions are much less common than legitimate transactions. A model trained directly on the original distribution can become biased toward the majority class.

**SMOTE (Synthetic Minority Over-sampling Technique)** creates synthetic minority-class training examples to provide the model with more fraud examples during training.

Importantly, SMOTE is applied **only after the train/validation split**. The validation data remains in its original distribution so that evaluation is not contaminated by synthetic samples.

---

## Why XGBoost?

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees sequentially. It can model nonlinear relationships and interactions between tabular features, making it suitable for structured transaction data.

The project uses XGBoost as the main classifier and extracts its built-in feature importance scores for interpretation.

---

## Threshold Tuning

A default classification threshold of `0.50` is not necessarily appropriate for fraud detection.

The notebook evaluates several probability thresholds and compares:

- **Precision** — among transactions predicted as fraud, how many are actually fraudulent.
- **Recall** — among actual fraudulent transactions, how many are detected.
- **F1-score** — harmonic mean of precision and recall.

The threshold selected in the notebook is an **experimental operating point for this case study**, not a production fraud-policy decision.

---

## Evaluation Metrics

The model is evaluated using:

- **ROC-AUC** — measures the model's ability to distinguish fraudulent and legitimate transactions across thresholds.
- **Precision** — important because excessive false fraud alerts can affect legitimate customers.
- **Recall** — important because missed fraudulent transactions can result in financial loss.
- **F1-score** — balances precision and recall.
- **Confusion Matrix** — shows true negatives, false positives, false negatives, and true positives.

Run `CaseStudy2.ipynb` to generate the actual results for the dataset.

---

## Project Structure

```text
├── CaseStudy2.ipynb         # Complete end-to-end Jupyter Notebook pipeline
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes raw datasets and temporary files
└── data/
    └── README.md            # Dataset download and placement instructions
```

---

## Dataset Setup

The raw IEEE-CIS files are intentionally **not committed to this repository** because of their large size and Kaggle competition terms.

Download the dataset from:

**https://www.kaggle.com/competitions/ieee-fraud-detection/data**

Then place these two files inside the `data/` directory:

```text
data/
├── train_transaction.csv
└── train_identity.csv
```

Do **not** add the test files or `sample_submission.csv` unless they are specifically required for a separate Kaggle submission workflow. The notebook for this case study uses the training data only.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Download the two required training files from Kaggle and place them in `data/` as described above.

### 3. Run the notebook

```bash
jupyter notebook CaseStudy2.ipynb
```

---

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- XGBoost
- matplotlib
- seaborn
- Jupyter Notebook

---

## Note

This is an **educational machine-learning case study**. The threshold and model behavior should be interpreted as experimental results for the assignment rather than as a production fraud-detection or financial decision system.
