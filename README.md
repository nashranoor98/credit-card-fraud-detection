# credit-card-fraud-detection
Case study 2 : Credit Card Fraud Detection — Apply XGBoost on a heavily imbalanced transactions dataset (e.g. Kaggle IEEE-CIS). Use SMOTE for oversampling, tune decision thresholds, and interpret results with feature importance scores.


# Credit Card Fraud Detection using XGBoost and SMOTE

A complete end-to-end Machine Learning pipeline designed to detect fraudulent transactions using the **IEEE-CIS Fraud Detection** dataset. This project implements **SMOTE (Synthetic Minority Over-sampling Technique)**, trains an **XGBoost Classifier**, and uses **custom decision threshold tuning**.

## Project Overview

In real-world fraud detection, transaction datasets are highly imbalanced—normal transactions vastly outnumber fraudulent ones. This project demonstrates how to handle the imbalance and train a model to identify fraudulent transactions.

### Key Highlights

* **Dataset:** Official [Kaggle IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) containing **590,540 transactions** and **394 transaction columns**, with additional identity information.
* **Exploratory Data Analysis (EDA):** Examined the class distribution between legitimate and fraudulent transactions.
* **Handling Imbalance:** Applied **SMOTE** strictly on the training split to synthesize minority fraud samples.
* **Modeling:** Trained an **XGBClassifier** capable of capturing complex non-linear patterns.
* **Threshold Tuning:** Compared different classification thresholds to study the balance between catching fraudulent transactions and minimizing false alarms.

---

## Performance Metrics

The model is evaluated on the held-out validation set using:

* **ROC-AUC**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

The actual values are generated when `CaseStudy2.ipynb` is run on the IEEE-CIS dataset.

---

## Project Structure

```text
├── CaseStudy2.ipynb         # Complete end-to-end Jupyter Notebook pipeline
└── README.md                # Project documentation
```

---

## Getting Started & Replication

Download the IEEE-CIS Fraud Detection training dataset from Kaggle and run this notebook in Google Colab or Jupyter Notebook.
