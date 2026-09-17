# Credit Card Fraud Detection

Case Study 2: Credit Card Fraud Detection using XGBoost on a heavily imbalanced transaction dataset. SMOTE is used for oversampling, decision thresholds are tuned, and feature importance is examined.

## Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

**Dataset:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The original training dataset contains **590,540 transactions** and **394 columns**.

## Steps Performed

1. **Loading and Studying Dataset**
2. **Splitting Dataset** into training and validation sets
3. **Applying SMOTE** to balance the training classes
4. **Training XGBoost** for fraud classification
5. **Threshold Tuning** to select a classification threshold
6. **Feature Importance** to identify important predictors
7. **Model Evaluation** using ROC-AUC, precision, recall, F1-score and confusion matrix

## Performance Metrics

Evaluation on the held-out validation set from the working sample yielded:

- **ROC-AUC Score:** `0.9117`
- **Selected Threshold:** `0.40`
- **Precision:** `0.7639`
- **Recall:** `0.7697`
- **F1-Score:** `0.7668`
- **Accuracy:** `0.8631`
- **Confusion Matrix:**

  `[[9017, 983],`
  ` [952, 3181]]`

**True Negatives: 9017 | False Positives: 983 | False Negatives: 952 | True Positives: 3181**

## Note

The notebook uses a balanced working sample of the IEEE-CIS training data so that SMOTE remains practical on memory-limited machines. The reported metrics are from this working sample, not from the complete 590,540-row dataset.

## Project Structure

```text
credit-card-fraud-detection/
├── CaseStudy2.ipynb
└── README.md
```

## Requirements

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib

This project is an academic machine learning case study intended for educational purposes.