import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore

from collections import Counter
import sys
import numpy as np
# Load data
data_path = 'data/features/features.csv'
df = pd.read_csv(data_path)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]  # Last column as target variable

# Compute z-scores using only class 0 as the reference distribution
X_class0 = X[y == 0]
means = X_class0.mean()
stds = X_class0.std(ddof=0)
z_scores = np.abs((X - means) / stds)

# Identify outliers in class 1 based on class 0 distribution
outlier_mask = (z_scores > 3).any(axis=1)
outliers = df[outlier_mask & (y == 1)]
non_outliers = df[~outlier_mask & (y == 1)]

# Identify outliers in class 0 based on class 0 distribution
outlier_mask_class0 = (z_scores > 3).any(axis=1)
outliers_class0 = df[outlier_mask_class0 & (y == 0)]
non_outliers_class0 = df[~outlier_mask_class0 & (y == 0)]

print("Class 0 Outliers count:", len(outliers_class0))
print("Class 0 Non-outliers count:", len(non_outliers_class0))

# Check distribution of y labels in outliers vs non-outliers for class 1
print("Class 1 Outliers count:", len(outliers))
print("Class 1 Non-outliers count:", len(non_outliers))

# Assign predicted labels: if a class 1 sample is an outlier, it's a TP; if not, it's a FN
# For class 0: if a class 0 sample is an outlier, it's a FP; if not, it's a TN

TP = ((y == 1) & outlier_mask).sum()
FN = ((y == 1) & ~outlier_mask).sum()
FP = ((y == 0) & outlier_mask).sum()
TN = ((y == 0) & ~outlier_mask).sum()

print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

