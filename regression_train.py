import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import sys
import numpy as np

# Load data
data_path = 'data/features/features.csv'
df = pd.read_csv(data_path)

# Check for null values
if df.isnull().any().any():
    print("Null values found in the dataset.")
    sys.exit(1)

# Use only selected features
selected_features = ['spectral_centroid', 'power_band_0_50Hz', 'kurtosis', 'power_band_50_100Hz']
X = df[selected_features]
y = df.iloc[:, -1]
X = X.iloc[:, :-1]
y = df.iloc[:, -1]

# Check class balance
class_counts = Counter(y)
print("Class distribution:", class_counts)
if min(class_counts.values()) / max(class_counts.values()) < 0.5:
    print("Warning: Classes are imbalanced.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Optionally, label the matrix for clarity if you have binary classification
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis for logistic regression (using absolute value of coefficients)
feature_names = X.columns
coefs = model.coef_[0] if model.coef_.shape[0] == 1 else np.mean(np.abs(model.coef_), axis=0)
feature_importance = pd.Series(np.abs(coefs), index=feature_names).sort_values(ascending=False)

print("\nFeature importance (absolute coefficient values):")
print(feature_importance)