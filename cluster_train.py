from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



# Load data
data_path = 'data/features/features.csv'
df = pd.read_csv(data_path)


# Use only selected features
# selected_features = ['spectral_centroid', 'power_band_0_50Hz', 'kurtosis', 'power_band_50_100Hz']
# X = df[selected_features]
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]

# --- 3. Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Dimensionality Reduction (PCA for 2D plot) ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 5. Clustering (K-Means) ---
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# --- 6. Compare cluster labels to actual y values ---

# If y is not numeric, encode it
if y.dtype == 'O':
    y_true = LabelEncoder().fit_transform(y)
else:
    y_true = y.values

# KMeans labels may be inverted, so check both possibilities
acc1 = accuracy_score(y_true, labels)
acc2 = accuracy_score(y_true, 1 - labels)
best_acc = max(acc1, acc2)
print(f"Best clustering accuracy: {best_acc:.2f}")

print("Confusion matrix:")
print(confusion_matrix(y_true, labels))
print("Confusion matrix (inverted):")
print(confusion_matrix(y_true, 1 - labels))

# Compute metrics for both label assignments (since KMeans labels can be inverted)
precision1 = precision_score(y_true, labels)
recall1 = recall_score(y_true, labels)
f1_1 = f1_score(y_true, labels)

precision2 = precision_score(y_true, 1 - labels)
recall2 = recall_score(y_true, 1 - labels)
f1_2 = f1_score(y_true, 1 - labels)

if best_acc == acc1:
    print(f"Precision: {precision1:.2f}")
    print(f"Recall: {recall1:.2f}")
    print(f"F1 Score: {f1_1:.2f}")
else:
    print(f"Precision: {precision2:.2f}")
    print(f"Recall: {recall2:.2f}")
    print(f"F1 Score: {f1_2:.2f}")

# --- 7. Visualization ---
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title("Unsupervised Classification with K-Means")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()