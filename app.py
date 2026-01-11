# -----------------------------
# Wine Dataset: PCA + Classifiers
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
wine = load_wine()
X = wine.data
y = wine.target

# Convert to DataFrame (optional, for exploration)
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("\nFirst 5 rows of dataset:\n", df.head())
print("\nClass distribution:\n", df['target'].value_counts())

# -----------------------------
# 2. Preprocessing
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# -----------------------------
# 3. PCA Analysis
# -----------------------------
# PCA for 95% variance
pca_95 = PCA(n_components=0.95)
pca_95.fit(X_train)

print("\nNumber of components for 95% variance:", pca_95.n_components_)

# Transform train and test using PCA 95%
X_train_pca = pca_95.transform(X_train)
X_test_pca = pca_95.transform(X_test)

# Display explained variance for each component
print("\nExplained variance ratio (95% PCA):")
for i, var in enumerate(pca_95.explained_variance_ratio_):
    print(f"Component {i+1}: {var:.4f}")

print("\nShape after PCA - X_train:", X_train_pca.shape)
print("Shape after PCA - X_test:", X_test_pca.shape)

# -----------------------------
# 4. Model Training & Evaluation
# -----------------------------
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_pca, y_train)
dt_pred = dt.predict(X_test_pca)
dt_acc = accuracy_score(y_test, dt_pred)
dt_cm = confusion_matrix(y_test, dt_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
rf_pred = rf.predict(X_test_pca)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Support Vector Machine
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_pca, y_train)
svm_pred = svm.predict(X_test_pca)
svm_acc = accuracy_score(y_test, svm_pred)
svm_cm = confusion_matrix(y_test, svm_pred)

# -----------------------------
# 5. Results
# -----------------------------
print("\n--- Test Accuracies ---")
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)
print("SVM Accuracy:", svm_acc)

print("\n--- Confusion Matrices ---")
print("Decision Tree:\n", dt_cm)
print("Random Forest:\n", rf_cm)
print("SVM:\n", svm_cm)

# Identify best model
accuracy_dict = {'Decision Tree': dt_acc, 'Random Forest': rf_acc, 'SVM': svm_acc}
best_model_name = max(accuracy_dict, key=accuracy_dict.get)
print("\nBest-performing classifier:", best_model_name)
print("Justification: Random Forest usually performs best due to ensemble learning reducing overfitting.")

# -----------------------------
# 6. Optional: Confusion Matrix Visualization
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18,5))

sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Decision Tree")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_title("SVM")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()
