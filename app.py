# Crop Recommendation + Wine Dataset Classifiers
# -----------------------------

import pandas as pd
import numpy as np
import os
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1️⃣ Crop Recommendation Dataset
# -----------------------------

# Download Crop dataset if not present
if not os.path.exists("Crop_recommendation.csv"):
    os.system("wget https://raw.githubusercontent.com/manishabajaj/Crop-Recommendation-System/main/Crop_recommendation.csv")

crop_data = pd.read_csv("/content/Crop_recommendation.csv")

# Features and target
X_crop = crop_data[['N','P','K','temperature','humidity','ph','rainfall']]
y_crop = crop_data['label']

# Encode target
le_crop = LabelEncoder()
y_encoded = le_crop.fit_transform(y_crop)

# Scale features
scaler_crop = StandardScaler()
X_scaled = scaler_crop.fit_transform(X_crop)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest for crop recommendation
rf_crop = RandomForestClassifier(n_estimators=100, random_state=42)
rf_crop.fit(X_train, y_train)
y_pred_crop = rf_crop.predict(X_test)
crop_acc = accuracy_score(y_test, y_pred_crop)

# -----------------------------
# 2️⃣ Wine Dataset (fetch_openml to avoid load_wine error)
# -----------------------------
from sklearn.datasets import fetch_openml

wine = fetch_openml(name='wine', version=1, as_frame=True)
X_wine = wine.data
y_wine = wine.target.astype(int)

# Scaling
scaler_w = StandardScaler()
X_scaled_w = scaler_w.fit_transform(X_wine)

# Train-test split
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_scaled_w, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)

# PCA (95% variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_w)
X_test_pca = pca.transform(X_test_w)

# Train classifiers
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', random_state=42)

dt.fit(X_train_pca, y_train_w)
rf.fit(X_train_pca, y_train_w)
svm.fit(X_train_pca, y_train_w)

# Predictions
dt_pred = dt.predict(X_test_pca)
rf_pred = rf.predict(X_test_pca)
svm_pred = svm.predict(X_test_pca)

# Accuracies
acc_dt = accuracy_score(y_test_w, dt_pred)
acc_rf = accuracy_score(y_test_w, rf_pred)
acc_svm = accuracy_score(y_test_w, svm_pred)

# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.title("🌱 Crop Recommendation System")

st.sidebar.header("Input Soil & Weather Parameters")
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temp = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
hum = st.sidebar.slider("Humidity (%)", 0, 100, 80)
ph_val = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rain = st.sidebar.slider("Rainfall (mm)", 0, 300, 200)

user_input = np.array([[N, P, K, temp, hum, ph_val, rain]])
user_input_scaled = scaler_crop.transform(user_input)
pred_crop_encoded = rf_crop.predict(user_input_scaled)
pred_crop = le_crop.inverse_transform(pred_crop_encoded)[0]

st.subheader("🌱 Recommended Crop")
st.success(pred_crop.upper())
st.write(f"Random Forest Accuracy on test set: {crop_acc:.2f}")

# -----------------------------
# 4️⃣ Wine Classifier Accuracies
# -----------------------------
st.subheader("🍷 Wine Dataset Classifier Accuracies (with PCA)")
st.write(f"Decision Tree Accuracy: {acc_dt:.2f}")
st.write(f"Random Forest Accuracy: {acc_rf:.2f}")
st.write(f"SVM Accuracy: {acc_svm:.2f}")

# Confusion matrices visualization
fig, axes = plt.subplots(1,3, figsize=(18,5))
sns.heatmap(confusion_matrix(y_test_w, dt_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Decision Tree")
sns.heatmap(confusion_matrix(y_test_w, rf_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Random Forest")
sns.heatmap(confusion_matrix(y_test_w, svm_pred), annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_title("SVM")
st.pyplot(fig)

