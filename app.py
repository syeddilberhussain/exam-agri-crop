# =========================================================
# Crop Recommendation & Wine Dataset Classifiers
# =========================================================
# RUN THIS APP USING TERMINAL OR COLAB:
# -----------------------------------------------
# In terminal: streamlit run app.py
# In Colab: just run the cell (ngrok code removed to avoid auth error)
# =========================================================

import pandas as pd
import numpy as np
import os
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

# =========================================================
# Crop Recommendation Model
# =========================================================
@st.cache_resource
def train_crop_model():
    # Dataset download (agar file na ho)
    if not os.path.exists("Crop_recommendation.csv"):
        os.system(
            "wget https://raw.githubusercontent.com/manishabajaj/Crop-Recommendation-System/main/Crop_recommendation.csv"
        )

    data = pd.read_csv("Crop_recommendation.csv")

    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, label_encoder, accuracy


crop_model, crop_scaler, crop_le, crop_acc = train_crop_model()


# =========================================================
# Wine Dataset Models (DT, RF, SVM)
# =========================================================
@st.cache_resource
def train_wine_models():
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC()

    dt.fit(X_train_pca, y_train)
    rf.fit(X_train_pca, y_train)
    svm.fit(X_train_pca, y_train)

    acc_dt = accuracy_score(y_test, dt.predict(X_test_pca))
    acc_rf = accuracy_score(y_test, rf.predict(X_test_pca))
    acc_svm = accuracy_score(y_test, svm.predict(X_test_pca))

    return acc_dt, acc_rf, acc_svm


acc_dt, acc_rf, acc_svm = train_wine_models()


# =========================================================
# Streamlit User Interface
# =========================================================
st.title("🌱 Crop Recommendation & 🍷 Wine Classification System")

st.sidebar.header("Enter Crop Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temp = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
hum = st.sidebar.slider("Humidity (%)", 0, 100, 80)
ph_val = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rain = st.sidebar.slider("Rainfall (mm)", 0, 300, 200)

user_input = crop_scaler.transform([[N, P, K, temp, hum, ph_val, rain]])
prediction = crop_model.predict(user_input)
crop_name = crop_le.inverse_transform(prediction)[0]

st.success(f"✅ Recommended Crop: **{crop_name.upper()}**")
st.write(f"🎯 Crop Model Accuracy: **{crop_acc:.2f}**")

st.subheader("🍷 Wine Dataset Model Accuracies")
st.write(f"Decision Tree Accuracy: **{acc_dt:.2f}**")
st.write(f"Random Forest Accuracy: **{acc_rf:.2f}**")
st.write(f"SVM Accuracy: **{acc_svm:.2f}**")

st.write("⚠️ Note: Ngrok code removed to avoid authentication error. Use local terminal to run app publicly.")
