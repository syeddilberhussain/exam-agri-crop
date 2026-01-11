# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.title("🌱 Crop Recommendation Dataset Visualizations")

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------
st.header("Dataset Preview")
data = pd.read_csv("Crop_recommendation.csv")
st.dataframe(data.head())

# ---------------------------
# 2️⃣ Feature Distributions
# ---------------------------
st.header("Feature Distributions (Histograms)")
if st.checkbox("Show Histograms"):
    fig, ax = plt.subplots(figsize=(12,8))
    data.hist(bins=20, figsize=(12,8))
    plt.suptitle("Feature Distributions")
    st.pyplot(fig)

# ---------------------------
# 3️⃣ Correlation Heatmap
# ---------------------------
st.header("Correlation Heatmap")
if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(data.drop('label', axis=1).corr(), annot=True, cmap="YlGnBu", ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

# ---------------------------
# 4️⃣ Example Model Accuracy Comparison
# ---------------------------
st.header("Model Accuracy Comparison")
models = ["Decision Tree", "KNN", "Random Forest"]
accuracies = [0.92, 0.88, 0.95]  # Example accuracies
if st.checkbox("Show Model Accuracy Chart"):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=models, y=accuracies, hue=models, palette="viridis", dodge=False, ax=ax)
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    st.pyplot(fig)

# ---------------------------
# 5️⃣ Random Forest Feature Importance
# ---------------------------
st.header("Random Forest Feature Importance")
if st.checkbox("Show Feature Importance"):
    X = data.drop('label', axis=1)
    y = data['label']

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    importances = rf_model.feature_importances_
    feature_names = X.columns

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=feature_names, y=importances, hue=feature_names, palette="magma", dodge=False, ax=ax)
    plt.title("Feature Importance (Random Forest)")
    plt.ylabel("Importance")
    st.pyplot(fig)


