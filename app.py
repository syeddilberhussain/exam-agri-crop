
# Import necessary libraries
import pandas as pd           # For data manipulation
import numpy as np            # For numerical operations
from sklearn.model_selection import train_test_split   # To split dataset into train and test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # To scale features & encode labels
from sklearn.neighbors import KNeighborsClassifier    # KNN classifier
import joblib                  # To save/load trained model, scaler, and label encoder
import streamlit as st
import os

# Load dataset
def load_data(path):
    """
    Loads the crop recommendation dataset from CSV file.
    Input: path (str) - path to CSV file
    Output: pandas DataFrame
    """
    df = pd.read_csv(path)
    return df

# Preprocess dataset
def preprocess_data(df):
    """
    Splits data into features (X) and target (y)
    Encodes target labels
    Scales features for KNN
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, label encoder
    """
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # Encode target labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Scale features for better KNN performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

# Train KNN model
def train_knn(X_train, y_train, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors classifier
    Input: X_train (scaled features), y_train (encoded labels), n_neighbors
    Output: trained KNN model
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Save trained model, scaler, and label encoder
def save_objects(model, scaler, label_encoder, model_path="crop_knn_model.pkl",
                 scaler_path="scaler.pkl", le_path="label_encoder.pkl"):
    """
    Saves the model, scaler, and label encoder using joblib
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, le_path)

# Load trained model, scaler, and label encoder
def load_objects(model_path="crop_knn_model.pkl", scaler_path="scaler.pkl", le_path="label_encoder.pkl"):
    """
    Loads the trained model, scaler, and label encoder
    Returns: model, scaler, label_encoder
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    return model, scaler, le

# Predict crop based on user input
def predict_crop(model, scaler, label_encoder, input_features):
    """
    Predicts the best crop for given input features
    Input:
        model - trained KNN model
        scaler - StandardScaler used for training
        label_encoder - LabelEncoder used for target
        input_features - list or array of 7 values: [N, P, K, temperature, humidity, ph, rainfall]
    Output:
        predicted crop name (str)
    """
    input_array = np.array([input_features])
    input_scaled = scaler.transform(input_array)
    pred_encoded = model.predict(input_scaled)
    pred_crop = label_encoder.inverse_transform(pred_encoded)[0]
    return pred_crop

# Download the dataset
# This URL points to a raw CSV file from a GitHub repository.
# If this URL becomes invalid, please find an alternative source for 'Crop_recommendation.csv'
# and update the URL accordingly.
print(f"Current working directory: {os.getcwd()}")
!wget https://raw.githubusercontent.com/manishabajaj/Crop-Recommendation-System/main/Crop_recommendation.csv
!ls -l Crop_recommendation.csv

# Load dataset
df = load_data("/content/Crop_recommendation.csv")

# Preprocess and train model
X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df)
knn = train_knn(X_train, y_train)

# Save objects
save_objects(knn, scaler, le)

# Load objects (for prediction in Streamlit)
model, scaler, le = load_objects()

# Streamlit UI
st.title("🌱 Crop Recommendation System")
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temp = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
hum = st.sidebar.slider("Humidity (%)", 0, 100, 80)
ph_val = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rain = st.sidebar.slider("Rainfall (mm)", 0, 300, 200)

user_input = [N, P, K, temp, hum, ph_val, rain]
predicted_crop = predict_crop(model, scaler, le, user_input)

st.subheader("🌱 Recommended Crop")
st.success(predicted_crop.upper())
