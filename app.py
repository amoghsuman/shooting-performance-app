import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# Helper to load compressed files
def load_gz_pickle(path):
    with gzip.open(path, 'rb') as f:
        return joblib.load(f)

# Load all components
model = load_gz_pickle("shooting_performance_model.pkl.gz")
scaler = load_gz_pickle("scaler.pkl.gz")
label_encoders = load_gz_pickle("label_encoders.pkl.gz")
feature_names = load_gz_pickle("feature_names.pkl.gz")

# UI starts
st.title("🎯 AI Shooting Accuracy Predictor")

# Input widgets
experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Expert"])
handedness = st.radio("Handedness", ["Left", "Right"])
fatigue = st.slider("Fatigue Level", 0, 100, 50)
score = st.slider("Score", 60, 100, 85)
grouping = st.slider("Grouping Size (cm)", 5.0, 20.0, 10.0)
reaction = st.slider("Reaction Time (sec)", 0.5, 3.0, 1.5)
pressure = st.slider("Pressure Level", 0, 100, 50)
wind = st.slider("Wind Speed (km/h)", 0, 20, 5)
temp = st.slider("Temperature (°C)", 10, 35, 25)
humidity = st.slider("Humidity (%)", 20, 80, 50)
lighting = st.selectbox("Lighting Conditions", ["Poor", "Average", "Good"])
altitude = st.slider("Altitude (m)", 100, 2000, 500)
training = st.selectbox("Training Type", ["Static Shooting", "Moving Target", "Competitive Match"])
shots = st.slider("Number of Shots", 5, 50, 20)
age = st.slider("Age", 18, 60, 30)

# Encode categories
experience_enc = label_encoders["Experience_Level"].transform([experience])[0]
handedness_enc = label_encoders["Handedness"].transform([handedness])[0]
lighting_enc = label_encoders["Lighting_Conditions"].transform([lighting])[0]
training_enc = label_encoders["Training_Type"].transform([training])[0]

# Construct input row
row_dict = {
    "Experience_Level": experience_enc,
    "Handedness": handedness_enc,
    "Fatigue_Level": fatigue,
    "Score": score,
    "Grouping_Size (cm)": grouping,
    "Reaction_Time (sec)": reaction,
    "Pressure_Level (0-100)": pressure,
    "Wind_Speed (km/h)": wind,
    "Temperature (°C)": temp,
    "Humidity (%)": humidity,
    "Lighting_Conditions": lighting_enc,
    "Altitude (m)": altitude,
    "Training_Type": training_enc,
    "Number_of_Shots": shots,
    "Age": age
}

# Ensure correct feature order
X_input = pd.DataFrame([row_dict])[feature_names]

# Apply scaler only to numerical columns
numerical_cols = scaler.feature_names_in_.tolist()
X_input[numerical_cols] = scaler.transform(X_input[numerical_cols])

# Predict
if st.button("Predict Shot Accuracy"):
    prediction = model.predict(X_input)[0]
    st.success(f"🎯 Predicted Shot Accuracy: {prediction:.2f}%")
