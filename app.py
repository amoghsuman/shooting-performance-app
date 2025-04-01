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

st.markdown("""
This app predicts a shooter's performance accuracy based on various physical, environmental, and behavioral inputs.

The prediction is made using a trained AI model that analyzes **15+ features** affecting a shooter's performance during training or competitions.
""")

# Input widgets
experience = st.selectbox(
    "Experience Level",
    ["Beginner", "Intermediate", "Expert"],
    help="Shooter's skill classification based on training and experience level."
)

handedness = st.radio(
    "Handedness",
    ["Left", "Right"],
    help="Which hand the shooter primarily uses to shoot."
)

fatigue = st.slider(
    "Fatigue Level",
    0, 100, 50,
    help="Shooter's physical tiredness level. Lower is better for performance."
)

score = st.slider(
    "Score",
    60, 100, 85,
    help="Total session score out of 100. Reflects prior shooting performance."
)

grouping = st.slider(
    "Grouping Size (cm)",
    5.0, 20.0, 10.0,
    help="Spread of bullet holes on the target. Smaller groupings indicate better aim."
)

reaction = st.slider(
    "Reaction Time (sec)",
    0.5, 3.0, 1.5,
    help="Time taken to react and shoot. Faster reaction improves performance."
)

pressure = st.slider(
    "Pressure Level",
    0, 100, 50,
    help="Shooter's perceived mental pressure during the session."
)

wind = st.slider(
    "Wind Speed (km/h)",
    0, 20, 5,
    help="Wind conditions during the session. Higher wind can reduce accuracy."
)

temp = st.slider(
    "Temperature (°C)",
    10, 35, 25,
    help="Ambient temperature during shooting. Extreme heat or cold may impact comfort."
)

humidity = st.slider(
    "Humidity (%)",
    20, 80, 50,
    help="Humidity level during the session. Can influence bullet trajectory slightly."
)

lighting = st.selectbox(
    "Lighting Conditions",
    ["Poor", "Average", "Good"],
    help="Quality of lighting at the shooting location."
)

altitude = st.slider(
    "Altitude (m)",
    100, 2000, 500,
    help="Elevation of shooting range above sea level. Higher altitudes can affect trajectory."
)

training = st.selectbox(
    "Training Type",
    ["Static Shooting", "Moving Target", "Competitive Match"],
    help="Type of session. Static targets are easiest, matches are hardest."
)

shots = st.slider(
    "Number of Shots",
    5, 50, 20,
    help="Total number of shots fired in the session. More shots can stabilize performance."
)

age = st.slider(
    "Age",
    18, 60, 30,
    help="Age of the shooter. Age may correlate with experience and physical sharpness."
)

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
