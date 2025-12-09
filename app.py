import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# Load trained model
# ===============================
with open("aqi_lgbm_full_model.pkl", "rb") as f:
    model = pickle.load(f)

# AQI Category bins
bins = [0, 50, 100, 150, 200, 300, 1000]
labels = ['Good','Moderate','Unhealthy SG','Unhealthy','Very Unhealthy','Hazardous']

# ===============================
# Streamlit App Layout
# ===============================
st.set_page_config(page_title="AQI Predictor", layout="centered")
st.title("🌤️ Air Quality Index (AQI) Predictor")

st.sidebar.header("Input Sensor Values")

# User input
def user_input_features():
    TempC = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
    Humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    PM25 = st.sidebar.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=50.0)
    PM10 = st.sidebar.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=80.0)
    NO2 = st.sidebar.number_input("NO2 (ppb)", min_value=0.0, max_value=1000.0, value=40.0)
    O3 = st.sidebar.number_input("O3 (ppb)", min_value=0.0, max_value=500.0, value=30.0)
    CO = st.sidebar.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=1.0)
    SO2 = st.sidebar.number_input("SO2 (ppb)", min_value=0.0, max_value=500.0, value=5.0)
    
    data = {
        'TempC': TempC,
        'Humidity': Humidity,
        'PM2.5': PM25,
        'PM10': PM10,
        'NO2 ppb': NO2,
        'O3 ppb': O3,
        'CO ppm': CO,
        'SO2 ppb': SO2
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Predict AQI numeric value
prediction = model.predict(input_df)[0]

# Predict AQI Category
category = pd.cut([prediction], bins=bins, labels=labels)[0]

# Display Results
st.subheader("Predicted AQI Value")
st.metric(label="AQI", value=f"{prediction:.2f}")

st.subheader("Predicted AQI Category")
st.markdown(f"**{category}**")

# Optional: Color-coded message
aqi_colors = {
    'Good': 'green',
    'Moderate': 'yellow',
    'Unhealthy SG': 'orange',
    'Unhealthy': 'red',
    'Very Unhealthy': 'purple',
    'Hazardous': 'maroon'
}

st.markdown(
    f"<h2 style='color: {aqi_colors.get(category, 'black')};'>Air Quality Status: {category}</h2>", 
    unsafe_allow_html=True
)
