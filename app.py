import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load model
with open("aqi_lgbm_full_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_columns = ['TempC','Humidity','PM2.5','PM10','NO2 ppb','O3 ppb','CO ppm']

# AQI Category bins (only your desired ranges)
bins = [150, 200, 300, 500]  # Start from 150
labels = ['Unhealthy', 'Very Unhealthy', 'Hazardous']

aqi_suggestions = {
    'Unhealthy': 'Avoid prolonged outdoor activities. Use mask if sensitive.',
    'Very Unhealthy': 'Stay indoors, limit outdoor exertion.',
    'Hazardous': 'Remain indoors, use air purifiers, avoid all outdoor activity.'
}

st.set_page_config(page_title="AQI Gauge Predictor", layout="centered")
st.title("🌤️ AQI Predictor with Gauge")

# User input
def user_input_features():
    TempC = st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
    Humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    PM25 = st.sidebar.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
    PM10 = st.sidebar.number_input("PM10 (µg/m³)", 0.0, 500.0, 80.0)
    NO2 = st.sidebar.number_input("NO2 (ppb)", 0.0, 1000.0, 40.0)
    O3 = st.sidebar.number_input("O3 (ppb)", 0.0, 500.0, 30.0)
    CO = st.sidebar.number_input("CO (ppm)", 0.0, 50.0, 1.0)
    
    data = {
        'TempC': TempC,
        'Humidity': Humidity,
        'PM2.5': PM25,
        'PM10': PM10,
        'NO2 ppb': NO2,
        'O3 ppb': O3,
        'CO ppm': CO
    }
    return pd.DataFrame([data])

input_df = user_input_features()
input_df = input_df[feature_columns]

# Prediction
prediction = model.predict(input_df)[0]
category = pd.cut([prediction], bins=bins, labels=labels)[0]
suggestion = aqi_suggestions.get(category, "Be careful!")

# Determine main pollutant (highest input value among pollutants)
pollutant_columns = ['PM2.5','PM10','NO2 ppb','O3 ppb','CO ppm']
main_pollutant = input_df[pollutant_columns].iloc[0].idxmax()

# Gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': f"Main Pollutant - {main_pollutant}\nAQI Category: {category}"},
    gauge={
        'axis': {'range': [150, 500]},  # Start gauge at 150
        'bar': {'color': "red"},
        'steps': [
            {'range': [150, 200], 'color': "orange"},      # Unhealthy
            {'range': [200, 300], 'color': "purple"},      # Very Unhealthy
            {'range': [300, 500], 'color': "maroon"},      # Hazardous
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# Suggestion
st.subheader("💡 Suggestion")
st.info(suggestion)
