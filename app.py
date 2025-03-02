import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image

model = tf.keras.models.load_model("MODEL (1).h5")

bg_image = "jjjjj.jpeg"  

st.set_page_config(page_title="Diabetic Neuropathy Prediction", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🦶 Diabetic Neuropathy Prediction App")
st.write("Enter patient details to predict the likelihood of Diabetic Neuropathy.")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
diabetes_duration = st.number_input("Diabetes Duration (years)", min_value=0, max_value=50, value=10)
emg_freq = st.number_input("EMG Signal Frequency (Hz)", min_value=80, max_value=150, value=120)
emg_amp = st.number_input("EMG Amplitude (mV)", min_value=2.0, max_value=5.0, value=3.5)
motor_ncv = st.number_input("Motor Nerve Conduction Velocity (m/s)", min_value=25, max_value=50, value=40)
sensory_ncv = st.number_input("Sensory Nerve Conduction Velocity (m/s)", min_value=20, max_value=45, value=35)
f_wave = st.number_input("F-Wave Latency (ms)", min_value=30, max_value=55, value=40)
emg_duration = st.number_input("EMG Signal Duration (ms)", min_value=5, max_value=20, value=10)
resting_emg = st.number_input("Resting EMG Activity (µV)", min_value=10, max_value=35, value=20)
muscle_affected = st.selectbox("Muscle Affected", ["Tibialis Anterior", "Gastrocnemius", "Quadriceps"])

gender_encoded = 1 if gender == "Male" else 0
muscle_encoded = ["Tibialis Anterior", "Gastrocnemius", "Quadriceps"].index(muscle_affected)

input_data = np.array([[age, diabetes_duration, emg_freq, emg_amp, motor_ncv, sensory_ncv, 
                        f_wave, emg_duration, resting_emg, gender_encoded, muscle_encoded]])

scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

input_data = input_data.reshape(1, input_data.shape[1], 1)

if st.button("Predict Neuropathy"):
    prediction = model.predict(input_data)
    result = "🟥 High Risk of Neuropathy" if prediction[0][0] > 0.5 else "🟩 Low Risk of Neuropathy"
    st.subheader(f"**Prediction: {result}**")
