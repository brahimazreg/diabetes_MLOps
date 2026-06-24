import streamlit as st
import requests

st.title("Diabetes Prediction")

pregnancies = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
blood_pressure = st.number_input("Blood Pressure", 0)
skin_thickness = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 1)

if st.button("Predict"):

    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=payload
    )

    st.write(response.json())
 