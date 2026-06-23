from src.data_processing import buil_features,load_data


import pandas as pd

def sample_input():
    return pd.DataFrame({
        "Pregnancies": [1, 2],
        "Glucose": [120, 85],
        "BloodPressure": [70, 80],
        "SkinThickness": [20, 25],
        "Insulin": [0, 30],
        "BMI": [25.0, 30.0],
        "DiabetesPedigreeFunction": [0.2, 0.4],
        "Age": [25, 35]
    })