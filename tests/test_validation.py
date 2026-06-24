from fastapi.testclient import TestClient
from src.api import app   # adjust if your path is different

client = TestClient(app)

def test_invalid_input():
    payload = {
        "Pregnancies": -10,  # invalid
        "Glucose": "abc",    # invalid type
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }

    response = client.post("/predict", json=payload)

    assert response.status_code != 200
