from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    json_data = response.json()
    assert "prediction" in json_data
    assert "probability" in json_data
    assert 0 <= json_data["probability"] <= 1