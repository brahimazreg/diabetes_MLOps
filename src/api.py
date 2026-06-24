from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from tests.sample_input import sample_input
from src.predict import predict

app = FastAPI()


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API"}

@app.get("/healthy")
def health():
    return {"status": "ok "}


@app.post("/predict")
def make_prediction(data: DiabetesInput):
    df = pd.DataFrame([data.model_dump()])
    prediction, probability = predict(df)

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model_version": "v1.0"
    }   



@app.get("/sample")
def get_sample_input():
    df = sample_input()
    return df.to_dict(orient="records")
