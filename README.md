# 🧠 Diabetes Prediction MLOps Project

An end-to-end **Machine Learning Operations (MLOps)** project for diabetes prediction using FastAPI, MLflow, Docker, CI/CD, and Streamlit monitoring dashboard.

---

## 🚀 Project Overview

This project demonstrates a full MLOps pipeline:

- Data preprocessing & feature engineering
- Model training (Logistic Regression)
- Experiment tracking with MLflow
- REST API deployment using FastAPI
- Docker containerization
- Automated testing with Pytest
- Model monitoring using logging + Streamlit dashboard

---

## 📊 ML Problem

Predict whether a patient has diabetes based on medical features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

# 🏗️ Project Structure

```
diabetes_MLOps/
│
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│
├── tests/
│   ├── test_predict.py
│   └── sample_input.py
│
├── models/
│   └── (saved models .joblib)
│
├── data/
│   └── raw/diabetes.csv
│
├── api.py
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---


---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- FastAPI
- Streamlit
- MLflow
- Docker
- Pytest
- Pandas / NumPy

---

## 🧪 Model Training

- Algorithm: Logistic Regression
- Pipeline: Scikit-learn Pipeline (Preprocessing + Model)
- Evaluation metrics:
  - Accuracy
  - Recall
  - F1-score
  - ROC-AUC

- Experiment tracking:
  - MLflow logs parameters, metrics, and models

---

## 📡 API (FastAPI)
    https://diabetes-mlops-rsds.onrender.com/docs


## 1. Clone repository

```bash
git clone https://github.com/your-username/diabetes_MLOps.git
cd diabetes_MLOps
```

---

## 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Training the Model

Run training pipeline:

```bash
python -m src.train
```

This will:

* Train model
* Save model in `/models`
* Log experiment in MLflow

---

# 📊 MLflow Tracking

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```
http://127.0.0.1:5000
```

You will see:

* Metrics (accuracy, recall, F1, ROC AUC)
* Parameters
* Model artifacts

---

# 🌐 Run FastAPI Server

```bash
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

### Endpoints:

* `GET /` → health check
* `GET /sample` → sample input
* `POST /predict` → prediction

---

# 🐳 Run with Docker

## Build image

```bash
docker build -t diabetes-api .
```

## Run container

```bash
docker run -p 8000:8000 diabetes-api
```

---

# 🧪 Run Tests

```bash
pytest
```

---

# 📦 Input Features

The model expects:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

---

# 📈 Model Metrics (Example)

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~0.71 |
| Recall   | ~0.52 |
| F1-score | ~0.56 |
| ROC AUC  | ~0.82 |

---

# 🔮 Future Improvements

* Add Streamlit UI for predictions
* Use MLflow Model Registry
* Add CI/CD pipeline (GitHub Actions)
* Deploy on cloud (Render / AWS / Azure)
* Add monitoring (data drift, model drift)

---

# 👨‍💻 Author
Brahim Azreg 
Built as a learning project for MLOps fundamentals:

* ML pipeline design
* Model tracking
* API deployment
* Dockerization
* CI/CD readiness

---

# 📌 Notes

* Dataset must be placed in `data/raw/diabetes.csv`
* Model is retrained from scratch on each run
* MLflow logs are stored locally using SQLite backend
