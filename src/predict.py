from src.config import DEFAULT_MODEL
from src.data_processing import *
from joblib import load
from functools import lru_cache

@lru_cache
def get_model():
    return load(DEFAULT_MODEL)

def predict(df):
    model = get_model()
    return model.predict(df)







