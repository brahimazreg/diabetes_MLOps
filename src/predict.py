from src.config import DEFAULT_MODEL
from src.data_processing import *
from joblib import load


model=load(DEFAULT_MODEL)

def predict(df):
    return model.predict(df)


if __name__=='__main__':
    df=load_data()
    print(predict(df.head()))