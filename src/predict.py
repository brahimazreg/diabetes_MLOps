from src.config import DEFAULT_MODEL
from src.data_processing import *
from joblib import load

from monitoring.logger import log_prediction

def get_model():
    return load(DEFAULT_MODEL)

def predict(df):
    model = get_model()
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    result = {
            "prediction": int(pred),
            "probability": float(prob)
        }

   
    log_prediction(df.to_dict(orient="records")[0], result["prediction"], result["probability"])
    return pred, prob
    
if __name__=='__main__':
    df =load_data()
    X,Y = buil_features(df)
    print(predict(X.head()))


    

    

    






