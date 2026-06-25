
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data_processing import *
from src.evaluate import evaluate
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from xgboost import XGBClassifier

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_experiment")

def train_model(model_name, model):

    X_train, X_test, Y_train, Y_test = get_train_test_data()
    
    # this to calculate the value of scale_pos_weight=1.87
    neg =(Y_train==0).sum()
    pos =(Y_train==1).sum()

    print("negative",neg  )
    print("positive",pos  )
    
    processor = None

    if model_name == "lr":
        processor = buil_preprocessor_regression(X_train)

    steps = []

    if processor:
        steps.append(("preprocessor", processor))

    steps.append(("model", model))
    pipeline = Pipeline(steps)

    with mlflow.start_run(run_name=model_name):

        pipeline.fit(X_train, Y_train)

        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        dump(
            pipeline,
            MODEL_PATH / f"{model.__class__.__name__}.joblib"
        )

        accuracy, recall, f1score, roc_auc = evaluate(pipeline,threshold=0.3)

        mlflow.log_param(
            "model",
            model.__class__.__name__
        )
              
        mlflow.log_params(model.get_params())
        mlflow.log_param("threshold", 0.3)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "recall": recall,
            "f1score": f1score,
            "roc_auc": roc_auc
        })

        mlflow.log_param("negative_samples", int(neg))
        mlflow.log_param("positive_samples", int(pos))
        mlflow.log_param("imbalance_ratio", float(neg / pos))
        
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="DiabetesModel",
            serialization_format='cloudpickle',
            signature=signature
        )

        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("framework", "sklearn")
        
        # Log training data shape
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("features", X_train.shape[1])
        
        # Save feature names
        mlflow.log_dict(
        {"features": list(X_train.columns)},
        "feature_schema.json"
        )
        
        

    return pipeline, X_test, Y_test
    
models={
    "lr": LogisticRegression(random_state=RANDOM_STATE ,max_iter=MAX_ITER),
    "rf": RandomForestClassifier(criterion='gini', n_estimators=200, random_state=RANDOM_STATE ),
    "xgb": XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE, scale_pos_weight=1.87,
)
}


if __name__=='__main__' :
    for model_name, model in models.items():
        train_model(model_name, model)
    
