from src.data_processing import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.evaluate import evaluate

def test_evaluate_returns_metrics():
    X_train, X_test, y_train, y_test = get_train_test_data()

    model = LogisticRegression()
    processor = buil_preprocessor_regression(X_train)

    pipeline = Pipeline([
        ("preprocessor", processor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    accuracy, recall, f1, roc_auc = evaluate(pipeline,threshold=0.3)

    assert 0 <= accuracy <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    assert 0 <= roc_auc <= 1