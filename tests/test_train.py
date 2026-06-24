from src.train import train_model
from sklearn.linear_model import LogisticRegression

def test_train_runs():
    model = LogisticRegression()

    pipeline, X_test, Y_test = train_model("test_lr", model)

    assert pipeline is not None
    assert X_test is not None
    assert Y_test is not None