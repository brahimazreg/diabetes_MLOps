from tests.sample_input import sample_input
from src.predict import predict


def test_predic():
    df=sample_input()
    prediction, probability = predict(df)   

    assert prediction in [0, 1]
    assert 0.0 <= probability <= 1.0

    
    