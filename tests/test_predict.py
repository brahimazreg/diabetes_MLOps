from tests.sample_input import sample_input
from src.predict import predict


def test_predic():
    df=sample_input()
    output =predict(df)
    assert output is not None
    assert set(output).issubset({0, 1})