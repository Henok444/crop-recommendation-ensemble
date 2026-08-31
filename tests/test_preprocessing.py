from src.Crop_recommendation.data import load_data
from src.Crop_recommendation.preprocessing import preprocess
from src.Crop_recommendation.preprocessing import split
from src.Crop_recommendation.config import load_config

config = load_config()

def test_preprocess():
    df = load_data()

    X , y = preprocess(df)
    assert X.shape == (2200,7)
    assert y.shape ==(2200,)
    assert list(X.columns) == [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]


def test_split():
    df = load_data()

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = split(X, y, test_size= config["data"]["test_size"] ,random_state= config["data"]["random_state"])

    assert len(X_train) == 1760
    assert len(X_test) == 440
    assert len(y_train) == 1760
    assert len(y_test) == 440