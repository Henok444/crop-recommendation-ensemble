from src.Crop_recommendation.validation import validate_features
from src.Crop_recommendation.model_io import load_model
from src.Crop_recommendation.prediction import predict
from src.Crop_recommendation.config import load_config
import pandas as pd 


config = load_config()


features = {
    "N": 90,
    "K": 43,
    "P": 50 , 
    "temperature": 22.5,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 210,
}


def test_prediction():
    model = load_model(config["model"]["path"])
    prediction = predict(model , features)

    assert isinstance(prediction , str)
    assert len(prediction) > 0


