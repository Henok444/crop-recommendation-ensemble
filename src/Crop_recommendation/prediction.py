import pandas as pd 
from src.Crop_recommendation.validation import validate_features
from src.Crop_recommendation.config import load_config
from src.Crop_recommendation.model_io import load_encoder

config = load_config()

column_order = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

def predict(model , features ):

    validate_features(features)

    x = pd.DataFrame([features] , columns= column_order)
    y_numeric = model.predict(x)
    lr = load_encoder(config["encoder"]["path"])
    y_text = lr.inverse_transform(y_numeric)
    return y_text[0]

