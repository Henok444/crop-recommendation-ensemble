import pandas as pd 
from src.Crop_recommendation.validation import validate_features


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
    y = model.predict(x)
    return y[0]

