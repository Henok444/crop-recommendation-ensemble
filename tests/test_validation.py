
from src.Crop_recommendation.validation import validate_features
import pytest 
valid_features = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 22.5,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 210,
}

def test_valid_feature():
    validate_features(valid_features)

def test_missing_feature():
    features = valid_features.copy()
    del features["K"]

    with pytest.raises(ValueError):
        validate_features(features)

def test_non_numeric_feature():
    features = valid_features.copy()
    features["N"] = "ninety"

    with pytest.raises(ValueError):
        validate_features(features)