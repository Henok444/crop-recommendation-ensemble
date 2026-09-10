from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.Crop_recommendation.model_io import load_model
from src.Crop_recommendation.prediction import predict
from src.Crop_recommendation.validation import validate_features


app = FastAPI(
    title="Crop Recommendation API",
    version="1.0.0",
)


MODEL_PATH = Path("models/random_forest.pkl")

model = load_model(MODEL_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict_(features: dict):
    try:
        validate_features(features)

        crop = predict(
            model,
            features,
        )

        return {
            "recommended_crop": crop,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )