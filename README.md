# Ensemble Learning for Precision Agriculture

A machine learning project that predicts suitable crop categories from soil nutrient measurements and environmental conditions. The project compares Random Forest and XGBoost classifiers and packages the trained Random Forest model behind a REST API built with FastAPI.

Live API: https://crop-recommendation-ensemble-1.onrender.com
API Documentation: https://crop-recommendation-ensemble-1.onrender.com/docs


## Overview

This project explores how supervised machine learning can support crop recommendation using soil and weather-related features.

Two tree-based classification algorithms are evaluated:

Random Forest

XGBoost

The project goes beyond model training by organizing the code into reusable modules, tracking experiments, evaluating model performance, interpreting feature importance with SHAP, and serving predictions through a containerized API.

## Dataset

The dataset contains approximately 2,200 observations, 7 input features, and 22 crop categories.

## Results

| Model         | Best hyperparameters            | CV F1 | Test weighted F1 |
|---------------|---------------------------------|-------|------------------|
| Random Forest | `max_depth=10, n_estimators=300` | 0.9949 | 0.9932 |
| XGBoost       | `gamma=0.1, learning_rate=0.1`  | 0.9914 | 0.9885 |

## System Architecture

The application follows a simple inference workflow:

- A client submits soil and environmental measurements.

- FastAPI validates the request.

- The prediction module loads or accesses the trained model.

- The model predicts a crop category.

- The API returns the prediction to the client.



## Project Structure

app/ --API application and endpoint definitions.

config/ -- Configuration used by the project.

data/ -- Dataset files, if included in the repository.

models/ -- Serialized model artifacts, if included.

notebooks/ -- Exploratory analysis and experiments.

results/ -- Evaluation outputs and reports.

src/ -- Reusable project logic, including prediction code.

tests/ --  Automated tests.

Dockerfile -- Instructions for building the container image.

requirements.txt -- Python dependencies.

# Set up guid 


git clone https://github.com/Henok444/crop-recommendation-ensemble.git
cd crop-recommendation-ensemble

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.api:app --reload

docker build -t crop-recommendation-api .
docker run --rm -p 8000:8000 crop-recommendation-api

pytest

curl http://127.0.0.1:8000/health

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.5,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
  }'



