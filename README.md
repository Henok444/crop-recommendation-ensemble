# Ensemble Learning for Precision Agriculture

Crop recommendation system built with ensemble methods — **Random Forest and
XGBoost** — on 2,200 observations across **22 crop categories** and 7
environmental features, with SHAP-based model interpretation.

## Overview

- **Dataset:** 2,200 observations, 22 crop classes, 7 features (N, P, K,
  temperature, humidity, pH, rainfall).
- **Models:** Random Forest and XGBoost classifiers, optimized with
  **GridSearchCV** and early stopping.
- **Results:** Weighted F1 of **0.9932 (Random Forest)** and **0.9885 (XGBoost)**
  on the held-out test set.
- **Interpretation:** **SHAP** analysis identifies rainfall, humidity, and soil
  nutrients as the primary predictors of crop suitability.

## Results

| Model         | Best hyperparameters            | CV F1 | Test weighted F1 |
|---------------|---------------------------------|-------|------------------|
| Random Forest | `max_depth=10, n_estimators=300` | 0.9949 | 0.9932 |
| XGBoost       | `gamma=0.1, learning_rate=0.1`  | 0.9914 | 0.9885 |

## Project Structure

```
notebooks/
  Practical_Ensemble_Learning_Precision_Agriculture.ipynb
    ├── 1. Introduction
    ├── 2. Dataset description & cleaning
    ├── 3. Exploratory data analysis (class distribution, correlations, rainfall/temperature by crop)
    ├── 4. Random Forest development (GridSearchCV, training, evaluation)
    ├── 5. XGBoost development (tuning, early stopping, evaluation)
    ├── 6. Model evaluation & interpretation (confusion matrices, SHAP)
    └── 7. Conclusion
data/
  Crop_recommendation.csv
```

## Setup

```
pip install -r requirements.txt
```

Open `notebooks/Practical_Ensemble_Learning_Precision_Agriculture.ipynb`
and run top-to-bottom.

## Built With

Python · Scikit-learn · XGBoost · SHAP · Pandas · NumPy · Matplotlib · Seaborn
