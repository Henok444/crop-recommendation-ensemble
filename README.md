# Ensemble Learning for Precision Agriculture

crop recommendation system built with ensemble methods **Random Forest and
XGBoost** on 2,200 observations across **22 crop categories** and 7
environmental features, with SHAP-based model interpretation.

## overview

- **dataset:** 2,200 observations, 22 crop classes, 7 features (N, P, K,
  temperature, humidity, pH, rainfall).
- **models:** Random Forest and XGBoost classifiers, optimized with
  **grridSearchCV** and early stopping.
- **Results:** Weighted F1 of **0.9932 (random Forest)** and **0.9885 (xGBoost)**
  on the held-out test set.
- **interpretation:** **SHAP** analysis identifies rainfall, humidity, and soil
  nutrients as the primary predictors of crop suitability.

## results

| model         | best hyperparameters            | CV F1 | test weighted F1 |
|---------------|---------------------------------|-------|------------------|
| Random forest | `max_depth=10, n_estimators=300` | 0.9949 | 0.9932 |
| XGBoost       | `gamma=0.1, learning_rate=0.1`  | 0.9914 | 0.9885 |



