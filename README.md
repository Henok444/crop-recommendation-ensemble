# a full system that predict the best crop to grow.

This project is an end to end  machine learning system for crop recommendation.

  - first it analayzed **Dataset:** 2,200 observations, 22 crop classes, 7 features (N, P, K,
  temperature, humidity, pH, rainfall).
  - then do experiment **Models:** Random Forest and XGBoost classifiers, optimized with
  **GridSearchCV** and early stopping.
  - then got **Results:** Weighted F1 of **0.9932 (Random Forest)** and **0.9885 (XGBoost)**
  on the held-out test set.
  - then **SHAP** analysis identifies rainfall, humidity, and soil nutrients as the primary predictors of crop suitability.


## Results

| Model         | Best hyperparameters            | CV F1 | Test weighted F1 |
|---------------|---------------------------------|-------|------------------|
| Random Forest | `max_depth=10, n_estimators=300` | 0.9949 | 0.9932 |
| XGBoost       | `gamma=0.1, learning_rate=0.1`  | 0.9914 | 0.9885 |


