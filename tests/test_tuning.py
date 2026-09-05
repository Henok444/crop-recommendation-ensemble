from src.Crop_recommendation.data import load_data
from src.Crop_recommendation.preprocessing import encode , preprocess , split
from src.Crop_recommendation.config import load_config
from src.Crop_recommendation.tuning import tune_random_forest

config = load_config()

def test_random_forest_tuning():
    df = load_data()
    df_encode = encode(df)
    X , y = preprocess(df_encode)


    X_train , X_test , y_train , y_test = split(
        X , y , random_state= config["data"]["random_state"] , test_size= config["data"]["test_size"]
    )

    param_grid = {
        "n_estimators":[7],
        "max_depth": [4]
    }
    search = tune_random_forest(X_train, y_train , param_grid , )

    assert search.best_params_["n_estimators"] == 7
    assert search.best_params_["max_depth"] == 4 
