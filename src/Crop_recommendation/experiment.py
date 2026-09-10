from src.Crop_recommendation.config import load_config
from src.Crop_recommendation.data import load_data
from src.Crop_recommendation.evaluation import evaluate_model
from src.Crop_recommendation.model import train_random_forest , train_xgboost
from src.Crop_recommendation.preprocessing import encode ,preprocess, split
from src.Crop_recommendation.tuning import tune_random_forest
from src.Crop_recommendation.dataset_info import get_dataset_info
from datetime import datetime




def run_random_forest_experiment():


    config = load_config()

    data = load_data()

    dataset_info = get_dataset_info(data)
    data2  , lr = encode(data)

    X , y = preprocess(data2)

    X_train , X_test , y_train , y_test = split(X , y , random_state= config["data"]["random_state"] , test_size=config["data"]["test_size"])


    model = train_random_forest(

        X_train,
        y_train,
        n_estimator=config["random_forest"]["n_estimator"],
        max_depth=config["random_forest"]["max_depth"],
        random_state=config["random_forest"]["random_state"]
    )
    results = evaluate_model(
        model=
        model,
        X_test=X_test,
        y_test=y_test,
    )
    time_stamp = datetime.now().isoformat()
    return {
        "experiment_id": time_stamp , 
        "time_stamp": time_stamp, 
        "model": "random_forest",
        "dataset": dataset_info,
        "parameters": {
            "n_estimators": config["random_forest"]["n_estimator"],
            "max_depth": config["random_forest"]["max_depth"],
        },
        "accuracy": results["accuracy"],
        "weighted_f1": results["weighted_f1"],
    }
def run_xgboost_experiment():


    config = load_config()

    data = load_data()
    dataset_info = get_dataset_info(data)
    data2 , lr = encode(data)

    X , y = preprocess(data2)

    X_train , X_test , y_train , y_test = split(X , y , random_state= config["data"]["random_state"] , test_size=config["data"]["test_size"])

    model = train_xgboost(
        X_train,y_train ,
        n_estimators=config["xg_boost"]["n_estimators"],
        gamma=config["xg_boost"]["gamma"] ,
        learning_rate=config["xg_boost"]["learning_rate"],
        random_state=config["xg_boost"]["random_state"] , 
    )

    result = evaluate_model(X_test,y_test,model)
    time_stamp = datetime.now().isoformat()

    return {
        "experiment_id": time_stamp , 
        "time_stamp": time_stamp , 
        "model": "xg_boost",
        "dataset": dataset_info,
        "parameters":
            {
                "n_estimators": config["xg_boost"]["n_estimators"],
                "gamma" : config["xg_boost"]["gamma"] ,
                "learning_rate": config["xg_boost"]["learning_rate"],
                "random_state" : config["xg_boost"]["random_state"]             
            } ,
        "accuracy" : result["accuracy"],
        "weighted_f1": result["weighted_f1"]

    }

def run_random_forest_tuning_experiment():

    config = load_config()

    data = load_data()
    dataset_info = get_dataset_info(data)
    data2 , lr  = encode(data)

    X , y = preprocess(data2)

    X_train , X_test , y_train , y_test = split(X , y , random_state= config["data"]["random_state"] , test_size=config["data"]["test_size"])

    param_grid = {
        "n_estimators": config["tuning"]["random_forest"]["n_estimators"],
        "max_depth": config["tuning"]["random_forest"]["max_depth"] , 
    
    }


    search = tune_random_forest(X_train , 
                                y_train , 
                                param_grid ,
                                cv = config["tuning"]["random_forest"]["cv"] ,
                                scoring= config["tuning"]["random_forest"]["scoring"] )
    best_model = search.best_estimator_

    result = evaluate_model(X_test , y_test , best_model)
    time_stamp = datetime.now().isoformat()

    return {
        "experimet_id": time_stamp, 
        "time_stamp": time_stamp,
        "model": "random_forest",
        "dataset": dataset_info,
        "best_parameters": search.best_params_,
        "cv_score": search.best_score_,
        "accuracy": result["accuracy"],
        "weighted_f1": result["weighted_f1"],
    }