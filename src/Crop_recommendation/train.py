from src.Crop_recommendation.data import load_data
from src.Crop_recommendation.preprocessing import preprocess 
from src.Crop_recommendation.preprocessing import split 
from src.Crop_recommendation.model import train_random_forest , train_xgboost
from src.Crop_recommendation.evaluation import evalute_model
from src.Crop_recommendation.config import load_config
from src.Crop_recommendation.model_io import save_model
from src.Crop_recommendation.preprocessing import encode
from pathlib import Path



def train_rm():

    config = load_config()

    data = load_data()
    encoded_data = encode(data)

    X , y = preprocess(encoded_data)

    X_train , X_test , y_train , y_test = split(X ,
                                                y,
                                                random_state= config["data"]["random_state"],
                                                test_size= config["data"]["test_size"]) 
    model = train_random_forest(X_train= X_train ,
                                y_train= y_train ,
                                max_depth=config["random_forest"]["max_depth"],
                                n_estimator= config["random_forest"]["n_estimator"],
                                random_state=config["random_forest"]["random_state"])
    model_path = Path(config["trained_model"]["random_forest_path"])
    save_model(model , model_path)
    print(f"model saved to: {model_path}")

    results = evalute_model(model=model , X_test= X_test , y_test= y_test)


    return model, results
def train_xg():

    config = load_config()

    data = load_data()
    encoded_data = encode(data)

    X , y = preprocess(encoded_data)

    X_train , X_test , y_train , y_test = split(
                                                X ,
                                                y,
                                                random_state= config["data"]["random_state"],
                                                test_size= config["data"]["test_size"]
                                                )
     
    



    model = train_xgboost(      X_train= X_train ,
                                y_train= y_train ,
                                n_estimators = config["xg_boost"]["n_estimators"],
                                random_state=config["xg_boost"]["random_state"],
                                gamma = config["xg_boost"]["gamma"],
                                learning_rate=  config["xg_boost"]["learning_rate"]
                                )

    
    model_path = Path(config["trained_model"]["xg_boost_path"])
    save_model(model , model_path)
    print(f"model saved to: {model_path}")

    results = evalute_model(model=model , X_test= X_test , y_test= y_test)


    return model, results

if __name__ == "__main__":
    random_forest_model , random_forest_results = train_rm()
    xgboost_model , xgboost_results = train_xg()
    print("Training complete.")
    print("Random forest results")
    print(f"Accuracy: {random_forest_results['accuracy']:.4f}")
    print(f"Weighted F1: {random_forest_results['weighted_f1']:.4f}")
    print("xgboost results")
    print(f"Accuracy: {xgboost_results['accuracy']:.4f}")
    print(f"Weighted F1: {xgboost_results['weighted_f1']:.4f}")

