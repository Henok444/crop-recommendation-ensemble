from src.Crop_recommendation.data import load_data
from src.Crop_recommendation.preprocessing import preprocess 
from src.Crop_recommendation.preprocessing import split 
from src.Crop_recommendation.model import train_random_forest
from src.Crop_recommendation.evaluation import evalute_model
from src.Crop_recommendation.config import load_config
from src.Crop_recommendation.model_io import save_model
from pathlib import Path



def train():

    config = load_config()

    data = load_data()

    X , y = preprocess(data)

    X_train , X_test , y_train , y_test = split(X ,
                                                y,
                                                random_state= config["data"]["random_state"],
                                                test_size= config["data"]["test_size"]) 
    model = train_random_forest(X_train= X_train ,
                                y_train= y_train ,
                                max_depth=config["random_forest"]["max_depth"],
                                n_estimator= config["random_forest"]["n_estimator"],
                                random_state=config["random_forest"]["random_state"])
    model_path = Path(config["model"]["path"])
    save_model(model , model_path)
    print(f"model saved to: {model_path}")

    results = evalute_model(model=model , X_test= X_test , y_test= y_test)


    return model, results

if __name__ == "__main__":
    model , results = train()
    print("Training complete.")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
