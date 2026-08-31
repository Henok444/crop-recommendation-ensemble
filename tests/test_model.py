from src.Crop_recommendation.model import train_random_forest
from src.Crop_recommendation.data import load_data 
from src.Crop_recommendation.preprocessing import preprocess , split




df = load_data()
X, y = preprocess(df)
X_train, X_test, y_train, y_test = split(X, y , test_size= 0.2 , random_state= 42)


def test_train_random_forest():

    model = train_random_forest(
        X_train,
        y_train,
        n_estimator= 10,
        max_depth =12,
        random_state = 41
        )
    assert len(model.estimators_) == 10