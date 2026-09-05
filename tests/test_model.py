from src.Crop_recommendation.model import train_random_forest
from src.Crop_recommendation.data import load_data 
from src.Crop_recommendation.preprocessing import preprocess , split
from src.Crop_recommendation.model import train_xgboost
from src.Crop_recommendation.preprocessing import encode




df = load_data()
df_encoded = encode(df)
X, y = preprocess(df_encoded)
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

def test_train_xg_boost():

    model = train_xgboost(
        X_train,
        y_train,
        n_estimators=10,
        learning_rate=0.1,
        gamma=0.1,
        random_state=42,
    )
    assert model.n_estimators == 10 

