from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_random_forest(
        X_train , 
        y_train ,
        param_grid,
        cv = 5,
        scoring = "f1_weighted",

):
    model = RandomForestClassifier(
        random_state=42 ,
        n_jobs=-1
    )
    search = GridSearchCV(
        estimator=model ,
        param_grid=param_grid,
        cv=cv ,
        scoring=scoring,
        n_jobs=-1

    )
    search.fit(X_train , y_train)

    return search