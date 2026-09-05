from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier



def train_random_forest(X_train , y_train , n_estimator  , max_depth , random_state   ):
    model = RandomForestClassifier(n_estimators= n_estimator ,
                                   max_depth=max_depth ,
                                   random_state=random_state , 
                                   n_jobs = -1)
    model.fit(X_train , y_train)
    return model

def train_xgboost(  
                    X_train,
                    y_train,
                    n_estimators,
                    learning_rate,
                    gamma,
                    random_state,
                    early_stopping_rounds=20
                    ):
    

    
    # Create validation split from training data
    X_train2, X_val, y_train2, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42 )

    # all the experiment is done in the notebook , 

    best_xgb = XGBClassifier(
                            learning_rate=learning_rate,       # best param from GridSearch
                            gamma=gamma,               # best param from GridSearch
                            n_estimators=n_estimators,        
                            eval_metric='mlogloss',
                            early_stopping_rounds=early_stopping_rounds, 
                            random_state=random_state
                            )

    best_xgb.fit(
                X_train2,
                y_train2,
                eval_set=[(X_val, y_val)],
                verbose=False
                )
    return best_xgb 

