from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

def evalute_model(X_test , y_test , model ):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred=y_pred , y_true=y_test)
    weighted_f1 = f1_score(y_pred=y_pred , y_true=y_test , average="weighted")
    report = classification_report(y_pred=y_pred , y_true=y_test)

    return {"accuracy" : accuracy,
            "weighted_f1": weighted_f1,
            "classification_report" : report,}