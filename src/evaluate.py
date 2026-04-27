from sklearn.metrics import (
    accuracy_score, f1_score,
    recall_score, precision_score
)


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
    }