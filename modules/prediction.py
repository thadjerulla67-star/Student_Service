import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category"
]

def predict_one(model, row):
    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(model.predict(X)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])
    return pred, probability

def get_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, pred, zero_division=0), 4),
        "TN": int(confusion_matrix(y_test, pred).ravel()[0]),
        "FP": int(confusion_matrix(y_test, pred).ravel()[1]),
        "FN": int(confusion_matrix(y_test, pred).ravel()[2]),
        "TP": int(confusion_matrix(y_test, pred).ravel()[3]),
    }
