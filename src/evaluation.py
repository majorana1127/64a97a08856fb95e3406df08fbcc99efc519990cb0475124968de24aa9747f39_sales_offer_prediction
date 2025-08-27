import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import json
import os

def evaluate_model():
    model = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/test_X.csv")
    y_test = pd.read_csv("data/test_y.csv").values.ravel()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    result = {"accuracy": float(acc), "f1_score": float(f1)}
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_results.json", "w") as f:
        json.dump(result, f, indent=2)

    np.savetxt("outputs/predictions.csv", y_pred, fmt="%d", delimiter=",")

    return result