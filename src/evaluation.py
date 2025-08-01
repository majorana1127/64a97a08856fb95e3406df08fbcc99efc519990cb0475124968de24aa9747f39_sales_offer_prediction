import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def evaluate_model():
    # Load model and test data
    model = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/test_X.csv")
    y_test = pd.read_csv("data/test_y.csv").values.ravel()

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Save results
    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    os.makedirs("outputs", exist_ok=True)
    np.savetxt("outputs/predictions.csv", y_pred, fmt="%d", delimiter=",")