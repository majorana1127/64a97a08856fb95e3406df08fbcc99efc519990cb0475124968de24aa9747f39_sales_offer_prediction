import os
import numpy as np

def evaluate_model(model, test_data):
    X_test, y_test = test_data
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    os.makedirs("outputs", exist_ok=True)
    np.savetxt("outputs/predictions.csv", y_pred, fmt="%d", delimiter=",")
