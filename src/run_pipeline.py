from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift
import mlflow

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    (
        X_train, X_test, y_train, y_test,
        X_train_drifted, y_train_drifted,
        X_test_drifted, y_test_drifted
    ) = preprocess_data()

    model = train_model()

    results = evaluate_model()

    drift_result = detect_drift("data/test_X.csv", "data/drifted_test.csv")
    mlflow.log_param("test_drift_detected", drift_result["drift_detected"])
    mlflow.log_param("test_overall_drift_score", drift_result["overall_drift_score"])

    if drift_result["drift_detected"]:
        raise ValueError("Data drift detected in test set! Model retraining required.")

if __name__ == "__main__":
    main()