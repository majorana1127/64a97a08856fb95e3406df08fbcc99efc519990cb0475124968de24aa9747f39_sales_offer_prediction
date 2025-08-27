import xgboost as xgb
import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc
import os

class CustomXGBModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
    def predict(self, context, model_input):
        return self.model.predict(model_input)

def train_model():
    mlflow.set_tracking_uri("http://localhost:5000")

    X_train = pd.read_csv("data/train_X.csv")
    y_train = pd.read_csv("data/train_y.csv").values.ravel()

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    params = {
        "max_depth": 5,
        "n_estimators": 200,
        "learning_rate": 0.1
    }

    os.makedirs("mlflow/artifacts", exist_ok=True)
    with mlflow.start_run():
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("learning_rate", params["learning_rate"])

        model = xgb_clf.set_params(**params).fit(X_train, y_train)
        joblib.dump(model, "mlflow/artifacts/model.pkl")

        mlflow.pyfunc.save_model(
            path="mlflow_model",
            python_model=CustomXGBModel(),
            artifacts={"model": "mlflow/artifacts/model.pkl"}
        )
        mlflow.log_artifacts("mlflow/artifacts")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return model