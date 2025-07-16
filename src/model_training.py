import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV

def train_model(train_data):
    X_train, y_train = train_data

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, "models/model.pkl")

    return best_model
