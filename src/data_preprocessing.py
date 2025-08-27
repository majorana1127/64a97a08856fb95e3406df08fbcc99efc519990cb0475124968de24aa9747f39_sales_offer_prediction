import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_ingestion import download_bank_data

def _flip_categoricals(df: pd.DataFrame, frac=0.12, rng=None):
    rng = rng or np.random.default_rng(42)
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        mask = rng.random(len(out)) < frac
        cats = out[col].dropna().unique()
        if len(cats) > 1 and mask.any():
            out.loc[mask, col] = rng.choice(cats, size=mask.sum(), replace=True)
    return out

def preprocess_data():
    bank_csv = download_bank_data()
    df = pd.read_csv(bank_csv)
    df = df.drop(columns=["duration"])
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    numeric_features = ["age", "balance", "day", "campaign", "pdays", "previous"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_processed = pipeline.fit_transform(X)
    feature_names = (
        numeric_features +
        list(pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features))
    )
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    X_train.to_csv("data/train_X.csv", index=False)
    X_test.to_csv("data/test_X.csv", index=False)
    y_train.to_csv("data/train_y.csv", index=False)
    y_test.to_csv("data/test_y.csv", index=False)

    X_train_num = X_train[numeric_features]
    std = X_train_num.std(ddof=0).replace(0, 1.0)
    noise = np.random.default_rng(0).normal(loc=0.0, scale=0.1 * std, size=X_train_num.shape)

    X_train_drifted = X_train.copy()
    X_test_drifted = X_test.copy()
    X_train_drifted[numeric_features] = X_train_num * 1.2 + noise
    X_test_num = X_test[numeric_features]
    X_test_drifted[numeric_features] = X_test_num * 1.2

    X_train_drifted = _flip_categoricals(X_train_drifted, frac=0.12)
    X_test_drifted = _flip_categoricals(X_test_drifted, frac=0.12)

    X_train_drifted.assign(target=y_train.reset_index(drop=True)).to_csv("data/drifted_train.csv", index=False)
    X_test_drifted.assign(target=y_test.reset_index(drop=True)).to_csv("data/drifted_test.csv", index=False)

    return (
        X_train, X_test, y_train, y_test,
        X_train_drifted, y_train, X_test_drifted, y_test
    )

# if __name__ == "__main__":
#     preprocess_data()