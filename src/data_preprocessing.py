import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data():
    df = pd.read_csv("data/bank.csv")

    df = df.drop(columns=["duration"])

    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    numeric_features = ["age", "balance", "day", "campaign", "pdays", "previous"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_processed = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train).to_csv("data/train_X.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/test_X.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/train_y.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/test_y.csv", index=False)

    return (X_train, y_train), (X_test, y_test)
