import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from mlflow.models.signature import infer_signature
from src.models.evaluate_model import evaluate_model
from src.features.build_departure_features import preprocess_features
import pandas as pd


def train():
    mlflow.set_experiment("departure_delay_prediction")

    with mlflow.start_run():
        df = pd.read_csv("data/processed/flights_clean.csv")
        df = preprocess_features(df)

        X = df.drop(columns=["IS_DELAYED", "FL_DATE"])
        y = df["IS_DELAYED"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Identify column types
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
        numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

        # Preprocessor with imputation
        numeric_transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        model = Pipeline(
            [("preprocessor", preprocessor), ("lr", LogisticRegression(max_iter=1000))]
        )

        model.fit(X_train, y_train)

        f1 = evaluate_model(model, X_val, y_val)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lr_model",
            signature=signature,
            input_example=X_train[:5],
        )

        joblib.dump(model, "models/departure_delay_model.joblib")
        mlflow.log_artifact("models/departure_delay_model.joblib")
        print(f"Logistic Regression model saved with F1: {f1:.4f}")


if __name__ == "__main__":
    train()
