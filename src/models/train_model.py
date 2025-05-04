import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
import xgboost as xgb
from mlflow.models.signature import infer_signature
from src.data.utils import load_data
from src.features.build_features import preprocess_data
from src.models.evaluate_model import evaluate_model


def train():
    mlflow.set_experiment("flight_delay_prediction")

    with mlflow.start_run():
        df = load_data("data/processed/flights_clean.csv")
        X, y = preprocess_data(df)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        f1 = evaluate_model(model, X_val, y_val)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgb_model",
            signature=signature,
            input_example=X_train[:5],
        )

        joblib.dump(model, "models/model.joblib")
        mlflow.log_artifact("models/model.joblib")


if __name__ == "__main__":
    train()
