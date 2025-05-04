import mlflow
import mlflow.sklearn
from pycaret.classification import load_model, predict_model
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def log_departure_model():
    mlflow.set_experiment("departure_delay_prediction")

    with mlflow.start_run():
        model = load_model("models/departure_model")

        # Sample input for signature and evaluation
        input_example = pd.DataFrame(
            [
                {
                    "AIRLINE": 3.0,
                    "ORIGIN": 120.0,
                    "DEST": 145.0,
                    "CRS_DEP_TIME": 1230.0,
                    "DISTANCE": 500.0,
                }
            ]
        )

        # Evaluate model on dummy batch
        eval_data = pd.DataFrame(
            [
                {
                    "AIRLINE": 3.0,
                    "ORIGIN": 120.0,
                    "DEST": 145.0,
                    "CRS_DEP_TIME": 1230.0,
                    "DISTANCE": 500.0,
                    "DEP_DELAY": 1,
                },
                {
                    "AIRLINE": 5.0,
                    "ORIGIN": 110.0,
                    "DEST": 148.0,
                    "CRS_DEP_TIME": 930.0,
                    "DISTANCE": 350.0,
                    "DEP_DELAY": 0,
                },
                {
                    "AIRLINE": 2.0,
                    "ORIGIN": 115.0,
                    "DEST": 140.0,
                    "CRS_DEP_TIME": 1500.0,
                    "DISTANCE": 600.0,
                    "DEP_DELAY": 1,
                },
            ]
        )
        predictions = predict_model(model, data=eval_data)
        f1 = f1_score(eval_data["DEP_DELAY"], predictions["prediction_label"])
        acc = accuracy_score(eval_data["DEP_DELAY"], predictions["prediction_label"])

        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # Log model with input example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="pycaret_departure_model",
            input_example=input_example,
        )

        print("Departure model logged to MLflow with evaluation metrics")


if __name__ == "__main__":
    log_departure_model()
