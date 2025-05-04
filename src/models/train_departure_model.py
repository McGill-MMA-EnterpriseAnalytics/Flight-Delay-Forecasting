import pandas as pd
import joblib
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, finalize_model, save_model
from src.data.utils import load_data
from src.features.build_departure_features import preprocess_departure_data


def train():
    mlflow.set_experiment("departure_delay_prediction")

    with mlflow.start_run():
        df = load_data("data/processed/flights_clean.csv")
        X, y = preprocess_departure_data(df)
        df_for_pycaret = pd.DataFrame(
            X, columns=["AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "DISTANCE"]
        )
        df_for_pycaret["DEP_DELAY"] = y

        setup(data=df_for_pycaret, target="DEP_DELAY", session_id=42, verbose=False)

        best_model = compare_models()
        final_model = finalize_model(best_model)

        save_model(final_model, "models/departure_model")
        mlflow.pyfunc.log_model("pycaret_departure_model", python_model=final_model)


if __name__ == "__main__":
    train()
