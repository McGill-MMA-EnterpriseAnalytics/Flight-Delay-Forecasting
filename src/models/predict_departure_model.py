from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib


def load_departure_model(model_path="models/departure_delay_model.joblib"):
    return joblib.load(model_path)


class FlightInput(BaseModel):
    AIRLINE: str
    ORIGIN: str
    DEST: str
    CRS_DEP_TIME: float
    CRS_ARR_TIME: float
    FL_DATE: str
    DISTANCE: float


def preprocess_input(input_data: FlightInput) -> pd.DataFrame:
    df = pd.DataFrame([input_data.dict()])

    # Feature engineering (must match training `preprocess_features`)
    df["CRS_DEP_HOUR"] = df["CRS_DEP_TIME"] // 100
    df["CRS_DEP_MINUTE"] = df["CRS_DEP_TIME"] % 100
    df["CRS_ARR_HOUR"] = df["CRS_ARR_TIME"] // 100
    df["CRS_ARR_MINUTE"] = df["CRS_ARR_TIME"] % 100

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY"] = df["FL_DATE"].dt.day

    df["DISTANCE_BIN"] = pd.qcut(df["DISTANCE"], q=5, labels=False, duplicates="drop")

    drop_cols = ["CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE", "DISTANCE"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df


def predict_departure_model(model, input_data: FlightInput):
    try:
        features = preprocess_input(input_data)
        prediction = int(model.predict(features)[0])
        score = (
            float(model.predict_proba(features)[0][1])
            if hasattr(model, "predict_proba")
            else "N/A"
        )
        return {
            "prediction_label": prediction,
            "prediction_score": score,
        }
    except Exception as e:
        return {"error": str(e)}
