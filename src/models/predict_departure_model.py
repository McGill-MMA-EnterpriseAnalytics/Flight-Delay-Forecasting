import numpy as np
from pycaret.classification import load_model, predict_model


def load_departure_model():
    return load_model("models/departure_model")


def predict_departure(model, input_dict):
    import pandas as pd

    df = pd.DataFrame([input_dict])
    prediction = predict_model(model, data=df)
    return int(prediction["prediction_label"][0])
