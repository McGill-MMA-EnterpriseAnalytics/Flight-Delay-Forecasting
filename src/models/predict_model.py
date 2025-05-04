import joblib
import numpy as np


def load_model(model_path="models/model.joblib"):
    return joblib.load(model_path)


def predict(model, input_dict):
    input_array = np.array(
        [
            input_dict["AIRLINE"],
            input_dict["ORIGIN"],
            input_dict["DEST"],
            input_dict["CRS_DEP_TIME"],
            input_dict["DISTANCE"],
        ]
    ).reshape(1, -1)
    return int(model.predict(input_array)[0])
