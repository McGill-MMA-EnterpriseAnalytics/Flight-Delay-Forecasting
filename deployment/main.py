from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict_arrival_model import (
    load_model as load_arrival_model,
    predict as predict_arrival,
)
from src.models.predict_departure_model import (
    load_departure_model,
    predict_departure_model,
    FlightInput as DepartureInput,
)

app = FastAPI()

arrival_model = load_arrival_model("models/model.joblib")
departure_model = load_departure_model("models/departure_delay_model.joblib")


# Input model for arrival
class ArrivalInput(BaseModel):
    AIRLINE: int
    ORIGIN: int
    DEST: int
    CRS_DEP_TIME: int
    DISTANCE: float


@app.get("/")
def root():
    return {"message": "Flight Delay Prediction API is live!"}


@app.post("/predict_arrival")
def predict_arrival_delay(input_data: ArrivalInput):
    result = predict_arrival(arrival_model, input_data.dict())
    return {"arrival_prediction": result}


@app.post("/predict_departure")
def predict_departure_delay(input_data: DepartureInput):
    return predict_departure_model(departure_model, input_data)
