from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict_model import load_model, predict

app = FastAPI()
model = load_model("models/model.joblib")


class FlightInput(BaseModel):
    AIRLINE: int
    ORIGIN: int
    DEST: int
    CRS_DEP_TIME: int
    DISTANCE: float


@app.get("/")
def root():
    return {"message": "Flight Delay Prediction API is live!"}


@app.post("/predict")
def predict_delay(input_data: FlightInput):
    result = predict(model, input_data.dict())
    return {"prediction": result}
