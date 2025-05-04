FROM python:3.11-slim

WORKDIR /app

# Install requirements
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY deployment/main.py ./main.py

# Copy models
COPY models/model.joblib ./models/model.joblib
COPY models/departure_delay_model.joblib ./models/departure_delay_model.joblib

# Copy prediction logic
COPY src/models/predict_arrival_model.py ./src/models/predict_arrival_model.py
COPY src/models/predict_departure_model.py ./src/models/predict_departure_model.py

# Expose port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
