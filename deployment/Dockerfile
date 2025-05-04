FROM python:3.11-slim

WORKDIR /app

# Install requirements
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code and model
COPY deployment/main.py ./main.py
COPY models/model.joblib ./models/model.joblib
COPY src/models/predict_model.py ./src/models/predict_model.py

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
