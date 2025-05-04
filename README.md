Flight-Delay-Prediction
==============================

## Problem Statement

Airlines continue to experience substantial financial losses and declining customer satisfaction due to **unpredictable flight arrival delays**. Despite historical schedule adjustments, existing models often struggle to accurately forecast delays under dynamic conditions such as:
- Weather disruptions
- Airport congestion
- Operational inefficiencies

These challenges lead to:
- Poor gate planning
- Missed passenger connections
- Higher compensation costs
- Erosion of brand loyalty

---

## Objective

Develop a **highly accurate and scalable machine learning solution** to predict both **departure and arrival delays** in advance.  
This enables:
- Proactive operations management
- Optimized resource allocation
- Enhanced customer experience

---

## Project Overview

### Goals
- Predict **departure delays** based only on **pre-takeoff** information
- Predict **arrival delays** incorporating **pre-flight and post-departure** data

### Tools & Frameworks
- PyCaret (AutoML)
- XGBoost + Optuna
- FastAPI
- MLflow
- Docker

---

## Dataset Overview

- **Source:** Kaggle - Flight Delay and Cancellation Dataset (2019–2023)
- **Size:** 3 million rows (`flights_sample_3m.csv`)
- **Regions:** Major U.S. airports
- **Time Period:** 2019 to 2023
- **Type:** Historical flight records

### Features
- `flight_date`, `airline`, `flight_number`
- `origin_airport`, `destination_airport`
- `scheduled_departure`, `scheduled_arrival`
- `departure_delay`, `arrival_delay`, `status`
- `air_time`, `distance`

### Targets
- `IS_DELAYED`: Departure Delay > 15 mins → 1 (Delayed), else 0
- `IS_ARRIVAL_DELAYED`: Arrival Delay > 15 mins → 1 (Delayed), else 0

### Challenges
- Missing/null values
- Cancellations and diversions
- Extreme outliers > 1000 minutes

---

## Exploratory Data Insights

1. **Class imbalance**: Majority (~82%) of flights are on time
2. **Skewed delays**: Departure delays are mostly minor, with few >300 mins
3. **Airline-specific**: JetBlue and Frontier have the most delays

---

## Departure Delay Prediction Model

- **Sample Size:** 300,000 rows
- **Framework:** PyCaret AutoML
- **Target Variable:** `IS_DELAYED`

### Preprocessing
- Drop leakage columns: `DEP_DELAY`, `TAXI_OUT`, `WHEELS_OFF`
- Time feature extraction: hour, minute, day, month
- Distance binning via quantile cuts
- SMOTE for class imbalance (automated via PyCaret)

### Modeling
- Compared: Logistic Regression, XGBoost, LightGBM, Random Forest
- Final Model: **Stacked Ensemble of top 3**
- GPU Enabled

---

## Arrival Delay Prediction Model

- **Sample Size:** 1,000,000 rows
- **Framework:** Manual XGBoost + Optuna (20 trials)
- **Target Variable:** `IS_ARRIVAL_DELAYED`

### Preprocessing
- Winsorized `ARR_DELAY` at 1st and 99th percentile
- Dropped leakage columns: `ARR_TIME`, `AIR_TIME`, `DEP_DELAY`
- SMOTE for imbalance after split
- GPU-enabled with `tree_method=hist` and `device=cuda`

### Modeling
- Feature engineering: Extracted time features + distance bins
- 3-Fold Stratified Cross-validation

---

## Deployment Architecture

### Components

1. **FastAPI Application**  
   - Accepts flight feature inputs
   - Sends REST request to model server

2. **MLflow Model Server**  
   - Hosts trained models
   - Responds with prediction output

3. **MLflow Tracking Server**  
   - Tracks experiments, hyperparameters, metrics
   - Manages model registry

4. **Docker**  
   - Ensures environment reproducibility
   - All services run in isolated containers


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
