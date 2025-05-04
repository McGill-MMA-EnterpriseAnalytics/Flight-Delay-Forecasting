import pandas as pd


def preprocess_features(df):
    # Drop rows with missing values in key columns
    important_cols = [
        "FL_DATE",
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "DISTANCE",
    ]
    df = df.dropna(subset=important_cols)

    # Create binary target variable
    df["IS_DELAYED"] = (df["DEP_DELAY"] > 15).astype(int)

    # Drop potential leakage columns
    leakage_cols = [
        "DEP_TIME",
        "DEP_DELAY",
        "TAXI_OUT",
        "WHEELS_OFF",
        "WHEELS_ON",
        "TAXI_IN",
        "ARR_TIME",
        "ARR_DELAY",
        "ELAPSED_TIME",
        "AIR_TIME",
        "CANCELLED",
        "CANCELLATION_CODE",
        "DIVERTED",
        "DELAY_DUE_CARRIER",
        "DELAY_DUE_WEATHER",
        "DELAY_DUE_NAS",
        "DELAY_DUE_SECURITY",
        "DELAY_DUE_LATE_AIRCRAFT",
    ]
    df = df.drop(
        columns=[col for col in leakage_cols if col in df.columns], errors="ignore"
    )

    print(f"Data after preprocessing: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
