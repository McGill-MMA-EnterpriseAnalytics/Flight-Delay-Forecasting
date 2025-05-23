import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
    important_cols = [
        "FL_DATE",
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "DISTANCE",
    ]
    df = df.dropna(subset=important_cols)

    lower_bound = np.percentile(df["ARR_DELAY"].dropna(), 1)
    upper_bound = np.percentile(df["ARR_DELAY"].dropna(), 99)
    df["ARR_DELAY"] = np.clip(df["ARR_DELAY"], lower_bound, upper_bound)

    le = LabelEncoder()
    for col in ["AIRLINE", "ORIGIN", "DEST"]:
        df[col] = le.fit_transform(df[col])

    X = df[["AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "DISTANCE"]]
    y = (df["ARR_DELAY"] > 15).astype(int)

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res
