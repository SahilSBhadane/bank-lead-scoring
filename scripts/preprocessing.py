import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- KNOWN LEAKAGE COLUMN ---
# 'duration' (last call length in seconds) is only known AFTER the call ends.
# duration=0 is a near-perfect predictor of "no" (call never connected), and
# successful calls average 2.5x longer than unsuccessful ones. UCI's own
# dataset documentation flags this: duration should be excluded from any
# model meant to predict conversion *before* a call happens, since a lead
# scoring model must work on information available BEFORE contact.
# We keep it available for an explicit benchmark-only comparison in
# modeling.py, but it is EXCLUDED from the real deliverable model.
LEAKAGE_COLS = ["duration"]

# pdays=999 is a sentinel meaning "never contacted before", not a real value.
# Left as-is for tree models (they can split on it), but documented here so
# it isn't mistaken for a data error.
SENTINEL_NOTE = "pdays == 999 means 'client was not previously contacted'"

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]


def split_features_target(df: pd.DataFrame, include_duration: bool = False):
    df = df.copy()
    y = (df["y"] == "yes").astype(int)

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    if include_duration:
        feature_cols = feature_cols + ["duration"]

    X = df[feature_cols]
    return X, y


def build_preprocessor(include_duration: bool = False) -> ColumnTransformer:
    numeric_cols = NUMERIC_COLS + (["duration"] if include_duration else [])

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
