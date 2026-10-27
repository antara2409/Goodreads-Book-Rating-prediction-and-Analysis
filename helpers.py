import json
import os

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"
MODEL_NAMES = ["Random Forest", "SVR", "Linear Regression", "KNN"]
PREFIX = {"Random Forest": "rf", "SVR": "svr", "Linear Regression": "lr", "KNN": "knn"}

# Column names of the original Goodreads dataset (used for single input and CSV upload)
RAW_COLS = [
    "ratings_count", "work_ratings_count", "work_text_reviews_count",
    "toread_count", "currently_reading", "books_count",
    "original_publication_year", "language_code", "authors", "tag_name",
]


def load_models():
    m = {"pre": joblib.load(f"{MODELS_DIR}/svr_preprocessor.joblib")}
    m["Random Forest"] = joblib.load(f"{MODELS_DIR}/rf_pipeline.joblib")
    m["KNN"] = joblib.load(f"{MODELS_DIR}/knn_pipeline.joblib")
    m["SVR"] = joblib.load(f"{MODELS_DIR}/svr_model.joblib")
    m["Linear Regression"] = joblib.load(f"{MODELS_DIR}/lr_model.joblib")
    return m


def load_metrics():
    out = {}
    for name, p in PREFIX.items():
        path = f"{MODELS_DIR}/{p}_metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning + feature engineering as the notebook (works for 1 or many rows)."""
    d = raw.copy()
    year = pd.to_numeric(d["original_publication_year"], errors="coerce")
    d["original_publication_year"] = year.fillna(year.median() if year.notna().any() else 2000)
    for c in ["ratings_count", "work_ratings_count", "work_text_reviews_count",
              "toread_count", "currently_reading", "books_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).clip(lower=0)
    d["language_code"] = d["language_code"].fillna("eng")
    d["authors"] = d["authors"].fillna("Unknown").replace("", "Unknown")
    d["tag_name"] = d["tag_name"].fillna("")

    out = pd.DataFrame(index=d.index)
    out["books_count"] = d["books_count"]
    out["book_age"] = 2026 - d["original_publication_year"]
    out["review_ratio"] = d["work_text_reviews_count"] / (d["work_ratings_count"] + 1)
    out["engagement_ratio"] = d["currently_reading"] / (d["toread_count"] + 1)
    out["ratings_count_log"] = np.log1p(d["ratings_count"])
    out["work_ratings_count_log"] = np.log1p(d["work_ratings_count"])
    out["work_text_reviews_count_log"] = np.log1p(d["work_text_reviews_count"])
    out["toread_count_log"] = np.log1p(d["toread_count"])
    out["currently_reading_log"] = np.log1p(d["currently_reading"])
    out["language_code"] = d["language_code"]
    out["authors"] = d["authors"]
    out["tag_name"] = d["tag_name"]
    return out


def predict(models, name, X: pd.DataFrame) -> np.ndarray:
    if name in ("Random Forest", "KNN"):          # full pipelines
        p = models[name].predict(X)
    else:                                         # SVR / Linear Regression
        p = models[name].predict(models["pre"].transform(X))
    return np.clip(p, 1, 5)


def predict_all(models, X: pd.DataFrame) -> dict:
    return {n: float(predict(models, n, X)[0]) for n in MODEL_NAMES}