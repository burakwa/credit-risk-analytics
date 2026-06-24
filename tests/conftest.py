"""
Paylaşılan pytest fixture'ları.
"""
import io
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Ham veri fixture'ı (German Credit Dataset şemasına uygun)
# ---------------------------------------------------------------------------
SAMPLE_ROWS = [
    {
        "Age": 22, "Sex": "male", "Job": 2, "Housing": "own",
        "Saving accounts": "little", "Checking account": "little",
        "Credit amount": 1169, "Duration": 6, "Purpose": "furniture/equipment",
        "Risk": "good",
    },
    {
        "Age": 49, "Sex": "female", "Job": 2, "Housing": "own",
        "Saving accounts": "little", "Checking account": "moderate",
        "Credit amount": 5951, "Duration": 48, "Purpose": "furniture/equipment",
        "Risk": "bad",
    },
    {
        "Age": 45, "Sex": "male", "Job": 2, "Housing": "free",
        "Saving accounts": None, "Checking account": None,   # NaN → unknown
        "Credit amount": 2096, "Duration": 12, "Purpose": "education",
        "Risk": "good",
    },
    {
        "Age": 53, "Sex": "male", "Job": 2, "Housing": "free",
        "Saving accounts": "little", "Checking account": "little",
        "Credit amount": 7882, "Duration": 42, "Purpose": "furniture/equipment",
        "Risk": "good",
    },
    {
        "Age": 35, "Sex": "male", "Job": 1, "Housing": "free",
        "Saving accounts": "little", "Checking account": "little",
        "Credit amount": 4870, "Duration": 24, "Purpose": "car",
        "Risk": "bad",
    },
]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """German Credit şemasına uygun küçük bir DataFrame döner."""
    return pd.DataFrame(SAMPLE_ROWS)


@pytest.fixture
def sample_csv(tmp_path, sample_df) -> str:
    """Geçici bir CSV dosyası oluşturur ve yolunu döner."""
    path = tmp_path / "german_credit_data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_csv_with_unnamed(tmp_path, sample_df) -> str:
    """'Unnamed: 0' sütunu içeren CSV döner (gerçek veri setini taklit eder)."""
    path = tmp_path / "german_credit_with_index.csv"
    sample_df.to_csv(path, index=True)   # index=True → 'Unnamed: 0' üretir
    return str(path)


# ---------------------------------------------------------------------------
# Sahte (mock) eğitilmiş pipeline fixture'ı
# ---------------------------------------------------------------------------
@pytest.fixture
def trained_pipeline(sample_df) -> Pipeline:
    """
    ModelTrainer'ı kullanmadan basit bir sklearn pipeline eğitir.
    Inference testlerinde gerçek model yerine bu kullanılır.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    binary_cols = ["Sex"]
    categorical_cols = ["Housing", "Saving accounts", "Checking account", "Purpose"]
    numerical_cols = ["Age", "Credit amount", "Duration"]

    df = sample_df.copy()
    df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    X = df.drop(columns=["Risk", "Job"])
    y = df["Risk"].map({"good": 1, "bad": 0})

    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")),
    ])
    bin_tf = Pipeline([
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num_tf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])

    pre = ColumnTransformer([
        ("num", num_tf, numerical_cols),
        ("cat", cat_tf, categorical_cols),
        ("bin", bin_tf, binary_cols),
    ])

    pipe = Pipeline([
        ("preprocessor", pre),
        ("classifier", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def saved_pipeline_path(tmp_path, trained_pipeline) -> str:
    """Eğitilen pipeline'ı tmp dizinine kaydeder ve yolunu döner."""
    import joblib
    path = tmp_path / "model.pkl"
    joblib.dump(trained_pipeline, path)
    return str(path)
