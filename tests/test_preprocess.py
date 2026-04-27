import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")
from preprocess import load_data, build_preprocessor

def make_df():
    return pd.DataFrame({
        "idade": [30, None, 45, 22],
        "renda": [5000.0, 3200.0, None, 7800.0],
        "genero": ["M", "F", None, "M"],
        "satisfacao": [3, 5, 2, 4],
        "churn": [0, 1, 0, 1]
    })

def test_load_data_splits_target():
    df = make_df()
    df.to_csv("/tmp/test.csv", index=False)
    X, y = load_data("/tmp/test.csv")
    assert "churn" not in X.columns
    assert len(y) == 4

def test_no_nulls_after_preprocessing():
    df = make_df()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    pre = build_preprocessor(X)
    X_out = pre.fit_transform(X)
    assert not np.any(np.isnan(X_out))

def test_no_inf_in_output():
    df = make_df()
    X = df.drop(columns=["churn"])
    pre = build_preprocessor(X)
    X_out = pre.fit_transform(X)
    assert np.all(np.isfinite(X_out))