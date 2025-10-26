"""
Unit tests for src/data_prep.py
--------------------------------
Tests the data preparation pipeline — loading, cleaning,
feature engineering, CLV computation, and splitting logic.
"""

import os
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_prep


@pytest.fixture
def sample_df():
    data = {
        "customerID": [f"000{i}" for i in range(1, 9)],
        "gender": ["Male", "Female"] * 4,
        "SeniorCitizen": [0, 1, 0, 1, 0, 1, 0, 1],
        "Partner": ["Yes", "No"] * 4,
        "Dependents": ["No", "Yes"] * 4,
        "tenure": [1, 5, 10, 15, 20, 25, 30, 35],
        "PhoneService": ["Yes"] * 8,
        "MultipleLines": ["No"] * 8,
        "InternetService": ["DSL", "Fiber optic"] * 4,
        "OnlineSecurity": ["No", "Yes"] * 4,
        "OnlineBackup": ["Yes", "No"] * 4,
        "DeviceProtection": ["No", "Yes"] * 4,
        "TechSupport": ["Yes", "No"] * 4,
        "StreamingTV": ["Yes", "No"] * 4,
        "StreamingMovies": ["No", "Yes"] * 4,
        "Contract": ["Month-to-month", "Two year"] * 4,
        "PaperlessBilling": ["Yes", "No"] * 4,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 4,
        "MonthlyCharges": [70, 80, 90, 100, 110, 120, 130, 140],
        "TotalCharges": [70, 400, 900, 1500, 2200, 3000, 3900, 4900],
        "Churn": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    }
    return pd.DataFrame(data)



def test_clean_data(sample_df):
    """Ensure TotalCharges is cleaned and missing filled correctly."""
    df = data_prep.clean_data(sample_df.copy())
    assert df["TotalCharges"].dtype in [float, np.float64, np.int64]
    assert df["TotalCharges"].isna().sum() == 0
    # Verify imputed TotalCharges for missing entry
    assert np.isclose(df.loc[1, "TotalCharges"], df.loc[1, "MonthlyCharges"] * df.loc[1, "tenure"], atol=1e-3)


def test_engineer_features(sample_df):
    """Check new engineered columns are added properly."""
    df = data_prep.clean_data(sample_df.copy())
    df = data_prep.engineer_features(df)

    # Check columns exist
    expected_cols = [
        "tenure_bucket", "services_count", "monthly_to_total_ratio", "internet_no_techsupport"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Check services_count logic
    assert all(df["services_count"].between(0, 9))
    assert df["internet_no_techsupport"].isin([0, 1]).all()


def test_compute_clv(sample_df):
    """Check CLV is computed correctly."""
    df = data_prep.clean_data(sample_df.copy())
    df = data_prep.compute_clv(df)
    assert "CLV" in df.columns
    assert (df["CLV"] > 0).all()
    assert np.isclose(df.loc[0, "CLV"], df.loc[0, "MonthlyCharges"] * df.loc[0, "ExpectedTenure"], atol=1e-3)


def test_split_and_save(tmp_path, sample_df, monkeypatch):
    """Verify data is split and saved correctly."""
    # Patch processed dir to temp folder
    monkeypatch.setattr(data_prep, "PROCESSED_DIR", tmp_path)
    df = data_prep.clean_data(sample_df.copy())
    df = data_prep.engineer_features(df)
    df = data_prep.compute_clv(df)
    data_prep.split_and_save(df)

    # Ensure output files exist
    for f in ["train.csv", "val.csv", "test.csv"]:
        assert (tmp_path / f).exists()

    # Validate stratified splits add up
    train = pd.read_csv(tmp_path / "train.csv")
    val = pd.read_csv(tmp_path / "val.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    total_len = len(train) + len(val) + len(test)
    assert total_len == len(df)


def test_load_raw_data(monkeypatch):
    """Mock loading data from URL (don’t hit network)."""
    dummy_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    monkeypatch.setattr(pd, "read_csv", lambda url: dummy_df)
    df = data_prep.load_raw_data("fake_url")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["A", "B"]
