import pandas as pd
import joblib
import pytest
import sys, os
sys.path.append(os.path.abspath("src"))
import predict


@pytest.mark.order(1)
def test_predict_script_runs(monkeypatch):
    """Ensure predict_all_models runs end-to-end and saves output."""
    # Mock prints for cleaner test output
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # Ensure models and val.csv exist
    assert os.path.exists("models/logistic.pkl"), "Missing logistic model file"
    assert os.path.exists("models/rf.pkl"), "Missing RF model file"
    assert os.path.exists("models/xgb.pkl"), "Missing XGB model file"
    assert os.path.exists("data/processed/val.csv"), "Missing validation dataset"

    # Run prediction
    results = predict.predict_all_models()

    # Assertions on result dataframe
    assert isinstance(results, pd.DataFrame)
    assert not results.empty, "Prediction results should not be empty"
    assert "logistic_pred" in results.columns
    assert "rf_pred" in results.columns
    assert "xgb_pred" in results.columns

    # Check output file
    assert os.path.exists("data/processed/predictions.csv"), "Predictions CSV not saved"


@pytest.mark.order(2)
def test_saved_predictions_are_valid():
    """Ensure predictions.csv contains expected structure and values."""
    output_path = "data/processed/predictions.csv"
    assert os.path.exists(output_path), "Prediction CSV not found"

    df = pd.read_csv(output_path)

    # Basic structural checks
    expected_cols = [
        "CustomerID",
        "logistic_pred",
        "logistic_proba",
        "rf_pred",
        "rf_proba",
        "xgb_pred",
        "xgb_proba",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Sanity check on predictions
    for col in ["logistic_pred", "rf_pred", "xgb_pred"]:
        assert set(df[col].unique()).issubset({0, 1}), f"{col} contains invalid labels"
    for col in ["logistic_proba", "rf_proba", "xgb_proba"]:
        assert df[col].between(0, 1).all(), f"{col} contains invalid probabilities"
