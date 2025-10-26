import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shutil
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# --- Fixtures ---------------------------------------------------------------

@pytest.fixture
def sample_data(tmp_path):
    """Create small fake training/validation CSVs to simulate real data."""
    train_df = pd.DataFrame({
        "customerID": ["0001", "0002", "0003", "0004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 1],
        "MonthlyCharges": [70.0, 120.4, 90.5, 60.3],
        "TotalCharges": [2397.6, 120.4, 1398.6, 120.4],
        "Churn": [1, 0, 1, 0],
    })

    val_df = pd.DataFrame({
        "customerID": ["0005", "0006"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [1, 0],
        "MonthlyCharges": [80.0, 100.0],
        "TotalCharges": [1500.0, 2500.0],
        "Churn": [1, 0],
    })

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    return {
        "train": train_path,
        "val": val_path,
        "models_dir": models_dir,
        "tmp_path": tmp_path
    }

# --- Tests ------------------------------------------------------------------

def test_train_models_runs(monkeypatch, sample_data):
    """
    Verify that the train_models.py script runs end-to-end
    and produces the expected model artifacts and metrics file.
    """

    # Patch file paths inside train_models
    monkeypatch.setattr("builtins.open", open)
    monkeypatch.chdir(sample_data["tmp_path"])

    # Copy train_models.py into temp src directory for import
    src_dir = sample_data["tmp_path"] / "src"
    src_dir.mkdir()
    original_script = Path(__file__).parents[1] / "src" / "train_models.py"
    shutil.copy(original_script, src_dir / "train_models.py")

    # Run script inside temp folder
    import runpy
    result = runpy.run_path(src_dir / "train_models.py")

    # --- Assertions ---
    models_path = sample_data["tmp_path"] / "models"
    assert (models_path / "logistic.pkl").exists(), "Logistic model not saved"
    assert (models_path / "rf.pkl").exists(), "Random Forest model not saved"
    assert (models_path / "xgb.pkl").exists(), "XGBoost model not saved"

    metrics_path = models_path / "metrics_summary.csv"
    assert metrics_path.exists(), "Metrics summary file not found"

    metrics_df = pd.read_csv(metrics_path)
    assert not metrics_df.empty, "Metrics summary is empty"
    assert set(["Model", "Precision", "Recall", "F1", "AUC"]).issubset(metrics_df.columns)

def test_saved_models_are_valid(monkeypatch, sample_data):
    """Ensure saved models can be loaded and used for prediction."""
    monkeypatch.chdir(sample_data["tmp_path"])

    src_dir = sample_data["tmp_path"] / "src"
    src_dir.mkdir()
    original_script = Path(__file__).parents[1] / "src" / "train_models.py"
    shutil.copy(original_script, src_dir / "train_models.py")

    import runpy
    runpy.run_path(src_dir / "train_models.py")

    # Load one model (logistic)
    model_path = sample_data["tmp_path"] / "models" / "logistic.pkl"
    model = joblib.load(model_path)

    # Prepare sample input
    sample = pd.DataFrame({
        "customerID": ["0099"],
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "MonthlyCharges": [99.0],
        "TotalCharges": [2000.0]
    })

    preds = model.predict(sample)
    assert preds.shape[0] == 1, "Prediction output shape mismatch"
