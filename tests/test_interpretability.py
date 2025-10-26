import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import pytest

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util

@pytest.fixture
def mock_data(tmp_path):
    """Create mock validation dataset compatible with interpretability.py"""
    data = {
        "customerID": [f"000{i}" for i in range(1, 6)],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "tenure": [5, 10, 15, 20, 25],
        "MonthlyCharges": [70.5, 80.2, 60.3, 90.0, 75.5],
        "Contract": ["Month-to-month", "Two year", "Month-to-month", "One year", "Two year"],
        "Churn": [0, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    val_path = processed_dir / "val.csv"
    df.to_csv(val_path, index=False)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    return tmp_path


def test_interpretability_script_runs(mock_data, monkeypatch):
    """Test if interpretability.py runs successfully and produces output"""
    src_path = Path(__file__).resolve().parents[1] / "src" / "interpretability.py"
    spec = importlib.util.spec_from_file_location("interpretability", src_path)
    interpretability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interpretability)

    # Change working directory to mock temp folder
    monkeypatch.chdir(mock_data)

    # Copy the script temporarily
    shutil.copy(src_path, mock_data / "interpretability.py")

    # Run script in isolation
    os.system(f"python {mock_data}/interpretability.py")

    # Verify output image exists
    output_path = mock_data / "reports" / "raw_feature_importance.png"
    assert output_path.exists(), "Expected raw_feature_importance.png not found in reports/"

    # Verify the output file is not empty
    assert output_path.stat().st_size > 0, "Generated importance plot is empty"


def test_import_and_structure(monkeypatch, mock_data):
    """Test internal DataFrame structure after mock training"""
    import matplotlib
    matplotlib.use("Agg")  # Prevent GUI popups during tests

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression

    val_df = pd.read_csv(mock_data / "data/processed/val.csv")
    X_val = val_df.drop(columns=["Churn", "customerID"])
    y_val = val_df["Churn"]

    num_cols = X_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_val.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X_val)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_processed, y_val)

    importance = np.abs(lr.coef_[0])
    imp_df = pd.DataFrame({
        "Feature": num_cols + [
            f"{col}_{val}" for col, vals in zip(cat_cols, preprocessor.named_transformers_["cat"].categories_) 
            for val in vals[1:]
        ],
        "Importance": importance
    })

    # Assertions
    assert "Feature" in imp_df.columns
    assert "Importance" in imp_df.columns
    assert not imp_df.empty
    assert np.all(imp_df["Importance"] >= 0)
