import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import clv_analysis


@pytest.fixture
def sample_df(tmp_path):
    """Provide a small mock dataset with CLV and Churn columns."""
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(1, 9)],
        "CLV": np.linspace(100, 1000, 8),
        "Churn": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    file_path = tmp_path / "train.csv"
    df.to_csv(file_path, index=False)
    return df


@pytest.mark.order(1)
def test_create_clv_quartiles(sample_df):
    """Verify CLV quartile labels are added and valid."""
    df = clv_analysis.create_clv_quartiles(sample_df.copy())
    assert "CLV_quartile" in df.columns
    assert set(df["CLV_quartile"].unique()) <= {"Low", "Medium", "High", "Premium"}


@pytest.mark.order(2)
def test_churn_rate_by_clv(sample_df):
    """Ensure churn rate summary is computed correctly."""
    df = clv_analysis.create_clv_quartiles(sample_df.copy())
    churn_summary = clv_analysis.churn_rate_by_clv(df)
    assert {"CLV_quartile", "ChurnRate"}.issubset(churn_summary.columns)
    assert churn_summary["ChurnRate"].between(0, 100).all()


@pytest.mark.order(3)
def test_plot_functions(tmp_path, sample_df, monkeypatch):
    """Check that plot functions save PNG files successfully."""
    df = clv_analysis.create_clv_quartiles(sample_df.copy())
    churn_summary = clv_analysis.churn_rate_by_clv(df)

    # Patch output dir to temp folder
    monkeypatch.setattr(clv_analysis, "OUTPUT_DIR", tmp_path)

    clv_analysis.plot_clv_distribution(df)
    clv_analysis.plot_churn_by_quartile(churn_summary)

    dist_path = tmp_path / "clv_distribution.png"
    churn_path = tmp_path / "churn_by_clv.png"
    assert dist_path.exists()
    assert churn_path.exists()


@pytest.mark.order(4)
def test_main_runs_without_error(monkeypatch, tmp_path, sample_df):
    """Run main() with mocked data paths to ensure full flow works."""
    # Prepare mock data
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(data_dir / "train.csv", index=False)

    # Patch directories
    monkeypatch.setattr(clv_analysis, "PROCESSED_DIR", str(data_dir))
    monkeypatch.setattr(clv_analysis, "OUTPUT_DIR", str(data_dir))

    # Run main pipeline
    clv_analysis.main()

    # Confirm output files
    assert (data_dir / "clv_distribution.png").exists()
    assert (data_dir / "churn_by_clv.png").exists()
