"""
data_prep.py
-------------
Loads and cleans the IBM Telco Churn dataset,
engineers CLV-related features, and saves train/val/test splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
PROCESSED_DIR = os.path.join("data", "processed")


def load_raw_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load dataset from URL."""
    df = pd.read_csv(url)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing TotalCharges and convert types."""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Replace NaN TotalCharges with MonthlyCharges * tenure
    missing_total = df["TotalCharges"].isna().sum()
    if missing_total > 0:
        df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)
        print(f"Filled {missing_total} missing TotalCharges values.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven features for CLV and churn modeling."""
    # tenure buckets
    bins = [0, 6, 12, 24, np.inf]
    labels = ["0-6m", "6-12m", "12-24m", "24m+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)

    # total number of services (simple count of 'Yes' in selected cols)
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["services_count"] = df[service_cols].apply(
        lambda x: np.sum(x == "Yes"), axis=1
    )

    # monthly_to_total_ratio
    df["monthly_to_total_ratio"] = df["TotalCharges"] / np.maximum(
        1, df["tenure"] * df["MonthlyCharges"]
    )

    # flag: internet but no tech support
    df["internet_no_techsupport"] = np.where(
        (df["InternetService"] != "No") & (df["TechSupport"] == "No"), 1, 0
    )

    return df


def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CLV = MonthlyCharges × ExpectedTenure."""
    # expected tenure assumption: mean tenure of non-churned customers
    expected_tenure = df.loc[df["Churn"] == "No", "tenure"].mean()
    df["ExpectedTenure"] = expected_tenure
    df["CLV"] = df["MonthlyCharges"] * df["ExpectedTenure"]
    print(f"Expected Tenure assumed: {expected_tenure:.1f} months")
    return df


def split_and_save(df: pd.DataFrame):
    """Split into train/val/test and save to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 60/20/20 split with stratification
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["Churn"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Churn"], random_state=42)

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print("✅ Saved processed splits to data/processed/")


def main():
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = compute_clv(df)
    split_and_save(df)


if __name__ == "__main__":
    main()
