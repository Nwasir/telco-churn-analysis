"""
src/clv_analysis.py
----------------
Analyzes Customer Lifetime Value (CLV) segments
and churn behavior for business insights.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("data", "processed")


def load_data():
    """Load processed train dataset."""
    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    print(f"✅ Loaded training data: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    return df_train


def create_clv_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Split CLV into 4 quartile segments."""
    df["CLV_quartile"] = pd.qcut(df["CLV"], q=4, labels=["Low", "Medium", "High", "Premium"])
    return df


def churn_rate_by_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate churn rate by CLV quartile."""
    churn_summary = (
        df.groupby("CLV_quartile")["Churn"]
        .mean()
        .reset_index()
        .rename(columns={"Churn": "ChurnRate"})
    )
    churn_summary["ChurnRate"] = churn_summary["ChurnRate"] * 100
    print("\n📊 Churn rate by CLV quartile:")
    print(churn_summary)
    return churn_summary


def plot_clv_distribution(df: pd.DataFrame):
    """Plot distribution of CLV values."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["CLV"], bins=30, kde=True, color="steelblue")
    plt.title("CLV Distribution")
    plt.xlabel("Customer Lifetime Value (CLV)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clv_distribution.png"))
    plt.close()
    print("📈 Saved: data/processed/clv_distribution.png")


def plot_churn_by_quartile(churn_summary: pd.DataFrame):
    """Plot churn rate by CLV quartile."""
    plt.figure(figsize=(7, 4))
    sns.barplot(x="CLV_quartile", y="ChurnRate", data=churn_summary, palette="Blues_d")
    plt.title("Churn Rate by CLV Segment")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("CLV Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "churn_by_clv.png"))
    plt.close()
    print("📉 Saved: data/processed/churn_by_clv.png")


def business_insights(churn_summary: pd.DataFrame):
    """Generate key business insights from CLV & churn relationship."""
    print("\n💡 Business Insights:")
    low_churn = churn_summary.iloc[0]["ChurnRate"]
    high_churn = churn_summary.iloc[-1]["ChurnRate"]

    print(f"1️⃣ Low-CLV customers churn at about {low_churn:.1f}% rate — often low tenure or monthly contracts.")
    print(f"2️⃣ Premium-CLV customers churn at only {high_churn:.1f}% — retaining them yields the biggest ROI.")
    print("3️⃣ Strategy: target Medium-CLV users with loyalty offers to boost retention before they downgrade.")


def main():
    df = load_data()
    df = create_clv_quartiles(df)
    churn_summary = churn_rate_by_clv(df)
    plot_clv_distribution(df)
    plot_churn_by_quartile(churn_summary)
    business_insights(churn_summary)
    print("\n✅ CLV analysis complete.")


if __name__ == "__main__":
    main()
