"""
evaluate.py
------------
Evaluates model performance using predictions from predict.py.
Generates metrics tables, ROC curves, and confusion matrices
for use in Streamlit app dashboards.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------
print("📦 Loading predictions from predict.py...")
pred_path = PROCESSED_DIR / "predictions.csv"
df = pd.read_csv(pred_path)
print(f"✅ Loaded predictions: {df.shape}")

# ---------------------------------------------------------------------
# Metric computation helper
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    """Return dictionary of model metrics."""
    return {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }

# ---------------------------------------------------------------------
# Evaluate all models
# ---------------------------------------------------------------------
metrics = {}

for model in ["logistic", "rf", "xgb"]:
    print(f"\n🔍 Evaluating {model.upper()} model...")
    y_true = df["Actual"]
    y_pred = df[f"{model}_pred"]
    y_proba = df[f"{model}_proba"]

    metrics[model] = compute_metrics(y_true, y_pred, y_proba)
    print(f"✅ {model} metrics: {metrics[model]}")

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics).T.round(4)
metrics_df.to_csv(PROCESSED_DIR / "model_metrics.csv", index=True)
print("\n💾 Saved metrics → data/processed/model_metrics.csv")

# ---------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------
print("\n📊 Generating confusion matrices...")
for model in ["logistic", "rf", "xgb"]:
    cm = confusion_matrix(df["Actual"], df[f"{model}_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {model.upper()}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{model}.png")
    plt.close()
    print(f"✅ Saved: reports/confusion_matrix_{model}.png")

# ---------------------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------------------
print("\n📈 Generating ROC curve overlay...")
plt.figure(figsize=(7, 6))
for model in ["logistic", "rf", "xgb"]:
    fpr, tpr, _ = roc_curve(df["Actual"], df[f"{model}_proba"])
    auc = metrics_df.loc[model, "ROC-AUC"]
    plt.plot(fpr, tpr, label=f"{model.upper()} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "roc_curves.png")
plt.close()
print("✅ Saved: reports/roc_curves.png")

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n📋 Model Evaluation Summary:")
print(metrics_df)
print("\n✅ Evaluation complete — ready for Streamlit app integration.")
