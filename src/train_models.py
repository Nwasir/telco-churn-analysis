# src/train_models.py
"""
Train and evaluate churn prediction models:
 - Logistic Regression
 - Random Forest
 - XGBoost

Saves:
 - models/logistic.pkl
 - models/rf.pkl
 - models/xgb.pkl
 - models/metrics_summary.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import os

# ---------------------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------------------
print("📦 Loading processed training and validation data...")
train_path = "data/processed/train.csv"
val_path = "data/processed/val.csv"

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
print(f"✅ Training data: {df_train.shape}, Validation data: {df_val.shape}")

# ---------------------------------------------------------------------
# 2. Define features and target
# ---------------------------------------------------------------------
target_col = "Churn"
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col].astype(int)
X_val = df_val.drop(columns=[target_col])
y_val = df_val[target_col].astype(int)

# ---------------------------------------------------------------------
# 3. Identify numeric and categorical columns
# ---------------------------------------------------------------------
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# ---------------------------------------------------------------------
# 4. Preprocessing pipeline
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ---------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000, random_state=42),
    "rf": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "xgb": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# ---------------------------------------------------------------------
# 6. Train and evaluate
# ---------------------------------------------------------------------
results = []

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\n🚀 Training {name.upper()}...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    y_prob = pipe.predict_proba(X_val)[:, 1]

    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    results.append({
        "Model": name,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "AUC": round(auc, 3)
    })

    # Save model
    model_path = f"models/{name}.pkl"
    joblib.dump(pipe, model_path)
    print(f"✅ Saved: {model_path}")

# ---------------------------------------------------------------------
# 7. Save metrics summary
# ---------------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("models/metrics_summary.csv", index=False)

print("\n📊 Model Performance Summary:")
print(metrics_df)
print("\n🏁 Training complete.")
