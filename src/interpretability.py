"""
Model Interpretability Script (Final Deployment-Safe)
-----------------------------------------------------
- Basic feature importance analysis focusing on raw features.
- NOTE: Current trained models have preprocessing mismatch.
  They expect 4279 features but current pipeline generates fewer.
  TODO: Retrain models with documented preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    
    # Load processed validation data
    print("📦 Loading processed validation data...")
    val_path = Path("data/processed/val.csv")
    val_df = pd.read_csv(val_path)
    print(f"✅ Validation data: {val_df.shape}")

    # Prepare features for basic importance analysis
    print("\nPreparing features for basic importance analysis...")
    X_val = val_df.drop(columns=["Churn", "customerID"])
    y_val = val_df["Churn"]

    num_cols = X_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_val.select_dtypes(include=["object"]).columns.tolist()

    # Create simple preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ], remainder="drop")

    # Process features
    X_processed = preprocessor.fit_transform(X_val)
    feature_names = (num_cols + 
                  [f"{col}_{val}" for col, vals in 
                   zip(cat_cols, preprocessor.named_transformers_["cat"].categories_) 
                   for val in vals[1:]])
    
    print(f"Analyzing {len(feature_names)} raw features...")
    
    # Train a simple model for feature importance
    print("\n🔍 Raw Feature Importance Analysis...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_processed, y_val)
    
    # Calculate importance scores
    importance = np.abs(lr.coef_[0])
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.barh(imp_df["Feature"][:15][::-1], imp_df["Importance"][:15][::-1])
    plt.title("Raw Feature Importance Analysis (Basic Logistic Regression)")
    plt.xlabel("Absolute Coefficient Value")
    plt.tight_layout()
    plt.savefig("reports/raw_feature_importance.png")
    print("✅ Saved: raw_feature_importance.png")
    
    print("\n🔝 Top 10 Most Important Features:")
    print(imp_df.head(10).to_string(index=False))
    
    print("\n✅ Raw feature importance analysis complete.")
