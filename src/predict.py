"""
src/predict.py
-----------
Generates churn predictions using trained models.
Handles both pipeline and non-pipeline models automatically.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Paths
DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUTPUT_DIR = os.path.join("data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_validation_data():
    """Load processed validation dataset."""
    val_path = os.path.join(DATA_DIR, "val.csv")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"❌ Validation data not found at {val_path}")
    df_val = pd.read_csv(val_path)
    print(f"✅ Loaded validation data: {df_val.shape}")
    return df_val


def load_model(model_name):
    """Load a trained model (pipeline or plain)."""
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found: {path}")
    model = joblib.load(path)
    print(f"📦 Loaded model: {model_name}")
    return model


def get_X_y(df):
    """Separate features and target if present."""
    if "Churn" in df.columns:
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
    else:
        X = df
        y = None
    return X, y


def safe_predict(model, X):
    """Predict with model, whether pipeline or not."""
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
    except Exception:
        # If model doesn't support predict_proba, fallback to predict()
        print("⚠️ Model lacks predict_proba, using predict() only.")
        y_pred_proba = model.predict(X)
    
    try:
        y_pred = model.predict(X)
    except Exception:
        print("⚠️ Model lacks predict(), thresholding probabilities at 0.5.")
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return y_pred, y_pred_proba


def predict_all_models():
    """Run predictions for all available models."""
    df_val = load_validation_data()
    X_val, y_val = get_X_y(df_val)

    results = pd.DataFrame()
    results["CustomerID"] = df_val.get("customerID", range(len(df_val)))

    for model_name in ["logistic", "rf", "xgb"]:
        try:
            model = load_model(model_name)
            y_pred, y_proba = safe_predict(model, X_val)
            results[f"{model_name}_pred"] = y_pred
            results[f"{model_name}_proba"] = y_proba
            print(f"✅ Predictions done for {model_name}")
        except Exception as e:
            print(f"❌ Failed for {model_name}: {e}")

    # If ground truth exists
    if y_val is not None:
        results["Actual"] = y_val

    # Save predictions
    output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"\n💾 Saved predictions → {output_path}")

    # Quick summary
    print("\n📊 Sample results:")
    print(results.head())
    return results


def main():
    predict_all_models()
    print("\n✅ Prediction pipeline complete.")


if __name__ == "__main__":
    main()
