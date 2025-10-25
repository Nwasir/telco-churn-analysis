### interpretability.py (Updated for Stability)
SHAP-based explanations were removed due to instability and dtype conversion errors during development.
The script now uses a fallback importance analysis based on feature correlations and scaling.
This ensures stable deployment across environments without heavy dependencies.
