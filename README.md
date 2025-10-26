## 🧠 Customer Churn Prediction & Customer Lifetime Value (CLV) Analysis
🎯 Overview

This project predicts customer churn and estimates Customer Lifetime Value (CLV) for a telecom company using the IBM Telco Customer Churn dataset.
It demonstrates how data science can help identify high-risk customers and prioritize retention efforts based on value.

The project follows a full ML workflow — from data preparation to modeling, interpretability, and interactive visualization via a Streamlit web app.

###  🚀 Key Objectives

- Predict which customers are likely to churn.

- Estimate each customer’s CLV and segment them by value.

- Provide interpretable explanations for model predictions.

### 🏗️ Project Structure

telco-churn-analysis/
│
├── src/
│   ├── data_prep.py              # Cleans, engineers, and splits data
│   ├── train_models.py           # Trains Logistic, RF, XGB models
│   ├── predict.py                # Runs predictions on validation data
│   ├── interpretability.py       # Generates SHAP & feature importance plots
│   ├── clv_analysis.py           # CLV segmentation & churn behavior analysis
│
├── data/
│   ├── raw/                      # Original dataset (IBM Telco Churn)
│   └── processed/                # Cleaned & split data, prediction outputs
│
├── models/                       # Saved trained models (.pkl)
├── app.py                        # Streamlit app with 3 tabs (Predict, Performance, CLV)
├── tests/                        # Pytest scripts for reliability checks
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
|__ Video
|__ AI_USAGE.md

### 🧩 Workflow Summary

1️⃣ **Data Preparation**

Loaded & cleaned the IBM Telco Customer Churn dataset.

Handled missing values in TotalCharges.

Split data into Train (60%) / Validation (20%) / Test (20%) with stratification.

2️⃣ **CLV Analysis**

Segmented customers into quartiles: Low / Medium / High / Premium.

Computed churn rate by CLV segment.

Plotted:

CLV distribution (clv_distribution.png)

Churn rate by CLV (churn_by_clv.png)

Extracted business insights on customer value vs. churn behavior.

3️⃣ **Modeling**

Trained three models:

Logistic Regression (baseline)

Random Forest

XGBoost

Tuned 2–3 key hyperparameters/model.

Evaluated using:

Precision, Recall, F1, AUC-ROC

Persisted trained models and preprocessors to /models.

4️⃣ **Interpretability**

Coefficient-based analysis for Logistic Regression

Generated:

Global feature importances

Local explanations for individual predictions

5️⃣ **Streamlit Web App**

Tabs:

Predict

User inputs customer details

Displays churn probability + CLV estimate

Shows SHAP/local feature importance

Model Performance

Comparison of Precision, Recall, F1, AUC

ROC curve overlay & confusion matrix

CLV Overview

CLV distribution histogram

Churn rate by CLV quartile

Key insights & recommendations

### 📊 Example Insights

Low-CLV customers churn at the highest rate — often short-tenure or monthly contracts.

Premium customers rarely churn, offering the highest ROI.

Retention strategies should target medium-value customers with loyalty offers.

### 🧪 Testing

All source modules are tested using pytest for reliability.

**To run all tests:** pytest -v

**Run preprocessing, training, and predictions:**

python src/data_prep.py
python src/train_models.py
python src/predict.py
python src/clv_analysis.py

**Launch the Streamlit app:** streamlit run app.py

🧠 Tools & Libraries
|-----------|----------|
| Category | Tools Used |
|--------|-----------|
| Data Manipulation | pandas, numpy |
| Modeling | scikit-learn, xgboost |
| Visualization | matplotlib, seaborn
| Interpretability | feature importance |
| Web App | streamlit |
| Testing| pytest |
| ersistence |joblib |

### 💡 Business Takeaway

By integrating churn prediction with CLV segmentation, businesses can prioritize retention efforts where they matter most — focusing on high-value customers at risk of churn while designing affordable retention incentives for medium-tier segments.