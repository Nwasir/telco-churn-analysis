import streamlit as st
import pandas as pd
from pandas import Timestamp
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# 🔧 Paths
# =========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

# =========================================
# 🏷️ Page Setup
# =========================================
st.set_page_config(page_title="Customer Churn Analysis", layout="wide", page_icon="📊")
st.title("📊 Customer Churn Analysis Dashboard")
st.markdown("""
    Welcome to the Customer Churn Analysis Dashboard! This tool helps you:
    - 🔮 Predict customer churn probability
    - 📈 Analyze model performance metrics
    - 🧠 Understand feature importance
    
    Select a tab below to get started.
""")

# Style for the tabs
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Create tabs with icons
predict_tab, performance_tab, importance_tab = st.tabs([
    "🔮 Predict",
    "📈 Model Performance",
    "🧠 CLV_Overview"
])

# =========================================
# 🧩 PAGE 1: Model Performance
# =========================================
with performance_tab:
    st.header("📈 Model Evaluation Results")

    metrics_path = DATA_DIR / "metrics_summary.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, index_col=0)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))

        best_model_name = metrics_df["AUC"].idxmax()
        st.success(f"🏆 Best Model: **{best_model_name.upper()}** (AUC = {metrics_df.loc[best_model_name, 'AUC']:.3f})")

    else:
        st.warning("No model metrics found. Run `evaluate.py` first.")

    # Confusion Matrix Section with Model Selector
    st.subheader("🧩 Confusion Matrix Viewer")

    # Dropdown to choose which model’s confusion matrix to display
    model_choice = st.selectbox(
        "Select Model to View Confusion Matrix",
        options=["Logistic Regression", "Random Forest", "XGBoost"],
        key="conf_matrix_selector"
    )

    # Map dropdown text to filenames
    model_map = {
        "Logistic Regression": "logistic",
        "Random Forest": "rf",
        "XGBoost": "xgb"
    }

    selected_model_key = model_map[model_choice]
    confusion_path = REPORTS_DIR / f"confusion_matrix_{selected_model_key}.png"

    if confusion_path.exists():
        st.image(
            str(confusion_path),
            caption=f"{model_choice} Confusion Matrix",
            width=600,
            output_format="auto"
        )
    else:
        st.warning(f"Confusion matrix not found for {model_choice}. Expected file: {confusion_path.name}")


    # ROC Curves
    roc_path = REPORTS_DIR / "roc_curves.png"
    if roc_path.exists():
        st.subheader("ROC Curve Comparison")
        st.image(str(roc_path), caption="ROC Curve Comparison",
                  width=600, output_format="auto")
    else:
        st.warning("ROC curve plot not found.")

    # =========================================
    # 🌟 Feature Importance Section
    # =========================================
    st.subheader("🌟 Feature Importance Viewer")

    importance_option = st.selectbox(
        "Select Model to View Feature Importance",
        options=["Logistic Regression", "Random Forest", "XGBoost"],
        key="feature_importance_selector"
    )

    # Map to expected filenames
    importance_files = {
        "Logistic Regression": REPORTS_DIR / "lg_importance.png",
        "Random Forest": REPORTS_DIR / "rf_importance.png",
        "XGBoost": REPORTS_DIR / "xgb_importance.png"
    }

    selected_importance_path = importance_files[importance_option]

    if selected_importance_path.exists():
        st.image(str(selected_importance_path), 
                 caption=f"{importance_option} Feature Importance", 
                 width=800, output_format="auto"
        )

    else:
        st.warning(f"Feature importance plot not found for {importance_option}. Expected file: `{selected_importance_path.name}`")

# =========================================
# 🧠 CLV Overview and Churn Insights
# =========================================
with importance_tab:
    st.subheader("💰 CLV Overview and Churn Insights")

    clv_dist_path = DATA_DIR / "clv_distribution.png"
    churn_by_clv_path = DATA_DIR / "churn_by_clv.png"

    col1, col2 = st.columns(2)

    if clv_dist_path.exists():
        col1.image(str(clv_dist_path), caption="Customer Lifetime Value (CLV) Distribution")
    else:
        col1.warning("Missing CLV distribution plot. Run `src/clv_analysis.py` first.")

    if churn_by_clv_path.exists():
        col2.image(str(churn_by_clv_path), caption="Churn Rate by CLV Segment")
    else:
        col2.warning("Missing churn-by-CLV plot. Run `src/clv_analysis.py` first.")

    # Show churn rate summary if available
    churn_summary_path = DATA_DIR / "train.csv"
    if churn_summary_path.exists():
        try:
            df_train = pd.read_csv(churn_summary_path)
            if "CLV" in df_train.columns:
                df_train["CLV_quartile"] = pd.qcut(df_train["CLV"], q=4, labels=["Low", "Medium", "High", "Premium"])
                churn_summary = (
                    df_train.groupby("CLV_quartile", observed=False)["Churn"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Churn": "ChurnRate"})
                )
                churn_summary["ChurnRate"] = churn_summary["ChurnRate"] * 100
                st.markdown("### 📊 Churn Rate by CLV Segment")
                st.dataframe(churn_summary.style.format({"ChurnRate": "{:.2f}%"}))
            else:
                st.info("CLV column not found in dataset. Please rerun CLV analysis.")
        except Exception as e:
            st.error(f"Error loading churn summary: {e}")
    else:
        st.info("Training data not found for CLV summary.")

    # Business insights
    st.markdown("### 💡 Key Business Insights")
    st.markdown("""
    - Low-CLV customers typically have **shorter tenure or monthly contracts**, leading to higher churn.
    - Premium-CLV customers show **lower churn rates**, providing **higher ROI** if retained.
    - Target **Medium-CLV users** with loyalty programs or bundle offers to **boost retention**.
    """)

# =========================================
# 🔮 Predict Single Customer
# =========================================
with predict_tab:
    st.header("🔮 Predict Customer Churn")

    # Initialize model variables
    models_info = {
        "Logistic Regression": {"file": "logistic.pkl", "desc": "Simple, interpretable model good for understanding feature importance"},
        "Random Forest": {"file": "rf.pkl", "desc": "Ensemble model that handles non-linear relationships well"},
        "XGBoost": {"file": "xgb.pkl", "desc": "Advanced gradient boosting model with high predictive power"}
    }
    
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"

    # ---- Load preprocessor if available ----
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

    # ---- Input Sections (Demographics, Services, Billing) ----
    col_dem, col_services, col_billing = st.columns(3, gap="large")

    # Demographics
    with col_dem:
        st.markdown('<div class="section-title card-style" style="color:#4b2a7a;">👤  Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=1, key="tenure")
        st.markdown('</div>', unsafe_allow_html=True)

    # Tenure bucket
    if tenure <= 6: tenure_bucket = "0-6m"
    elif tenure <= 12: tenure_bucket = "6-12m"
    elif tenure <= 24: tenure_bucket = "12-24m"
    else: tenure_bucket = "24m+"

    # Services
    with col_services:        
        st.markdown('<div class="section-title card-style" style="color:#4b2a7a;">🛠️  Services</div>', unsafe_allow_html=True)
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multiple_lines")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="online_security")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="online_backup")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device_protection")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_support")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies")
        
        # Derived service counts
        service_cols = [phone, multiple_lines, internet, online_security,
                        online_backup, device_protection, tech_support,
                        streaming_tv, streaming_movies]
        services_count = sum(1 for s in service_cols if s == "Yes")
        internet_no_techsupport = 1 if (internet in ["DSL", "Fiber optic"] and tech_support == "No") else 0
        st.markdown('</div>', unsafe_allow_html=True)

    # Billing
    with col_billing:        
        st.markdown('<div class="section-title card-style" style="color:#4b2a7a;">💳  Contract & Payment</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment_method"
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0, step=5.0, key="monthly")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * monthly_charges, step=100.0, key="total")
        
        # CLV calculation
        expected_tenure = 37.57
        clv = monthly_charges * expected_tenure
        ratio = total_charges / max(1, tenure * monthly_charges)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Model Selection ----
    st.markdown("""
        <style>
        .model-header-container {
            display: flex;
            justify-content: center;
            width: 100%;
            margin: 2rem 0;
        }
        .model-header {
            text-align: center;
            color: #4b2a7a;
            font-size: 24px;
            font-weight: 600;
            margin: 0;
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="model-header-container">
            <h3 class="model-header">🤖 Model Selection</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        selected_model = st.selectbox("Select Prediction Model", list(models_info.keys()), key="model_selector")
        st.markdown(f'<div class="model-description">{models_info[selected_model]["desc"]}</div>', unsafe_allow_html=True)

    model_path = MODEL_DIR / models_info[selected_model]["file"]
    if not model_path.exists():
        st.error(f"⚠️ {selected_model} model file not found. Please train the model first.")
        st.stop()
    model = joblib.load(model_path)

    
    threshold = 0.5   

    # ---- Predict Button ----
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_clicked = st.button("🔮 Predict", key="predict_button")
        if predict_clicked:
            # Build input_df
            dummy_id = "PRED-" + pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            input_df = pd.DataFrame([{
                "customerID": dummy_id,
                "gender": gender,
                "SeniorCitizen": 1 if str(senior).lower() in ["1","yes","true"] else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "tenure_bucket": tenure_bucket,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "services_count": services_count,
                "monthly_to_total_ratio": ratio,
                "internet_no_techsupport": internet_no_techsupport,
                "ExpectedTenure": expected_tenure,
                "CLV": clv
            }])
            
            X_processed = preprocessor.transform(input_df) if preprocessor else input_df
            prob = model.predict_proba(X_processed)[0][1]
            prediction = 1 if prob >= threshold else 0
            confidence = prob if prediction == 1 else (1 - prob)
            risk_label = "HIGH" if prob >= threshold else "LOW"
            clv_label = "Below Average" if clv < 3000 else "Average" if clv < 5000 else "Above Average"
            conf_label = "High" if confidence >= 0.9 else "Moderate"

            # ---- Styled Result Cards ----
            st.markdown("""
            <style>
            h3.centered-header { 
                text-align: center !important;
                width: 100%;
                display: block;
                margin: 1em auto;
                font-size: 1.5em;
                font-weight: 600;
            }
            .metric-card { border-radius: 12px; padding: 25px; text-align: center; color: white; font-weight: 600; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
            .high-risk { background-color: #e74c3c; }
            .clv-card { background-color: #d98c2c; }
            .confidence-card { background-color: #27ae60; }
            .metric-value { font-size: 2em; margin: 10px 0; }
            .metric-sub { font-size: 0.9em; opacity: 0.9; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Prediction Results")
            st.markdown("#### Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card high-risk">
                    <div>HIGH</div>
                    <div>Churn Probability</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div class="metric-sub">{risk_label} RISK</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card clv-card">
                    <div>$</div>
                    <div>Customer Lifetime Value</div>
                    <div class="metric-value">${clv:,.0f}</div>
                    <div class="metric-sub">{clv_label}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card confidence-card">
                    <div>HIGH</div>
                    <div>Prediction Confidence</div>
                    <div class="metric-value">{conf_label}</div>
                    <div class="metric-sub">{confidence*100:.0f}% Certain</div>
                </div>""", unsafe_allow_html=True)

            # ---- Business Insights ----
            st.markdown("### 💡 Business Insights")
            if prediction == 1:
                st.markdown(f"""
                - The model predicts **high likelihood of churn ({prob*100:.1f}%)**.
                - Customer’s CLV is **${clv:,.0f} ({clv_label})**, meaning potential revenue loss if churned.
                - Recommend targeting with **retention incentives**.
                - Focus on **service quality and contract engagement** to reduce churn risk.
                """)
            else:
                st.markdown(f"""
                - The customer is **unlikely to churn** (churn probability {prob*100:.1f}%).
                - CLV of **${clv:,.0f} ({clv_label})** indicates a valuable and stable customer.
                - Maintain retention through **consistent communication**.
                - Consider **upselling or cross-selling** premium plans.
                """)


