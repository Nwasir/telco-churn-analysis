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
    "🧠 Feature Importance"
])

# =========================================
# 🧩 PAGE 1: Model Performance
# =========================================
with performance_tab:
    st.header("📈 Model Evaluation Results")

    metrics_path = DATA_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, index_col=0)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))

        best_model_name = metrics_df["ROC-AUC"].idxmax()
        st.success(f"🏆 Best Model: **{best_model_name.upper()}** (ROC-AUC = {metrics_df.loc[best_model_name, 'ROC-AUC']:.3f})")

    else:
        st.warning("No model metrics found. Run `evaluate.py` first.")

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    models = ["logistic", "rf", "xgb"]
    for col, model in zip(cols, models):
        path = REPORTS_DIR / f"confusion_matrix_{model}.png"
        if path.exists():
            col.image(str(path), caption=f"{model.upper()} Confusion Matrix")
        else:
            col.warning(f"Missing: {path.name}")

    # ROC Curves
    roc_path = REPORTS_DIR / "roc_curves.png"
    if roc_path.exists():
        st.subheader("ROC Curve Comparison")
        st.image(str(roc_path))
    else:
        st.warning("ROC curve plot not found.")

# =========================================
# 🧠 PAGE 2: Feature Importance
# =========================================
with importance_tab:
    st.header("🧠 Model Interpretability")

    importance_path = REPORTS_DIR / "raw_feature_importance.png"
    shap_path = REPORTS_DIR / "rf_shap_summary.png"

    col1, col2 = st.columns(2)

    if importance_path.exists():
        col1.image(str(importance_path), caption="Raw Feature Importance")
    else:
        col1.warning("Missing raw feature importance plot.")

    if shap_path.exists():
        col2.image(str(shap_path), caption="SHAP Summary Plot")
    else:
        col2.warning("Missing SHAP plot.")

# =========================================
# 🔮 PAGE 3: Predict Single Customer
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
    preprocessor = None
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)

    # Add CSS for form sections (styling for column layout)
    st.markdown("""
        <style>
        .stApp .form-columns .css-1lcbmhc { padding: 0; }
        .section-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-title .icon {
            font-size: 20px;
        }
        .card-style {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px 22px;
            border: 1px solid #eef2f6;
            box-shadow: 0 4px 10px rgba(16,24,40,0.04);
        }
        /* style the built-in Streamlit selectboxes/inputs to look like cards */
        .stSelectbox > div[data-baseweb="select"] {
            background-color: #f3f6f8 !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            height: 44px !important;
        }
        .stNumberInput > div > div {
            background-color: #f3f6f8 !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            height: 44px !important;
        }
        .stSlider > div div[role="slider"] {
            margin-top: 6px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Use three Streamlit columns so widgets render inside them (widgets must be created inside columns)
    col_dem, col_services, col_billing = st.columns(3, gap="large")

    # Demographics column (matches screenshot: 'Demographics')
    with col_dem:
        st.markdown('<div class="section-title card-style" style="color:#4b2a7a;">👤  Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=1, key="tenure")
        st.markdown('</div>', unsafe_allow_html=True)

    # Compute tenure bucket from tenure input (used downstream)
    if tenure <= 6:
        tenure_bucket = "0-6m"
    elif tenure <= 12:
        tenure_bucket = "6-12m"
    elif tenure <= 24:
        tenure_bucket = "12-24m"
    else:
        tenure_bucket = "24m+"

    # Services column
    with col_services:        
        st.markdown('<div class="section-title card-style" style="color:#4b2a7a;">�  Services</div>', unsafe_allow_html=True)
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multiple_lines")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="online_security")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="online_backup")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device_protection")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_support")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies")
        st.markdown('</div>', unsafe_allow_html=True)

    # Billing column (Contract & Payment)
    with col_billing:        
        st.markdown('<div class="section-title card-style" style="color:#4b2a7a;">💳  Contact & Payment</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment_method"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Calculate base price guidance
    base_price = 20.0  # Base connection fee
    if phone == "Yes":
        base_price += 20.0
        if multiple_lines == "Yes":
            base_price += 10.0
    if internet == "DSL":
        base_price += 30.0
    elif internet == "Fiber optic":
        base_price += 50.0
    
    # Add service prices
    service_price = sum(10.0 for s in [online_security, online_backup, device_protection,
                                     tech_support, streaming_tv, streaming_movies] if s == "Yes")
    recommended_price = base_price + service_price

    # Now render monthly and total charges inside the billing column so help text uses computed recommended_price
    with col_billing:
        # Add some spacing inside the billing card
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            help=f"Recommended price range: ${recommended_price:.2f} - ${recommended_price * 1.2:.2f}",
            key="monthly_charges"
        )
        # Calculate expected total charges range
        min_expected = monthly_charges * (tenure * 0.9)  # Allow 10% lower for promotions
        max_expected = monthly_charges * (tenure * 1.1)  # Allow 10% higher for fees
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            step=10.0,
            help=f"Expected range for {tenure} months: ${min_expected:.2f} - ${max_expected:.2f}",
            key="total_charges"
        )

    # Close Services section
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Close Services section (HTML wrapper no longer used for widgets)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Add separator and styling for model selection
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Add custom CSS for centering and styling
    st.markdown("""
        <style>
        .model-section {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        .centered-header {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 20px;
        }
        .model-description {
            text-align: center;
            padding: 15px 20px;
            margin: 20px auto;
            max-width: 600px;
            border-radius: 5px;
            background-color: #f8f9fa;
            border: 1px solid #e1e4e8;
        }
        .stButton > button {
            margin: 30px auto;
            display: block;
            padding: 12px 50px;
            min-width: 300px;
            border-radius: 25px;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
        }
        .prediction-container {
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
        
    # Model Selection Section with container
    st.markdown('<div class="model-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="centered-header">🤖 Model Selection</h3>', unsafe_allow_html=True)
    
    # Center the model selection dropdown
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_model = st.selectbox(
            "Select Prediction Model",
            list(models_info.keys()),
            key="model_selector"
        )
        
        # Show model description in centered info box
        st.markdown(f'<div class="model-description">{models_info[selected_model]["desc"]}</div>', 
                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Set model path based on selection
    model_path = MODEL_DIR / models_info[selected_model]["file"]
    
    # Check if model exists
    if not model_path.exists():
        st.error(f"⚠️ {selected_model} model file not found. Please train the model first.")
        st.stop()
    
    # Load the selected model
    model = joblib.load(model_path)
    
    # Centered predict button with more space around it
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("🔮 Predict", key="predict_button", use_container_width=True)
    
    if predict_clicked:
        # Count total services
        service_cols = [phone, multiple_lines, internet, online_security,
                      online_backup, device_protection, tech_support,
                      streaming_tv, streaming_movies]
        services_count = sum(1 for s in service_cols if s == "Yes")
        
        # Calculate monthly_to_total_ratio
        ratio = total_charges / max(1, tenure * monthly_charges)
        
        # Determine internet_no_techsupport flag
        internet_no_techsupport = 1 if (internet != "No" and tech_support == "No") else 0
        
        # Use the same expected tenure as in training
        expected_tenure = 37.57  # from data_prep.py
        clv = monthly_charges * expected_tenure
    
        # Create a dummy customerID for prediction
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
    
        if preprocessor:
            try:
                X_processed = preprocessor.transform(input_df)
            except Exception:
                X_processed = input_df
        else:
            X_processed = input_df
    
        prediction = model.predict(X_processed)[0]
        prob = model.predict_proba(X_processed)[0][1]
    
        # Create prediction results container
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        st.markdown('<h3>Prediction Result</h3>', unsafe_allow_html=True)
        
        # Style the prediction box
        st.markdown("""
            <style>
            .result-box {
                background-color: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin: 20px auto;
                max-width: 500px;
                text-align: center;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
            .result-title {
                font-size: 1.2em;
                margin-bottom: 20px;
                color: #333;
            }
            .probability-bar {
                margin: 15px 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Display results in a centered box
        # st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title result-box">Model Used: {selected_model}</div>', unsafe_allow_html=True)
        st.write("**Churn Prediction:**", "Yes 🟥" if prediction == 1 else "No 🟩")
        st.write("**Confidence Score:**")
        
        # Create a progress bar for the probability
        with st.container():
            if prediction == 1:
                st.progress(prob)
            else:
                st.progress(1 - prob)
            st.write(f"**Churn Probability:** {prob:.2%}")
        
        st.markdown('</div></div>', unsafe_allow_html=True)
