# ---------------------------------------------------------
# FINAL Streamlit App for Customer Churn Prediction
# With Correct SHAP Using XGBoost Booster (pred_contribs)
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Global Yes/No mapping
binary = {"Yes": 1, "No": 0}


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()


def build_input_df(values):
    """Creates a single-row DataFrame."""
    return pd.DataFrame({k: [v] for k, v in values.items()})


def encode_and_align(df_row, full_cols):
    df_ohe = pd.get_dummies(df_row)
    df_ohe = df_ohe.reindex(columns=full_cols, fill_value=0)
    return df_ohe


def plot_shap_bar(shap_values_row, feature_names, top_n=12):
    """Horizontal bar plot for SHAP values."""
    shap_abs = np.abs(shap_values_row)
    ser = pd.Series(shap_abs, index=feature_names)
    ser = ser.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ser[::-1].plot.barh(ax=ax)
    ax.set_title("Top SHAP Feature Impacts")
    ax.set_xlabel("Absolute SHAP value")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Load model + columns
# ---------------------------------------------------------

@st.cache_resource
def load_assets():
    model = safe_load("xgb_churn_model.pkl")
    full_cols = safe_load("model_columns.pkl")

    if not hasattr(model, "predict_proba"):
        st.error("Loaded model is not a classifier.")
        st.stop()

    return model, full_cols


model, full_cols = load_assets()


# ---------------------------------------------------------
# UI — Input Form
# ---------------------------------------------------------

st.title("📉 Telecom Customer Churn Prediction")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=10)
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        mult_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col3:
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Credit card (automatic)", "Bank transfer (automatic)", "Mailed check"])

    submitted = st.form_submit_button("Predict")


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    # Build full input dictionary
    values = {
        "gender": gender,
        "MultipleLines": mult_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaymentMethod": payment,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "SeniorCitizen": binary[senior],
        "Partner": binary[partner],
        "Dependents": binary[dependents],
        "PhoneService": binary[phone],
        "PaperlessBilling": binary[paperless],
    }

    df_raw = build_input_df(values)
    df_model = encode_and_align(df_raw, full_cols)

    # Prediction
    prob = model.predict_proba(df_model)[0][1]
    pred = int(prob >= 0.30)

    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Churn Probability", f"{prob*100:.2f}%")
        st.write("Prediction:", "🔥 **Will Churn**" if pred == 1 else "🟢 **Will Stay**")

    # ---------------------------------------------------------
    # Correct SHAP via XGBoost Booster
    # ---------------------------------------------------------

    booster = model.get_booster()
    shap_vals = booster.predict(xgb.DMatrix(df_model), pred_contribs=True)

    # Remove last column (XGBoost bias term)
    shap_row = shap_vals[0][:-1]

    st.subheader("Local SHAP Feature Impact")
    fig = plot_shap_bar(shap_row, df_model.columns)
    st.pyplot(fig)
