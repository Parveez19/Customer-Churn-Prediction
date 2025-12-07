import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Yes/No mapping
binary = {"Yes": 1, "No": 0}

# Keep normal threshold
THRESHOLD = 0.30

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()


def build_input_df(values):
    return pd.DataFrame({k: [v] for k, v in values.items()})


def encode_and_align(df_row, full_cols):
    df_ohe = pd.get_dummies(df_row)
    df_ohe = df_ohe.reindex(columns=full_cols, fill_value=0)
    return df_ohe


def plot_shap_bar(shap_values_row, feature_names, top_n=12):
    shap_abs = np.abs(shap_values_row)
    ser = pd.Series(shap_abs, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ser[::-1].plot.barh(ax=ax)
    ax.set_title("Top SHAP Feature Impacts")
    ax.set_xlabel("Absolute SHAP Value")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Load assets
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
# UI
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
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        mult_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

    with col3:
        device_prot = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
                "Mailed check",
            ],
        )

    submitted = st.form_submit_button("Predict")


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    # Build model input
    values = {
        "gender": gender,
        "SeniorCitizen": binary[senior],
        "Partner": binary[partner],
        "Dependents": binary[dependents],
        "PhoneService": binary[phone],
        "PaperlessBilling": binary[paperless],
        "tenure": tenure,
        "MonthlyCharges": monthly,
        # REMOVE WRONG TotalCharges CALCULATION
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
    }

    df_raw = build_input_df(values)
    df_model = encode_and_align(df_raw, full_cols)

    # Predict
    prob = model.predict_proba(df_model)[0][1]
    pred = int(prob >= THRESHOLD)

    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Churn Probability", f"{prob * 100:.2f}%")
        st.write("Prediction:", "🔥 **Will Churn**" if pred == 1 else "🟢 **Will Stay**")

    # SHAP
    booster = model.get_booster()
    shap_vals = booster.predict(xgb.DMatrix(df_model), pred_contribs=True)
    shap_row = shap_vals[0][:-1]

    st.subheader("Local SHAP Feature Impact")
    fig = plot_shap_bar(shap_row, df_model.columns)
    st.pyplot(fig)
