"""
Customer Churn Predictor — Streamlit App
----------------------------------------
This app lets a business user (e.g. a CSM or retention analyst) enter a
customer's profile and instantly see:
  1. Whether that customer is at risk of churning
  2. Which specific factors are driving that risk

Product rationale: the goal isn't just a prediction — it's an actionable
signal. A CSM shouldn't need to read a data science report; they should be
able to act on a customer record they already have open.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Page config — set before any other Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load or train model artefacts
# ---------------------------------------------------------------------------
MODEL_PATH = "model_artefacts.pkl"


@st.cache_resource
def load_artefacts():
    """
    Load pre-trained model artefacts saved by the notebook.
    If they don't exist yet, show a clear message rather than crashing.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artefacts = load_artefacts()

# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.markdown(
    """
    Enter a customer's details below to see their **churn risk** and the key
    factors driving it. Designed to help retention teams prioritise outreach.
    """
)

if artefacts is None:
    st.warning(
        "⚠️ Model artefacts not found. Please run the Jupyter notebook first "
        "(`churn_analysis.ipynb`) — it will save `model_artefacts.pkl` in this "
        "directory."
    )
    st.stop()

model: LogisticRegression = artefacts["model"]
scaler: StandardScaler = artefacts["scaler"]
feature_names: list = artefacts["feature_names"]
explainer = artefacts["explainer"]

# ---------------------------------------------------------------------------
# UI — Input form
# ---------------------------------------------------------------------------
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "Tenure (months)",
        min_value=0, max_value=72, value=12,
        help="How long the customer has been with the company.",
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0, max_value=200.0, value=65.0, step=1.0,
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges),
        step=10.0,
    )
    contract = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        help="Month-to-month customers churn at much higher rates.",
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"],
    )
    payment_method = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"])

st.markdown("---")
st.markdown("**Add-on Services**")

col3, col4, col5 = st.columns(3)
with col3:
    phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", options=["No", "Yes", "No phone service"]
    )
with col4:
    online_security = st.selectbox(
        "Online Security", options=["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", options=["No", "Yes", "No internet service"]
    )
with col5:
    tech_support = st.selectbox(
        "Tech Support", options=["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", options=["No", "Yes", "No internet service"]
    )

# ---------------------------------------------------------------------------
# Feature engineering — must exactly mirror what the notebook produces
# ---------------------------------------------------------------------------

def build_feature_row() -> pd.DataFrame:
    """
    Translate raw UI inputs into the same engineered feature vector the model
    was trained on. Keeping this in one place makes it easier to audit parity
    with the notebook's preprocessing.
    """
    row = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        # One-hot: Contract (Month-to-month is the reference/dropped category)
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        # One-hot: InternetService (DSL is reference)
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        # One-hot: PaymentMethod (Bank transfer is reference)
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        # Binary flags
        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
    }
    df = pd.DataFrame([row])
    # Ensure column order matches training exactly
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("🔍 Predict Churn Risk", type="primary"):
    input_df = build_feature_row()
    input_scaled = scaler.transform(input_df)

    churn_prob = model.predict_proba(input_scaled)[0][1]
    churn_pred = churn_prob >= 0.5

    # ---- Risk badge ----
    if churn_prob >= 0.7:
        risk_label, risk_color = "HIGH RISK", "🔴"
    elif churn_prob >= 0.4:
        risk_label, risk_color = "MEDIUM RISK", "🟡"
    else:
        risk_label, risk_color = "LOW RISK", "🟢"

    st.markdown("---")
    st.subheader("Prediction Result")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Churn Probability", f"{churn_prob:.1%}")
    metric_col2.metric("Risk Level", f"{risk_color} {risk_label}")

    # ---- SHAP waterfall plot — shows WHY the model made this call ----
    st.subheader("What's Driving This Prediction?")
    st.markdown(
        "Each bar shows how much a feature *pushes the risk up (red) or down "
        "(blue)* from the baseline churn rate across all customers."
    )

    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ---- Plain-language summary for non-technical users ----
    st.subheader("Plain-English Summary")
    sv = shap_values[0].values
    fn = feature_names

    # Top 3 risk drivers
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    top_features = [(fn[i], sv[i]) for i in sorted_idx[:3]]

    summary_lines = []
    for feat, val in top_features:
        direction = "increasing" if val > 0 else "reducing"
        summary_lines.append(f"- **{feat}** is {direction} churn risk")

    st.markdown("\n".join(summary_lines))

    if churn_pred:
        st.info(
            "💡 **Recommended action:** This customer shows elevated churn signals. "
            "Consider a proactive retention offer — a contract upgrade incentive or "
            "a personal check-in from their account manager."
        )
    else:
        st.success(
            "✅ This customer appears stable. Standard engagement cadence is appropriate."
        )
