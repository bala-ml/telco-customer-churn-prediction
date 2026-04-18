import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.inference.predictor import predict

st.set_page_config(
    page_title="Customer Churn Predictor", page_icon="📊", layout="centered"
)

# Font
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
* {
    font-family: 'Poppins', sans-serif !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.title("📊 Customer Churn Predictor")
st.write("Predict whether a customer will churn or stay")

st.subheader("Enter customer details and click **Predict**")

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 100, 1)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    DeviceProtection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# More fields
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

# Predict Button
if st.button("🔍 Predict"):

    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    result = predict(input_data)

    prediction = result["prediction"]
    probability = result["probability"]
    risk_level = result["risk_level"]
    diagnosis = result["diagnosis"]

    st.divider()

    st.metric("Churn Probability", probability)
    if risk_level == "High":
        st.error(f"Risk Level: {risk_level}")
    elif risk_level == "Medium":
        st.warning(f"Risk Level: {risk_level}")
    else:
        st.success(f"Risk Level: {risk_level}")

    if prediction == 1:
        st.error(f"Model Prediction: {diagnosis}")
    else:
        st.success(f"Model Prediction: {diagnosis}")
