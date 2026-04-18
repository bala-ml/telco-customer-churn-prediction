import os
import logging
from pathlib import Path
import joblib
import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env content to env vars
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "model" / "cardio_prediction_model.joblib"

model = joblib.load(MODEL_PATH)

LOG_PATH = PROJECT_ROOT / "logs" / "app.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)

# load the trained model only once (module-level cache)
model = load(MODEL_PATH)
logging.info("Model loaded successfully.")


def predict(input_data: dict):

    df = pd.DataFrame([input_data])

    df = df.drop(columns=["customerID"], errors="ignore")
    df = pd.get_dummies(df, drop_first=True)

    model_features = model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    # get predicted class
    prediction = int(model.predict(df)[0])

    # get prediction probability
    probability = float(model.predict_proba(df)[0][1])
    prob_percent = round(probability * 100, 2)

    if prob_percent >= 60:
        risk_level = "High"
    elif prob_percent >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    diagnosis = (
        "Customer likely to churn" if prediction == 1 else "Customer likely to stay"
    )

    logging.info(
        f"Model provided a prediction: {prediction}, probability: {probability}"
    )

    return {
        "prediction": prediction,
        "probability": f"{prob_percent}%",
        "risk_level": risk_level,
        "diagnosis": diagnosis,
    }


# example usage
sample_input = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}
result = predict(input_data=sample_input)
print(result)
