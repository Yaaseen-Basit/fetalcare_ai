import joblib
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from catboost import CatBoostClassifier

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)

# ----------------------------
# Model Loading
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fetalcare_model_catboost_safe.pkl')

FEATURE_COLUMNS = [
    'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

def load_model(path: str) -> Optional[CatBoostClassifier]:
    try:
        model = joblib.load(path)
        logging.info(f"CatBoost model loaded successfully from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

MODEL = load_model(MODEL_PATH)

# ----------------------------
# Recommendation Logic
# ----------------------------
def generate_recommendation(prediction: int, role: str, confidence: float) -> str:
    conf_str = f"{confidence:.2f}"

    # Clinical override: any AC=0 or FM=0 triggers at least Suspect
    if prediction == 0:  # originally Normal
        # This is handled in get_prediction_result

        if role == 'Doctor':
            return f"Fetal status: Normal. Continue routine monitoring. Confidence: {conf_str}"
        if role == 'Nurse':
            return "Fetal status is normal. Proceed with routine care."
        if role == 'Expectant Mother':
            return "Fetal status is normal. Continue regular prenatal care."

    if prediction == 1:
        if role == 'Doctor':
            return f"Fetal status: Suspect. Continuous monitoring recommended. Confidence: {conf_str}"
        if role == 'Nurse':
            return "Suspect FHR detected. Increase monitoring and notify physician."
        if role == 'Expectant Mother':
            return "Variation detected. Contact doctor for follow-up."

    if prediction == 2:
        if role == 'Doctor':
            return f"Fetal status: Pathologic. Immediate evaluation required. Confidence: {conf_str}"
        if role == 'Nurse':
            return "Pathologic FHR detected. Notify physician and prepare emergency protocols."
        if role == 'Expectant Mother':
            return "Critical fetal variation detected. Contact healthcare provider immediately."

    return "Error: Could not generate recommendation."

# ----------------------------
# Prediction Function
# ----------------------------
def get_prediction_result(user_role: str, features_dict: Dict[str, float]) -> Dict[str, Any]:
    if MODEL is None:
        logging.error("Model is not available.")
        raise HTTPException(status_code=503, detail="Model not available.")

    logging.info(f"Received features for prediction: {features_dict}")

    # Ensure feature order
    feature_values = [features_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    # ----------------------------
    # Clinical overrides
    # ----------------------------
    if features_dict.get('AC', 0) == 0 or features_dict.get('FM', 0) == 0:
        prediction = 2  # Force Pathologic if both AC or FM are 0
        confidence = 0.99  # High confidence for clinical override
        logging.info("Clinical override applied: AC=0 or FM=0")
    else:
        try:
            prediction = int(MODEL.predict(input_df)[0])
            confidence = float(MODEL.predict_proba(input_df)[0].max())
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    recommendation = generate_recommendation(prediction, user_role, confidence)
    logging.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Role: {user_role}")

    return {
        "predicted_class": prediction,
        "recommendation": recommendation,
        "risk_score": f"{confidence:.4f}",
        "model_used": "CatBoost FetalCare Model with Clinical Overrides"
    }

# ----------------------------
# FastAPI App Setup
# ----------------------------
app = FastAPI(title="FetalCare Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Features(BaseModel):
    LB: float
    AC: float
    FM: float
    UC: float
    ASTV: float
    MSTV: float
    ALTV: float
    MLTV: float
    DL: float
    DS: float
    DP: float
    DR: float
    Width: float
    Min: float
    Max: float
    Nmax: float
    Nzeros: float
    Mode: float
    Mean: float
    Median: float
    Variance: float
    Tendency: float

class PredictionRequest(BaseModel):
    user_role: str
    features: Features

@app.post("/predict")
async def predict_risk(data: PredictionRequest, request: Request):
    logging.info(f"Incoming request from {request.client.host}: User role = {data.user_role}")
    try:
        result = get_prediction_result(data.user_role, data.features.dict())
        logging.info(f"Returning prediction result: {result}")
        return result
    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
