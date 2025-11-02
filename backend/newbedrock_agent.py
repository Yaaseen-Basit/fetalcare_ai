import joblib
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from catboost import CatBoostClassifier
import boto3
import json
import concurrent.futures
import time

# ----------------------------
# AWS Bedrock Runtime Setup
# ----------------------------
BEDROCK_REGION = "ap-south-1"  # Change if needed
MODEL_ID = "deepseek.v3-v1:0"  # DeepSeek on-demand model

try:
    bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    logging.info(f"Bedrock runtime client initialized for region {BEDROCK_REGION}.")
except Exception as e:
    logging.error(f"Failed to initialize Boto3 Bedrock client: {e}")
    bedrock_runtime_client = None

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)

# ----------------------------
# CatBoost Model Loading
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fetalcare_model_catboost_safe.pkl')
FEATURE_COLUMNS = [
    'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

def load_model(path: str) -> Optional[CatBoostClassifier]:
    try:
        if not os.path.exists(path):
            logging.warning(f"Model file not found at {path}. Model will be unavailable.")
            return None
        model = joblib.load(path)
        logging.info(f"CatBoost model loaded successfully from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

MODEL = load_model(MODEL_PATH)

# ----------------------------
# Bedrock LLM Recommendation
# ----------------------------
# def generate_bedrock_advice(prediction: int, user_role: str, max_retries: int = 2) -> str:
#     class_mapping = {0: "Normal", 1: "Suspect", 2: "Pathologic"}
#     risk_label = class_mapping.get(prediction, "Unknown")

#     prompt = (
#         f"As a professional obstetrician, provide brief advice "
#         f"(under 3 sentences) for a {user_role}. "
#         f"Fetal risk status: {risk_label}. Be concise and professional."
#     )

#     fallback_advice = f"Fetal risk status: {risk_label}. Please consult your healthcare provider."

#     if bedrock_runtime_client is None:
#         logging.error("Bedrock client not initialized. Returning fallback advice.")
#         return fallback_advice

#     request_body = {
#         "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
#         "inference_config": {"temperature": 0.7, "max_tokens_to_sample": 150},
#         "additional_model_request_fields": {}
#     }

#     for attempt in range(max_retries + 1):
#         try:
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 future = executor.submit(
#                     bedrock_runtime_client.converse,
#                     modelId=MODEL_ID,
#                     contentType="application/json",
#                     accept="application/json",
#                     body=json.dumps(request_body)
#                 )
#                 response = future.result(timeout=30)

#             response_body = json.loads(response['body'].read())
#             if 'messages' in response_body and len(response_body['messages']) > 0:
#                 # Extract assistant's text
#                 return response_body['messages'][0]['content'][0]['text'].strip()

#         except Exception as e:
#             logging.warning(f"Attempt {attempt+1} - Bedrock call failed: {e}")
#             if attempt < max_retries:
#                 time.sleep(1)
#             else:
#                 logging.error("Max retries reached. Returning fallback advice.")
#                 return fallback_advice

#     return fallback_advice
def generate_bedrock_advice(prediction: int, user_role: str, max_retries: int = 2) -> str:
    class_mapping = {0: "Normal", 1: "Suspect", 2: "Pathologic"}
    risk_label = class_mapping.get(prediction, "Unknown")

    prompt = (
        f"As a professional obstetrician, provide brief advice "
        f"(under 3 sentences) for a {user_role}. "
        f"Fetal risk status: {risk_label}. Be concise and professional."
    )

    fallback_advice = f"Fetal risk status: {risk_label}. Please consult your healthcare provider."

    if bedrock_runtime_client is None:
        return fallback_advice

    for attempt in range(max_retries + 1):
        try:
            response = bedrock_runtime_client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.7, "maxTokens": 150}
            )
            
            # Extract response from converse API
            if 'output' in response and 'message' in response['output']:
                return response['output']['message']['content'][0]['text'].strip()

        except Exception as e:
            logging.warning(f"Attempt {attempt+1} - Bedrock call failed: {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                return fallback_advice

    return fallback_advice
# ----------------------------
# Prediction Logic
# ----------------------------
def get_prediction_result(user_role: str, features_dict: Dict[str, float]) -> Dict[str, Any]:
    if MODEL is None:
        logging.error("Model not available.")
        raise HTTPException(status_code=503, detail="Machine learning model not loaded.")

    logging.info(f"Received features for prediction.")

    feature_values = [features_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    prediction = -1
    confidence = 0.0

    # Clinical override: AC=0 or FM=0 → Pathologic (2)
    if features_dict.get('AC', 0) == 0 or features_dict.get('FM', 0) == 0:
        prediction = 2
        confidence = 0.99
        logging.info("Clinical override applied: AC=0 or FM=0")
    else:
        try:
            prediction = int(MODEL.predict(input_df)[0])
            confidence = float(MODEL.predict_proba(input_df)[0].max())
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    recommendation = generate_bedrock_advice(prediction, user_role)
    logging.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Role: {user_role}")

    return {
        "predicted_class": prediction,
        "recommendation": recommendation,
        "risk_score": f"{confidence:.4f}",
        "model_used": "CatBoost FetalCare Model + DeepSeek-V3.1 LLM Recommendation"
    }

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="FetalCare Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Features(BaseModel):
    LB: float; AC: float; FM: float; UC: float
    ASTV: float; MSTV: float; ALTV: float; MLTV: float
    DL: float; DS: float; DP: float; DR: float
    Width: float; Min: float; Max: float; Nmax: float; Nzeros: float
    Mode: float; Mean: float; Median: float; Variance: float; Tendency: float

class PredictionRequest(BaseModel):
    user_role: str
    features: Features

@app.post("/predict")
async def predict_risk(data: PredictionRequest, request: Request):
    logging.info(f"Incoming request from {request.client.host}: User role = {data.user_role}")
    try:
        result = get_prediction_result(data.user_role, data.features.model_dump())
        logging.info(f"Returning prediction result.")
        return result
    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unhandled exception in predict_risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))
