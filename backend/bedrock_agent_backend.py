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
import time

# ----------------------------
# AWS Bedrock Runtime Setup
# ----------------------------
BEDROCK_REGION = "ap-south-1"  # Change if needed
bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

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
        model = joblib.load(path)
        logging.info(f"CatBoost model loaded successfully from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

MODEL = load_model(MODEL_PATH)

def generate_bedrock_advice(prediction: int, user_role: str) -> str:
    class_mapping = {0: "Normal", 1: "Suspect", 2: "Pathologic"}
    risk_label = class_mapping.get(prediction, "Unknown")
    model_id = "deepseek.v3-v1:0" # selected model

    system_prompt = "You are a professional obstetrician. Generate a short, professional guideline-based advice."
    user_prompt = f"A patient has a fetal risk status of '{risk_label}'. Generate a role-specific advice for a {user_role}."

    # --- CORRECTED PAYLOAD STRUCTURE (Messages API) ---
    # DeepSeek on Bedrock requires a 'messages' array for chat-like interactions,
    # and includes configuration under 'max_tokens', 'temperature', etc.
    request_body = {
        "system": system_prompt, # Use a dedicated system prompt key
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ],
        "max_tokens": 512,  # Set a reasonable limit
        "temperature": 0.7,
        "top_p": 0.9,       # Recommended inference parameter
        "stop_sequences": ["\n\n"]
    }
    # --- END CORRECTED PAYLOAD ---

    try:
        for attempt in range(3):
            try:
                response = bedrock_runtime_client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)  # Use the corrected request_body
                )
                
                output_bytes = response['body'].read()
                output_json = json.loads(output_bytes)
                
                # --- CORRECTED RESPONSE PARSING ---
                # For the Messages API format (like Claude/DeepSeek), the response is a 'message' object
                # where the text is in the 'content' array.
                if 'content' in output_json and output_json['content']:
                    # Extract the first text block from the content list
                    for content_block in output_json['content']:
                        if content_block['type'] == 'text':
                            return content_block['text'].strip()
                    
                # Fallback check if the model returns a simpler structure (less likely but good to have)
                elif 'completion' in output_json:
                    return output_json['completion'].strip()
                
                logging.warning(f"Response parsing failed for {model_id}: {output_json}")
                time.sleep(1) # Go to next attempt

            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed for {model_id}: {e}")
                time.sleep(1)

        logging.error(f"Bedrock LLM failed after retries for {model_id}")
        return f"Fetal risk status: {risk_label}. Please rely on standard clinical protocols."

    except Exception as e:
        logging.error(f"Unexpected Bedrock error for {model_id}: {e}")
        return f"Fetal risk status: {risk_label}. Recommendation generation failed."
# ----------------------------
# Prediction Logic
# ----------------------------
def get_prediction_result(user_role: str, features_dict: Dict[str, float]) -> Dict[str, Any]:
    if MODEL is None:
        logging.error("Model not available.")
        raise HTTPException(status_code=503, detail="Model not available.")

    logging.info(f"Received features for prediction: {features_dict}")

    feature_values = [features_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    # Clinical override: AC=0 or FM=0 → Pathologic
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
            raise HTTPException(status_code=500, detail=str(e))

    # Get Bedrock LLM recommendation
    recommendation = generate_bedrock_advice(prediction, user_role)
    logging.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Role: {user_role}")

    return {
        "predicted_class": prediction,
        "recommendation": recommendation,
        "risk_score": f"{confidence:.4f}",
        "model_used": "CatBoost FetalCare Model + Bedrock LLM Recommendation"
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
        result = get_prediction_result(data.user_role, data.features.dict())
        logging.info(f"Returning prediction result: {result}")
        return result
    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
