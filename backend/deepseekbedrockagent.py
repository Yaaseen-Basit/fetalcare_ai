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

# --- Constants ---
BEDROCK_REGION = "ap-south-1"
CLINICAL_OVERRIDE_PREDICTION = 2
CLINICAL_OVERRIDE_CONFIDENCE = 0.99
FALLBACK_ADVICE = "Fetal risk status: {risk_label}. Please consult your healthcare provider."

# ----------------------------
# AWS Bedrock Runtime Setup
# ----------------------------
bedrock_runtime_client = None
bedrock_client = None
MODEL_ID = None  # Initialize MODEL_ID to prevent NameError

try:
    bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    bedrock_client = boto3.client("bedrock", region_name=BEDROCK_REGION)
    logging.info(f"Bedrock clients initialized successfully for region {BEDROCK_REGION}")
except Exception as e:
    logging.error(f"Failed to initialize Boto3 Bedrock clients: {e}")

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
    """Loads the machine learning model from a given path."""
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
# Bedrock LLM Recommendation Helpers
# ----------------------------
def get_available_bedrock_models():
    """Retrieves a list of available Bedrock models in the configured region."""
    if not bedrock_client:
        return []
    try:
        response = bedrock_client.list_foundation_models()
        models = response.get('modelSummaries', [])
        available_models = []
        
        for model in models:
            model_info = {
                'modelId': model['modelId'],
                'modelName': model['modelName'],
                'provider': model.get('providerName', 'Unknown'),
                'inputModalities': model.get('inputModalities', []),
                'outputModalities': model.get('outputModalities', []),
                'customizationsSupported': model.get('customizationsSupported', [])
            }
            available_models.append(model_info)
        
        return available_models
    except Exception as e:
        logging.error(f"Failed to retrieve available models from Bedrock: {e}")
        return []

def find_suitable_model():
    """Identifies a suitable text generation model from available Bedrock models."""
    available_models = get_available_bedrock_models()
    
    preferred_providers = ['DeepSeek', 'Anthropic', 'Amazon', 'Meta', 'AI21 Labs', 'Cohere']
    
    for provider in preferred_providers:
        for model in available_models:
            if (provider.lower() in model['provider'].lower() and 
                'TEXT' in model['inputModalities'] and
                'TEXT' in model['outputModalities']):
                logging.info(f"Selected model: {model['modelId']} from {model['provider']}")
                return model['modelId']
    
    # Fallback to the first available text model
    for model in available_models:
        if 'TEXT' in model['inputModalities'] and 'TEXT' in model['outputModalities']:
            logging.info(f"Selected fallback model: {model['modelId']}")
            return model['modelId']
    
    return None

def _prepare_request_body(model_id: str, prompt: str) -> Dict[str, Any]:
    """Prepares the model-specific request body for the Bedrock API."""
    
    # Common parameters
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9

    # Use the modern MESSAGES format for models that support it
    messages_payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    if 'anthropic' in model_id.lower():
        if "claude-3" in model_id.lower():
            messages_payload['anthropic_version'] = "bedrock-2023-05-31"
            return messages_payload
        else:
            # Fallback to older Claude 'prompt' API
            return {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
    
    elif 'deepseek' in model_id.lower():
        messages_payload['max_tokens'] = MAX_TOKENS
        # Use 'stop' as determined by previous fix
        messages_payload['stop'] = ["<|end of sentence|>", "[/INST]", "<|User|>"] 
        return messages_payload

    elif 'meta' in model_id.lower():
        # Llama 3 models use the messages API
        messages_payload['max_gen_len'] = MAX_TOKENS
        return messages_payload

    elif 'amazon' in model_id.lower():
        # Titan model format (uses 'inputText')
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "topP": TOP_P
            }
        }
    
    # Default/Generic format (e.g., AI21, older Cohere/Llama)
    return {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

def _parse_response_body(model_id: str, response_body: Dict[str, Any]) -> Optional[str]:
    """
    Parses the model-specific response body to extract the generated text.
    Made more robust to handle various model formats.
    """
    
    # 1. Check for modern Messages API response structure first (Claude 3, Llama 3, DeepSeek)
    if response_body.get('content') and isinstance(response_body['content'], list):
        # Anthropic/Meta Messages API response
        for item in response_body['content']:
            if item.get('type') == 'text':
                return item.get('text', '').strip()
    
    # 2. Check for legacy/provider-specific structures
    if 'anthropic' in model_id.lower():
        # Older Claude response format (completion)
        return response_body.get('completion', '').strip()
    
    elif 'deepseek' in model_id.lower():
        # DeepSeek response format (choices -> text)
        choices = response_body.get('choices', [])
        return choices[0].get('text', '').strip() if choices else None
        
    elif 'amazon' in model_id.lower():
        # Titan response format (results -> outputText)
        results = response_body.get('results', [])
        return results[0].get('outputText', '').strip() if results else None
        
    elif 'meta' in model_id.lower():
        # Llama response format (generation)
        return response_body.get('generation', '').strip()
        
    elif 'ai21' in model_id.lower():
        # AI21 Jurassic response format: completions -> data -> text
        completions = response_body.get('completions', [])
        if completions and completions[0].get('data'):
            return completions[0]['data'].get('text', '').strip()

    elif 'cohere' in model_id.lower():
        # Cohere Command response format: generations -> text
        generations = response_body.get('generations', [])
        if generations:
            return generations[0].get('text', '').strip()
            
    # 3. Generic/Last-Resort Extraction (Covers unexpected structures)
    # Check common fields in case a model uses a simple output
    for field in ['completion', 'generation', 'outputText', 'text', 'output']:
        if field in response_body:
            text_data = response_body[field]
            if isinstance(text_data, str):
                # Simple string output
                return text_data.strip()
            # Handle cases where the output is a nested dictionary with a 'text' key
            if isinstance(text_data, dict) and 'text' in text_data:
                return text_data['text'].strip()
                
    # If all parsing attempts fail
    logging.warning(f"Failed to parse text from model {model_id}. Response keys: {list(response_body.keys())}")
    return None

# --- LLM Model Initialization ---
MODEL_ID = find_suitable_model()
if MODEL_ID:
    logging.info(f"Using Bedrock model: {MODEL_ID}")
else:
    logging.warning("No suitable Bedrock model found. LLM features will use a static fallback.")


def generate_bedrock_advice(prediction: int, user_role: str, max_retries: int = 2) -> str:
    """Generates a text-based recommendation using the selected Bedrock LLM."""
    class_mapping = {0: "Normal", 1: "Suspect", 2: "Pathologic"}
    risk_label = class_mapping.get(prediction, "Unknown")

    prompt = (
        f"As a professional obstetrician, provide brief advice "
        f"(under 3 sentences) for a {user_role}. "
        f"Fetal risk status: {risk_label}. Be concise and professional."
    )

    current_fallback_advice = FALLBACK_ADVICE.format(risk_label=risk_label)

    if bedrock_runtime_client is None or MODEL_ID is None:
        logging.error("Bedrock client or model not available. Returning static fallback advice.")
        return current_fallback_advice

    request_body = _prepare_request_body(MODEL_ID, prompt)

    for attempt in range(max_retries + 1):
        try:
            logging.info(f"Attempt {attempt + 1} - Calling Bedrock with model: {MODEL_ID}")
            
            response = bedrock_runtime_client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            logging.info(f"Bedrock raw response received")
            
            generated_text = _parse_response_body(MODEL_ID, response_body)
            
            if generated_text and len(generated_text) > 10:
                logging.info(f"Successfully extracted Bedrock advice: {generated_text[:100]}...")
                return generated_text
            else:
                return current_fallback_advice

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} - Bedrock call failed: {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                logging.error("Max retries reached. Returning static fallback advice.")
                return current_fallback_advice

    return current_fallback_advice

# -----------------------------------------------------------
# Prediction Service with Clinical Logic
# -----------------------------------------------------------

class PredictionService:
    """Main service handling prediction and medical logic."""

    def predict_risk(self, user_role: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict risk level and generate recommendation."""
        if MODEL is None:
            logging.error("Model not available.")
            raise HTTPException(status_code=503, detail="Machine learning model not loaded.")

        logging.info("Prediction request for role: %s", user_role)

        # 1. Apply clinical override first (replaces the old logic in get_prediction_result)
        if self._requires_clinical_override(features):
            logging.info("Clinical override triggered (AC=0 or FM=0)")
            prediction, confidence = CLINICAL_OVERRIDE_PREDICTION, CLINICAL_OVERRIDE_CONFIDENCE
        else:
            try:
                # Run ML prediction
                feature_values = [features.get(col, 0.0) for col in FEATURE_COLUMNS]
                input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
                prediction = int(MODEL.predict(input_df)[0])
                confidence = float(MODEL.predict_proba(input_df)[0].max())
                
                # 2. Apply clinical sanity check on the ML prediction
                prediction = self._apply_clinical_sanity_check(features, prediction)

            except Exception as e:
                logging.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        logging.info("Calling Bedrock for recommendation...")
        recommendation = generate_bedrock_advice(prediction, user_role)
        
        # Check if it's fallback advice
        is_fallback = "Please consult your healthcare provider" in recommendation 
        advice_source = "Bedrock" if not is_fallback else "Fallback"
        
        logging.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Role: {user_role}")
        logging.info(f"Advice Source: {advice_source}")

        return {
            "predicted_class": prediction,
            "recommendation": recommendation,
            "risk_score": f"{confidence:.4f}",
            "model_used": "CatBoost FetalCare Model + AWS Bedrock LLM",
            "advice_source": advice_source,
            "bedrock_model": MODEL_ID if not is_fallback else "Fallback"
        }

    def _requires_clinical_override(self, features: Dict[str, float]) -> bool:
        """Trigger override if fetal movement or acceleration is zero."""
        # Clinical Rule 1: Immediate Pathologic (2) if no accelerations (AC) or fetal movements (FM)
        return features.get('AC', 0) == 0 or features.get('FM', 0) == 0

    def _apply_clinical_sanity_check(self, features: Dict[str, float], prediction: int) -> int:
        """
        Correct model output based on established medical FHR baselines (ACOG/FIGO).
        Normal FHR: 110–160 bpm
        Bradycardia (severe): < 100 bpm
        Tachycardia (severe): > 180 bpm
        """
        lb = features.get('LB', 0)

        # Clinical Rule 2: Override to Pathologic for Extreme Baseline FHR
        if lb < 100 or lb > 180:
            logging.info("Sanity check: Extreme LB=%.1f bpm detected, forced Pathologic (2)", lb)
            return 2
            
        # Clinical Rule 3: Correct Suspect/Pathologic to Normal if Baseline FHR is reassuring
        # If the machine learning model output a 1 (Suspect) or 2 (Pathologic), but the
        # primary FHR baseline (LB) is perfectly normal, we correct to Normal (0).
        if 110 <= lb <= 160 and prediction > 0:
            logging.warning(
                "Sanity correction: LB=%.1f is Normal, corrected prediction from %d to Normal (0)", 
                lb, prediction
            )
            return 0
            
        # The prediction remains the same if none of the above rules apply (i.e., mild Brady/Tachy
        # outside of 110-160 but not extreme, or other non-reassuring features caused the 1 or 2).
        return prediction


# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="FetalCare Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Instantiate the service
prediction_service = PredictionService() 


class Features(BaseModel):
    LB: float; AC: float; FM: float; UC: float
    ASTV: float; MSTV: float; ALTV: float; MLTV: float
    DL: float; DS: float; DP: float; DR: float
    Width: float; Min: float; Max: float; Nmax: float; Nzeros: float
    Mode: float; Mean: float; Median: float; Variance: float; Tendency: float

class PredictionRequest(BaseModel):
    user_role: str
    features: Features

@app.get("/")
async def root():
    return {"message": "FetalCare Backend API is running"}

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "bedrock_available": bedrock_runtime_client is not None,
        "bedrock_model_available": MODEL_ID is not None
    }

@app.get("/health/bedrock")
async def check_bedrock_health():
    """Checks the status and model availability of AWS Bedrock."""
    if bedrock_runtime_client is None:
        return {"status": "error", "message": "Bedrock client not initialized"}
    
    try:
        available_models = get_available_bedrock_models()
        model_available = MODEL_ID is not None
        
        response_data = {
            "status": "success" if model_available else "warning",
            "message": "Bedrock is accessible",
            "selected_model": MODEL_ID,
            "model_available": model_available,
            "available_models_count": len(available_models)
        }
        
        return response_data
        
    except Exception as e:
        return {"status": "error", "message": f"Bedrock health check failed: {str(e)}"}

@app.get("/models")
async def list_models():
    """List all available Bedrock models"""
    try:
        models = get_available_bedrock_models()
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to list models: {str(e)}"}

@app.post("/predict")
async def predict_risk(data: PredictionRequest, request: Request):
    logging.info(f"Incoming request from {request.client.host}. User role: {data.user_role}")
    try:
        # Use the new PredictionService instance
        result = prediction_service.predict_risk(data.user_role, data.features.model_dump())
        logging.info(f"Returning prediction result. Advice source: {result.get('advice_source', 'Unknown')}")
        return result
    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unhandled exception in predict_risk: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)