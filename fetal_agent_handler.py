import json
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
MODEL_FILE = 'fetalcare_model_catboost_safe.pkl'  # Updated to CatBoost clinical model

FEATURE_COLUMNS = [
    'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

def load_model(file_path: str) -> Optional[CatBoostClassifier]:
    """Load the serialized CatBoost model from file."""
    try:
        model = joblib.load(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model '{file_path}': {e}")
        return None

MODEL = load_model(MODEL_FILE)

# --- RECOMMENDATION LOGIC ---
def generate_recommendation(prediction: int, role: str, confidence: float) -> str:
    """Generate role-based recommendations based on prediction."""
    conf_str = f"{confidence:.2f}"

    if prediction == 2:  # Pathologic
        if role == 'Doctor':
            return f"Fetal status: Pathologic. Immediate evaluation required. Confidence: {conf_str}"
        elif role == 'Nurse':
            return "Pathologic FHR detected. Notify physician and prepare emergency protocols."
        elif role == 'Expectant Mother':
            return "Critical fetal variation detected. Contact your healthcare provider immediately."

    if prediction == 1:  # Suspect
        if role == 'Doctor':
            return f"Fetal status: Suspect. Continuous monitoring recommended. Confidence: {conf_str}"
        elif role == 'Nurse':
            return "Suspect FHR detected. Increase monitoring and notify physician."
        elif role == 'Expectant Mother':
            return "Variation detected. Contact your doctor for follow-up."

    # Normal
    if role == 'Doctor':
        return f"Fetal status: Normal. Continue routine monitoring. Confidence: {conf_str}"
    if role == 'Nurse':
        return "Fetal status is normal. Proceed with routine care."
    if role == 'Expectant Mother':
        return "Fetal status is normal. Continue regular prenatal care."

    return "Error: Could not generate recommendation."

# --- PREDICTION ---
def get_prediction_result(user_role: str, features_dict: Dict[str, float]) -> Dict[str, Any]:
    """Prepare data, run CatBoost model, apply clinical overrides, and generate recommendation."""
    if MODEL is None:
        raise Exception("Model not available.")

    # Prepare input DataFrame
    feature_values = [features_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    # Run model prediction
    prediction = int(MODEL.predict(input_df)[0])
    confidence = float(MODEL.predict_proba(input_df)[0].max())

    # --- Clinical Overrides ---
    AC_val = features_dict.get("AC", 0)
    FM_val = features_dict.get("FM", 0)
    override_applied = False

    if AC_val == 0 or FM_val == 0:
        override_applied = True
        prediction = max(prediction, 2)  # Force Pathologic
        confidence = max(confidence, 0.95)

    recommendation = generate_recommendation(prediction, user_role, confidence)

    return {
        "predicted_class": prediction,
        "recommendation": recommendation,
        "risk_score": f"{confidence:.4f}",
        "model_used": "CatBoost FetalCare Clinical Model",
        "override_applied": override_applied
    }

# --- AWS LAMBDA HANDLER ---
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        request_body = json.loads(event.get('body', '{}'))
        user_role = request_body.get("user_role")
        features = request_body.get("features", {})

        if not user_role or not features:
            raise ValueError("Missing user_role or features.")

        result = get_prediction_result(user_role, features)
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result)
        }

    except Exception as e:
        print(f"Error in Lambda handler: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
