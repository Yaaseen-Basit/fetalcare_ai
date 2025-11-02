import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/predict"

# Risk classification labels and colors
RISK_LABELS = {
    0: {"name": "Normal", "color": "#10B981"},
    1: {"name": "Suspect", "color": "#F59E0B"},
    2: {"name": "Pathologic", "color": "#EF4444"},
}

# Streamlit page configuration
st.set_page_config(page_title="FetalCare Agent", layout="wide")
st.title("FetalCare Agent: AI-Powered Risk Prediction")
st.markdown("Enter key CTG parameters to assess fetal health status.")

# Blue button styling
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #2563EB;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 1em;
            font-weight: bold;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #1E40AF;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# User role selection
user_role = st.radio("Select User Role:", ("Doctor", "Nurse", "Expectant Mother"), horizontal=True)
if user_role == "Expectant Mother":
    st.warning("This tool is informational only. Please consult your doctor for medical advice.")

# Input mode selection
input_mode = st.radio("Input Mode:", ["Basic", "Advanced"], horizontal=True)

# Initialize session state for basic features
if "LB" not in st.session_state:
    st.session_state.update({"LB": 130, "AC": 2, "FM": 1, "UC": 3})

# Basic input parameters
baseline_fhr = st.number_input("Baseline FHR (LB, bpm)", 50, 200, st.session_state["LB"], key="LB")
accelerations = st.number_input("Accelerations (AC, per 10 min)", 0, 20, st.session_state["AC"], key="AC")
fetal_movements = st.number_input("Fetal Movements (FM, per 10 min)", 0, 20, st.session_state["FM"], key="FM")
uterine_contractions = st.number_input("Uterine Contractions (UC, per 10 min)", 0, 10, st.session_state["UC"], key="UC")

# Update variables from session state
baseline_fhr = st.session_state["LB"]
accelerations = st.session_state["AC"]
fetal_movements = st.session_state["FM"]
uterine_contractions = st.session_state["UC"]

# Advanced features
if input_mode == "Advanced":
    with st.expander("Advanced Features"):
        abnormal_stv = st.number_input("Abnormal STV (%)", 0.0, 100.0, 25.0)
        mean_stv = st.number_input("Mean STV (ms)", 0.0, 100.0, 1.0)
        abnormal_ltv = st.number_input("Abnormal LTV (%)", 0.0, 100.0, 5.0)
        mean_ltv = st.number_input("Mean LTV (ms)", 0.0, 50.0, 10.0)
        light_decelerations = st.number_input("Light Decelerations", 0, 10, 0)
        severe_decelerations = st.number_input("Severe Decelerations", 0, 5, 0)
        prolonged_decelerations = st.number_input("Prolonged Decelerations", 0, 5, 0)
        repetitive_decelerations = st.number_input("Repetitive Decelerations", 0, 5, 0)
        width = st.number_input("Width", 0, 180, 90)
        min_fhr = st.number_input("Min FHR", 50, 150, 110)
        max_fhr = st.number_input("Max FHR", 150, 200, 150)
        mode_val = st.number_input("Mode", 100, 180, 130)
        mean_val = st.number_input("Mean", 100, 180, 131)
        median_val = st.number_input("Median", 100, 180, 130)
        variance = st.number_input("Variance", 0, 300, 20)
        tendency = st.selectbox("Tendency", [-1, 0, 1], index=1)
        n_max = st.number_input("Nmax", 0, 10, 3)
        n_zeros = st.number_input("Nzeros", 0, 10, 0)
else:
    # Auto-calculate derived features
    abnormal_stv = max(0.0, 50 - accelerations * 2 - abs(baseline_fhr - 130)/2)
    mean_stv = max(0.1, 2 - accelerations * 0.1)
    abnormal_ltv = min(100, abs(baseline_fhr - 130))
    mean_ltv = min(50, abs(baseline_fhr - 130)/2 + 5)
    light_decelerations = 1 if baseline_fhr < 100 else 0
    severe_decelerations = 1 if baseline_fhr < 90 else 0
    prolonged_decelerations = 0
    repetitive_decelerations = 0
    width = max(60, min(120, 90 + (baseline_fhr - 130)//2))
    min_fhr = max(50, baseline_fhr - 20)
    max_fhr = min(180, baseline_fhr + 20)
    mode_val = mean_val = median_val = baseline_fhr
    variance = max(5, abs(baseline_fhr - 130))
    n_max = 3
    n_zeros = 0
    tendency = -1 if baseline_fhr < 110 else (1 if baseline_fhr > 150 else 0)

# Generate risk assessment
if st.button("Generate Risk Assessment"):
    payload = {
        "user_role": user_role,
        "features": {
            "LB": baseline_fhr,
            "AC": accelerations,
            "FM": fetal_movements,
            "UC": uterine_contractions,
            "ASTV": abnormal_stv,
            "MSTV": mean_stv,
            "ALTV": abnormal_ltv,
            "MLTV": mean_ltv,
            "DL": light_decelerations,
            "DS": severe_decelerations,
            "DP": prolonged_decelerations,
            "DR": repetitive_decelerations,
            "Width": width,
            "Min": min_fhr,
            "Max": max_fhr,
            "Mode": mode_val,
            "Mean": mean_val,
            "Median": median_val,
            "Variance": variance,
            "Tendency": tendency,
            "Nmax": n_max,
            "Nzeros": n_zeros
        }
    }

    result = None
    override_applied = False
    try:
        with st.spinner("Generating fetal risk assessment..."):
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            if accelerations == 0 or fetal_movements == 0:
                override_applied = True
    except requests.exceptions.RequestException as e:
        st.error(f"Backend request failed: {e}")

    if result:
        # Risk classification
        risk_class = result.get("predicted_class")
        risk_info = RISK_LABELS.get(risk_class, {"name": "Unknown", "color": "#6B7280"})
        border_style = "5px solid #EF4444" if override_applied else "none"

        st.markdown(f"""
            <div style='background-color:{risk_info['color']}; padding:15px; border-radius:10px; color:white; text-align:center; border:{border_style}'>
                <h2 style='margin:0;'>{risk_info['name'].upper()}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Recommendation display
        recommendation_text = result.get("recommendation", "No recommendation available.")

        if user_role == "Expectant Mother":
            bg_color = "#FEF3C7"
            text_color = "#78350F"
        elif user_role == "Nurse":
            bg_color = "#DBEAFE"
            text_color = "#1E40AF"
        else:
            bg_color = "#DCFCE7"
            text_color = "#065F46"

        st.markdown(f"""
            <div style='background-color:{bg_color}; color:{text_color}; padding:20px; border-radius:12px;
                        font-size:1.2em; font-weight:bold; text-align:center; margin-top:10px;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
                {recommendation_text}
            </div>
        """, unsafe_allow_html=True)

        # Confidence score for professionals
        if user_role in ("Doctor", "Nurse"):
            st.markdown(f"Confidence Score: {result.get('risk_score', 'N/A')}")

        # Clinical override warning
        if override_applied:
            st.warning("Clinical override applied: AC=0 or FM=0 → High-risk fetal status detected.")
