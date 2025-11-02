import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

# Constants
API_URL = "http://127.0.0.1:8000/predict"

RISK_LABELS = {
    0: {"name": "NORMAL", "color": "#10B981", "description": "Healthy fetal status"},
    1: {"name": "SUSPECT", "color": "#F59E0B", "description": "Requires monitoring"},
    2: {"name": "PATHOLOGIC", "color": "#EF4444", "description": "Critical condition"}
}

COLORS = {
    "primary": "#2563EB",
    "secondary": "#64748B",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#F8FAFC",
    "card": "#FFFFFF"
}

# Page Configuration
st.set_page_config(
    page_title="FetalCare • Medical Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown(f"""
<style>
.main {{ background-color: {COLORS['background']}; }}
.risk-card {{
    background-color: {COLORS['card']};
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 4px solid;
}}
.critical-action-box {{
    background-color: #FEE2E2;
    border: 3px solid {COLORS['danger']};
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    text-align: center;
}}
.metric-card {{
    background-color: {COLORS['card']};
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    text-align: center;
}}
.recommendation-box {{
    background-color: #EFF6FF;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid {COLORS['primary']};
    margin: 1rem 0;
}}
.stButton>button {{
    background-color: {COLORS['primary']};
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    font-weight: 600;
    border: none;
}}
.stButton>button:hover {{
    background-color: #1E40AF;
}}
.header {{
    background: linear-gradient(135deg, {COLORS['primary']}, #1E40AF);
    color: white;
    padding: 2rem;
    border-radius: 0 0 20px 20px;
    margin-bottom: 2rem;
}}
.role-indicator {{
    background-color: #E0F2FE;
    color: {COLORS['primary']};
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}}
</style>
""", unsafe_allow_html=True)


# Components
def create_risk_gauge(risk_level: int, risk_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level", 'font': {'size': 24}},
        delta={'reference': 0, 'increasing': {'color': COLORS['danger']}},
        gauge={
            'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': RISK_LABELS[risk_level]['color']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.6], 'color': COLORS['success']},
                {'range': [0.6, 1.4], 'color': COLORS['warning']},
                {'range': [1.4, 2], 'color': COLORS['danger']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_level
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_parameter_chart(features: dict) -> go.Figure:
    categories = ['Baseline FHR', 'Accelerations', 'Movements', 'Contractions', 'Variability']
    lb, ac, fm, uc, astv = features['LB'], features['AC'], features['FM'], features['UC'], features['ASTV']

    lb_score = 0
    if lb > 160:
        lb_score = min((lb - 160) * 2.5, 100)
    elif lb < 120:
        lb_score = min((120 - lb) * 2.5, 100)

    values = [
        lb_score,
        100 - min(ac * 50, 100),
        100 - min(fm * 50, 100),
        min(uc * 20, 100),
        min(astv * 2.5, 100)
    ]

    color = COLORS['danger'] if max(values) > 50 else COLORS['primary']
    fill = 'rgba(239, 68, 68, 0.3)' if max(values) > 50 else 'rgba(37, 99, 235, 0.3)'

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line=dict(color=color),
        fillcolor=fill
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def get_pathologic_details(features: dict) -> str:
    concerns = []

    if features['LB'] > 180:
        concerns.append(f"Severe Tachycardia ({features['LB']} bpm)")
    elif features['LB'] < 100:
        concerns.append(f"Severe Bradycardia ({features['LB']} bpm)")
    if features['AC'] == 0:
        concerns.append("Absent Accelerations")
    if features['ASTV'] > 40:
        concerns.append(f"Reduced Short-Term Variability ({features['ASTV']}%)")
    if features['DS'] > 0:
        concerns.append(f"Severe Decelerations ({features['DS']})")
    if features['DP'] > 0:
        concerns.append(f"Prolonged Decelerations ({features['DP']})")

    if not concerns:
        concerns.append("High Risk Score Based on Composite Indicators.")

    return " / ".join(concerns)


def render_sidebar():
    st.markdown("### System Status")

    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.success("Backend Connected")
            st.markdown(f"**ML Model:** {'✅ Loaded' if data.get('model_loaded') else '❌ Error'}")
            st.markdown(f"**LLM:** {data.get('bedrock_model_available', 'Unknown')}")
        else:
            st.error("Backend Issues")
    except Exception:
        st.error("Service Unavailable")

    st.markdown(
        f"<div style='margin-top:2rem; padding:1rem; background:{COLORS['card']}; border-radius:8px;'>"
        f"<small>Last status check: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small></div>",
        unsafe_allow_html=True
    )


def render_header():
    st.markdown(f"""
    <div class="header">
        <h1>FetalCare</h1>
        <p>Advanced Fetal Health Monitoring & Clinical Intelligence</p>
    </div>
    """, unsafe_allow_html=True)


# Main App
def main():
    render_header()
    with st.sidebar:
        render_sidebar()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Clinical Parameters")
        user_role = st.radio("Medical Role", ["Doctor", "Nurse", "Expectant Mother"], horizontal=True)
        st.markdown(f'<div class="role-indicator">Current Role: {user_role}</div>', unsafe_allow_html=True)

        if user_role == "Expectant Mother":
            st.info("This tool provides insights only. Always consult a doctor for decisions.")

        input_mode = st.radio("Input Mode", ["Basic Assessment", "Advanced Analysis"], horizontal=True)
        st.markdown("#### Core Fetal Parameters")

        col_a, col_b = st.columns(2)
        with col_a:
            baseline_fhr = st.slider("Baseline FHR (bpm)", 50, 200, 130)
            accelerations = st.slider("Accelerations (per 10min)", 0, 20, 2)
        with col_b:
            fetal_movements = st.slider("Fetal Movements (per 10min)", 0, 20, 1)
            uterine_contractions = st.slider("Uterine Contractions (per 10min)", 0, 10, 3)

        if input_mode == "Advanced Analysis":
            with st.expander("Advanced Parameters", expanded=True):
                col1b, col2b = st.columns(2)
                with col1b:
                    dl = st.number_input("Light Decelerations (DL)", 0.0, 10.0, 0.5)
                    ds = st.number_input("Severe Decelerations (DS)", 0.0, 10.0, 0.1)
                    dp = st.number_input("Prolonged Decelerations (DP)", 0.0, 10.0, 0.2)
                    dr = st.number_input("Abnormal Decelerations (%)", 0.0, 100.0, 1.0)
                    width = st.number_input("Histogram Width", 0.0, 50.0, 10.0)
                    min_val = st.number_input("Histogram Min", 50.0, 150.0, 90.0)
                    max_val = st.number_input("Histogram Max", 100.0, 200.0, 160.0)
                    nmax = st.number_input("Histogram Peaks", 0.0, 10.0, 3.0)
                    nzeros = st.number_input("Histogram Zeros", 0.0, 10.0, 1.0)
                with col2b:
                    mode = st.number_input("Histogram Mode", 50.0, 200.0, 120.0)
                    mean = st.number_input("Histogram Mean", 50.0, 200.0, 130.0)
                    median = st.number_input("Histogram Median", 50.0, 200.0, 128.0)
                    variance = st.number_input("Histogram Variance", 0.0, 1000.0, 30.0)
                    tendency = st.number_input("Histogram Tendency", -2.0, 2.0, 0.0)
                    astv = st.number_input("Abnormal STV (%)", 0.0, 100.0, 15.0)
                    mstv = st.number_input("Mean STV", 0.0, 10.0, 1.2)
                    altv = st.number_input("Abnormal LTV (%)", 0.0, 100.0, 10.0)
                    mltv = st.number_input("Mean LTV", 0.0, 10.0, 2.5)
        else:
            dl = ds = dp = dr = 0
            width, min_val, max_val, nmax, nzeros = 10, 90, 160, 3, 0
            mode, mean, median, variance, tendency = 120, 130, 128, 30, 0
            astv, mstv, altv, mltv = 15, 1.0, 10, 2.5

    with col2:
        st.markdown("### Fetal Health Assessment")

        features = {
            "LB": baseline_fhr, "AC": accelerations, "FM": fetal_movements, "UC": uterine_contractions,
            "DL": dl, "DS": ds, "DP": dp, "DR": dr,
            "Width": width, "Min": min_val, "Max": max_val,
            "Nmax": nmax, "Nzeros": nzeros, "Mode": mode, "Mean": mean,
            "Median": median, "Variance": variance, "Tendency": tendency,
            "ASTV": astv, "MSTV": mstv, "ALTV": altv, "MLTV": mltv
        }

        if st.button("Generate Assessment", use_container_width=True):
            payload = {"user_role": user_role, "features": features}
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                if response.status_code == 200:
                    st.session_state.result = response.json()
                    st.session_state.show_result = True
                else:
                    st.error(f"Assessment failed ({response.status_code}).")
            except Exception as e:
                st.error(f"Connection error: {e}")

        if st.session_state.get("show_result") and st.session_state.get("result"):
            result = st.session_state.result
            risk_class = result.get("predicted_class", 0)
            risk_info = RISK_LABELS.get(risk_class, RISK_LABELS[0])
            risk_score = float(result.get("risk_score", risk_class))

            if risk_class == 2:
                details = get_pathologic_details(features)
                st.markdown(f"""
                <div class="critical-action-box">
                    <h1 style="color:{COLORS['danger']};">⚠️ CRITICAL DIAGNOSIS: PATHOLOGIC STATE ⚠️</h1>
                    <p><b>Immediate clinical confirmation required.</b></p>
                    <p><b>Key Indicators:</b> {details}</p>
                    <p><b>Recommendation:</b> {result.get("recommendation", "Consult full report.")}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card" style="border-left-color:{risk_info['color']}">
                    <h2 style="color:{risk_info['color']}">{risk_info['name']}</h2>
                    <p>{risk_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Diagnostic Visualization")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(create_risk_gauge(risk_class, risk_score), use_container_width=True)
            with c2:
                st.plotly_chart(create_parameter_chart(features), use_container_width=True)

            if risk_class != 2:
                st.markdown(f"""
                    <div class="recommendation-box">
                        <h4>Clinical Recommendation</h4>
                        <p>{result.get("recommendation", "Consult clinical report for next steps.")}</p>
                    </div>
                """, unsafe_allow_html=True)


if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "result" not in st.session_state:
    st.session_state.result = None

if __name__ == "__main__":
    main()
