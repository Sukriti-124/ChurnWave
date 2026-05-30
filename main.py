import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnWave | Telecom",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d0f1a;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111320 !important;
    border-right: 1px solid #1e2235;
}
[data-testid="stSidebar"] * { color: #c8cde0 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 0.9rem !important; 
    padding: 6px 0 !important; 
}

/* ── Headers ── */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

/* ── Cards ── */
.cs-card {
    background: #141628;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.cs-card-accent {
    background: linear-gradient(135deg, #141a35 0%, #0d1228 100%);
    border: 1px solid #2a3155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Step pills ── */
.step-pill {
    display: inline-block;
    background: #1a1f3a;
    border: 1px solid #2a3155;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #7b8ec8;
    margin-bottom: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.step-pill.active {
    background: #1e3a8a;
    border-color: #3b82f6;
    color: #93c5fd;
}

/* ── Risk badges ── */
.badge-high {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #fca5a5;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-medium {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    border: 1px solid #f59e0b;
    color: #fcd34d;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-low {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid #10b981;
    color: #6ee7b7;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Prediction result box ── */
.result-high {
    background: linear-gradient(135deg, rgba(127,29,29,0.3) 0%, rgba(69,10,10,0.2) 100%);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-low {
    background: linear-gradient(135deg, rgba(6,78,59,0.3) 0%, rgba(2,44,34,0.2) 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-medium {
    background: linear-gradient(135deg, rgba(120,53,15,0.3) 0%, rgba(69,26,3,0.2) 100%);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ── Metric overrides ── */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #e8eaf0 !important;
}
[data-testid="stMetricLabel"] { color: #7b8ec8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Form elements ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: #1a1f3a !important;
    border-color: #2a3155 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
label { color: #9ba3c0 !important; font-size: 0.85rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1e3a8a, #3b82f6) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111320 !important;
    border-radius: 8px !important;
    border: 1px solid #1e2235 !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7b8ec8 !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2a5e !important;
    color: #93c5fd !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2235 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr { border-color: #1e2235 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #141628 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 10px !important;
}

/* ── Alert/info ── */
[data-testid="stAlert"] {
    background: #1a1f3a !important;
    border: 1px solid #2a3155 !important;
    color: #c8cde0 !important;
    border-radius: 10px !important;
}

/* ── Spinner text ── */
.stSpinner > div { color: #7b8ec8 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #141628 !important;
    border: 2px dashed #2a3155 !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: #2a3155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    'Gender', 'Age', 'Married', 'Number of Dependents', 'Zip Code',
    'Latitude', 'Longitude', 'Number of Referrals', 'Tenure in Months',
    'Phone Service', 'Avg Monthly Long Distance Charges', 'Multiple Lines',
    'Internet Service', 'Avg Monthly GB Download', 'Online Security',
    'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
    'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data',
    'Paperless Billing', 'Monthly Charge', 'Total Charges', 'Total Refunds',
    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue',
    'TotalServices'
]

MODEL_COLORS = {
    'Random Forest': '#3b82f6',
    'XGBoost': '#8b5cf6',
    'Gradient Boosting': '#f59e0b',
    'Logistic Regression': '#10b981'
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models['Random Forest']       = pickle.load(open('models/random_forest_model.pkl', 'rb'))
            models['Gradient Boosting']   = pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
            models['Logistic Regression'] = pickle.load(open('models/logistic_regression_model.pkl', 'rb'))
            try:
                models['XGBoost'] = pickle.load(open('models/xgboost_model.pkl', 'rb'))
            except Exception:
                pass  # XGBoost optional if not installed

            scaler         = joblib.load('models/scaler.pkl')
            label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
            model_perf     = pd.read_csv('models/model_performance.csv', index_col='Model')

        return models, scaler, label_encoders, model_perf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('telecom_customer_churn.csv')
    except Exception:
        return None

@st.cache_data
def get_feature_importances():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = pickle.load(open('models/random_forest_model.pkl', 'rb'))
        gb = pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
    fi_rf = pd.Series(rf.feature_importances_, index=FEATURE_NAMES)
    fi_gb = pd.Series(gb.feature_importances_, index=FEATURE_NAMES)
    avg   = ((fi_rf + fi_gb) / 2).sort_values(ascending=False)
    return avg, fi_rf.sort_values(ascending=False), fi_gb.sort_values(ascending=False)

@st.cache_data
def load_zip_lookup():
    try:
        return pickle.load(open('zip_lookup.pkl', 'rb'))
    except Exception:
        return {}

models, scaler, label_encoders, model_performance = load_models()
dataset = load_dataset()
zip_lookup = load_zip_lookup()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def prepare_input(df):
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0
    df = df[FEATURE_NAMES]
    return scaler.transform(df)

def predict_all(X, selected_models):
    preds, probs = {}, {}
    for name in selected_models:
        if name in models:
            try:
                probs[name]  = models[name].predict_proba(X)[:, 1]
                preds[name]  = models[name].predict(X)
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {e}")
    return preds, probs

def risk_label(p):
    if p >= 0.7:  return "High Risk",   "badge-high"
    if p >= 0.35: return "Medium Risk", "badge-medium"
    return "Low Risk", "badge-low"

def gauge(value, title, height=260):
    color = "#ef4444" if value > 0.7 else ("#f59e0b" if value > 0.35 else "#10b981")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': '#e8eaf0', 'family': 'Syne'}},
        title={'text': title, 'font': {'size': 13, 'color': '#9ba3c0', 'family': 'DM Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': '#5a637a'}, 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#1a1f3a',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35],  'color': 'rgba(16,185,129,0.12)'},
                {'range': [35, 70], 'color': 'rgba(245,158,11,0.12)'},
                {'range': [70, 100],'color': 'rgba(239,68,68,0.12)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8eaf0'
    )
    return fig

def dark_chart_layout(fig, title="", height=400):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family='Syne', size=15, color='#e8eaf0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,22,40,0.6)',
        font=dict(family='DM Sans', color='#9ba3c0'),
        xaxis=dict(gridcolor='#1e2235', zerolinecolor='#1e2235'),
        yaxis=dict(gridcolor='#1e2235', zerolinecolor='#1e2235'),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ba3c0'))
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <div style='font-family: Syne; font-size: 1.4rem; font-weight: 800; 
                    background: linear-gradient(90deg, #3b82f6, #8b5cf6); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ChurnWave
        </div>
        <div style='font-size: 0.72rem; color: #5a637a; margin-top: 2px; letter-spacing: 0.08em; text-transform: uppercase;'>
            Telecom Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🎯  Single Prediction", "📂  Bulk Prediction", "📊  Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#5a637a; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.6rem;'>Active Models</div>", unsafe_allow_html=True)

    available_model_names = list(models.keys())
    selected_models = []
    for mname in available_model_names:
        col_dot, col_check = st.columns([1, 8])
        with col_dot:
            dot_color = MODEL_COLORS.get(mname, '#5a637a')
            st.markdown(f"<div style='width:8px;height:8px;border-radius:50%;background:{dot_color};margin-top:12px;'></div>", unsafe_allow_html=True)
        with col_check:
            if st.checkbox(mname, value=True, key=f"model_{mname}"):
                selected_models.append(mname)

    if not selected_models:
        st.warning("Select at least one model")
        selected_models = available_model_names[:1]

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#5a637a;'>
        <div style='margin-bottom:4px;'>📦 {len(models)} models loaded</div>
        <div style='margin-bottom:4px;'>📋 {len(FEATURE_NAMES)} features</div>
        <div>🗂 7,043 training records</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE HEADER HELPER
# ─────────────────────────────────────────────
def page_header(icon, title, subtitle):
    st.markdown(f"""
    <div style='padding: 1.5rem 0 1rem 0;'>
        <div style='font-size: 0.75rem; color: #5a637a; font-weight: 600; letter-spacing: 0.1em; 
                    text-transform: uppercase; margin-bottom: 0.5rem;'>{icon} {subtitle}</div>
        <div style='font-family: Syne; font-size: 2rem; font-weight: 800; color: #e8eaf0; 
                    line-height: 1.1;'>{title}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# ═══════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION (Wizard)
# ═══════════════════════════════════════════════════════
if page == "🎯  Single Prediction":
    page_header("🎯", "Single Customer Prediction", "Churn Prediction")

    # Wizard step state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    TOTAL_STEPS = 4
    step = st.session_state.wizard_step

    # ── Progress bar ──
    st.progress(step / TOTAL_STEPS)
    step_labels = ["Demographics", "Services", "Add-ons & Billing", "Financial Details"]
    cols_steps = st.columns(TOTAL_STEPS)
    for i, label in enumerate(step_labels):
        active = "active" if (i + 1) == step else ""
        done   = "✓ " if (i + 1) < step else f"{i+1}. "
        cols_steps[i].markdown(f"<div class='step-pill {active}'>{done}{label}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────
    # STEP 1 — Demographics
    # ──────────────────────────────────────
    if step == 1:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### 👤 Customer Demographics")
        st.markdown("<div style='color:#5a637a; font-size:0.85rem; margin-bottom:1rem;'>Basic customer profile information</div>", unsafe_allow_html=True)

        fd = st.session_state.form_data
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender  = st.selectbox("Gender", ["Male", "Female"], index=0 if fd.get('gender','Male')=='Male' else 1)
        with c2:
            age     = st.number_input("Age", 18, 100, fd.get('age', 35))
        with c3:
            married = st.selectbox("Married", ["Yes", "No"], index=0 if fd.get('married','Yes')=='Yes' else 1)
        with c4:
            num_dep = st.number_input("Number of Dependents", 0, 10, fd.get('num_dep', 0))

        c5, c6, c7 = st.columns(3)
        with c5:
            num_ref  = st.number_input("Number of Referrals", 0, 50, fd.get('num_ref', 0))
        with c6:
            tenure   = st.number_input("Tenure in Months", 0, 100, fd.get('tenure', 12))
        with c7:
            zip_code = st.number_input("Zip Code (CA)", 90001, 96150, fd.get('zip_code', 90001))

        # Auto-derive lat/long from zip — show city as confirmation
        zip_info = zip_lookup.get(int(zip_code), {})
        latitude  = zip_info.get('Latitude',  34.0522)
        longitude = zip_info.get('Longitude', -118.2437)
        city      = zip_info.get('City', 'Unknown')
        if zip_info:
            st.markdown(f"<div style='font-size:0.8rem; color:#5a6380; margin-top:-0.3rem;'>📍 <strong style='color:#7b8ec8;'>{city}</strong> — coordinates resolved automatically from zip code</div>", unsafe_allow_html=True)
        else:
            st.warning("Zip code not in dataset — using default coordinates.")

        st.markdown("</div>", unsafe_allow_html=True)

        _, btn_col = st.columns([3, 1])
        if btn_col.button("Next →", use_container_width=True):
            st.session_state.form_data.update(dict(
                gender=gender, age=age, married=married, num_dep=num_dep,
                num_ref=num_ref, tenure=tenure, zip_code=zip_code,
                latitude=latitude, longitude=longitude
            ))
            st.session_state.wizard_step = 2
            st.rerun()

    # ──────────────────────────────────────
    # STEP 2 — Core Services
    # ──────────────────────────────────────
    elif step == 2:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### 📞 Core Services")
        st.markdown("<div style='color:#5a637a; font-size:0.85rem; margin-bottom:1rem;'>Primary telecom service subscriptions</div>", unsafe_allow_html=True)

        fd = st.session_state.form_data
        c1, c2, c3 = st.columns(3)
        with c1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"],
                index=0 if fd.get('phone_service','Yes')=='Yes' else 1)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"],
                index=["No","Yes","No phone service"].index(fd.get('multiple_lines','No')))
        with c2:
            internet_service = st.selectbox("Internet Service", ["No", "Yes"],
                index=["No","Yes"].index(fd.get('internet_service','Yes')))
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"],
                index=0 if fd.get('unlimited_data','Yes')=='Yes' else 1)
        with c3:
            avg_gb = st.number_input("Avg Monthly GB Download", 0, 85,
                int(fd.get('avg_gb', 10)), step=1,
                help="From billing system — average gigabytes downloaded per month")

        st.markdown("</div>", unsafe_allow_html=True)
        back_col, _, next_col = st.columns([1, 2, 1])
        if back_col.button("← Back", use_container_width=True):
            st.session_state.wizard_step = 1; st.rerun()
        if next_col.button("Next →", use_container_width=True):
            st.session_state.form_data.update(dict(
                phone_service=phone_service, multiple_lines=multiple_lines,
                internet_service=internet_service,
                avg_gb=avg_gb, unlimited_data=unlimited_data
            ))
            st.session_state.wizard_step = 3; st.rerun()

    # ──────────────────────────────────────
    # STEP 3 — Add-ons & Billing
    # ──────────────────────────────────────
    elif step == 3:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### 📡 Add-on Services & Billing")
        st.markdown("<div style='color:#5a637a; font-size:0.85rem; margin-bottom:1rem;'>Additional service subscriptions and billing preferences</div>", unsafe_allow_html=True)

        fd = st.session_state.form_data
        opts = ["No", "Yes", "No internet service"]

        c1, c2, c3 = st.columns(3)
        with c1:
            online_sec = st.selectbox("Online Security",   opts, index=opts.index(fd.get('online_sec','No')))
            online_bak = st.selectbox("Online Backup",     opts, index=opts.index(fd.get('online_bak','No')))
            dev_prot   = st.selectbox("Device Protection", opts, index=opts.index(fd.get('dev_prot','No')))
        with c2:
            prem_tech  = st.selectbox("Premium Tech Support", opts, index=opts.index(fd.get('prem_tech','No')))
            stream_tv  = st.selectbox("Streaming TV",         opts, index=opts.index(fd.get('stream_tv','No')))
            stream_mov = st.selectbox("Streaming Movies",     opts, index=opts.index(fd.get('stream_mov','No')))
        with c3:
            stream_mus = st.selectbox("Streaming Music",  opts, index=opts.index(fd.get('stream_mus','No')))
            paperless  = st.selectbox("Paperless Billing", ["Yes", "No"],
                index=0 if fd.get('paperless','Yes')=='Yes' else 1)

        st.markdown("</div>", unsafe_allow_html=True)
        back_col, _, next_col = st.columns([1, 2, 1])
        if back_col.button("← Back", use_container_width=True):
            st.session_state.wizard_step = 2; st.rerun()
        if next_col.button("Next →", use_container_width=True):
            st.session_state.form_data.update(dict(
                online_sec=online_sec, online_bak=online_bak, dev_prot=dev_prot,
                prem_tech=prem_tech, stream_tv=stream_tv, stream_mov=stream_mov,
                stream_mus=stream_mus, paperless=paperless
            ))
            st.session_state.wizard_step = 4; st.rerun()

    # ──────────────────────────────────────
    # STEP 4 — Financial + Predict
    # ──────────────────────────────────────
    elif step == 4:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### 💳 Financial Information")
        st.markdown("<div style='color:#5a637a; font-size:0.85rem; margin-bottom:1rem;'>From billing system — enter values from the customer account</div>", unsafe_allow_html=True)

        fd = st.session_state.form_data
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_charge = st.number_input("Monthly Charge ($)",             0.0, value=fd.get('monthly_charge', 70.0),  format="%.2f")
            total_charges  = st.number_input("Total Charges ($)",              0.0, value=fd.get('total_charges', 1000.0), format="%.2f")
        with c2:
            total_refunds  = st.number_input("Total Refunds ($)",              0.0, value=fd.get('total_refunds', 0.0),    format="%.2f")
            total_extra    = st.number_input("Total Extra Data Charges ($)",   0,   value=fd.get('total_extra', 0))
        with c3:
            total_ld       = st.number_input("Total Long Distance Charges ($)", 0.0, value=fd.get('total_ld', 0.0), format="%.2f")

        # Derived values — shown as info, not inputs
        tenure_val       = fd.get('tenure', 12)
        avg_ld_derived   = round(total_ld / tenure_val, 2) if tenure_val > 0 else 0.0
        total_rev_derived = round(total_charges - total_refunds + total_extra + total_ld, 2)

        st.markdown(f"""
        <div style='background:#0f1525; border:1px solid #1e2235; border-radius:8px; 
                    padding:0.75rem 1rem; margin-top:0.5rem; display:flex; gap:2rem;'>
            <div>
                <div style='font-size:0.7rem; color:#3a4260; text-transform:uppercase; 
                            letter-spacing:0.08em;'>Auto-calculated</div>
            </div>
            <div>
                <div style='font-size:0.75rem; color:#5a6380;'>Avg Monthly LD Charges</div>
                <div style='font-size:1rem; font-weight:600; color:#7b8ec8;'>${avg_ld_derived:.2f}</div>
            </div>
            <div>
                <div style='font-size:0.75rem; color:#5a6380;'>Total Revenue</div>
                <div style='font-size:1rem; font-weight:600; color:#7b8ec8;'>${total_rev_derived:.2f}</div>
            </div>
            <div style='font-size:0.75rem; color:#3a4260; align-self:center; font-style:italic;'>
                Derived from fields above — no input needed
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        back_col, _, predict_col = st.columns([1, 1, 2])
        if back_col.button("← Back", use_container_width=True):
            st.session_state.wizard_step = 3; st.rerun()

        run = predict_col.button("🔮  Run Prediction", use_container_width=True)

        if run:
            fd.update(dict(monthly_charge=monthly_charge, total_charges=total_charges,
                           total_refunds=total_refunds, total_extra=total_extra,
                           total_ld=total_ld,
                           avg_ld=avg_ld_derived,
                           total_revenue=total_rev_derived))

            services_count = sum(1 for s in [fd.get('online_sec'), fd.get('online_bak'),
                                              fd.get('dev_prot'), fd.get('prem_tech'),
                                              fd.get('stream_tv'), fd.get('stream_mov'),
                                              fd.get('stream_mus')] if s == "Yes")

            input_df = pd.DataFrame([{
                'Gender': 1 if fd['gender'] == "Male" else 0,
                'Age': fd['age'],
                'Married': 1 if fd['married'] == "Yes" else 0,
                'Number of Dependents': fd['num_dep'],
                'Zip Code': fd['zip_code'],
                'Latitude': fd['latitude'],
                'Longitude': fd['longitude'],
                'Number of Referrals': fd['num_ref'],
                'Tenure in Months': fd['tenure'],
                'Phone Service': 1 if fd['phone_service'] == "Yes" else 0,
                'Avg Monthly Long Distance Charges': fd['avg_ld'],   # derived: total_ld / tenure
                'Multiple Lines': 1 if fd['multiple_lines'] == "Yes" else 0,
                'Internet Service': 1 if fd['internet_service'] == "Yes" else 0,
                'Avg Monthly GB Download': fd['avg_gb'],
                'Online Security': 1 if fd['online_sec'] == "Yes" else 0,
                'Online Backup': 1 if fd['online_bak'] == "Yes" else 0,
                'Device Protection Plan': 1 if fd['dev_prot'] == "Yes" else 0,
                'Premium Tech Support': 1 if fd['prem_tech'] == "Yes" else 0,
                'Streaming TV': 1 if fd['stream_tv'] == "Yes" else 0,
                'Streaming Movies': 1 if fd['stream_mov'] == "Yes" else 0,
                'Streaming Music': 1 if fd['stream_mus'] == "Yes" else 0,
                'Unlimited Data': 1 if fd['unlimited_data'] == "Yes" else 0,
                'Paperless Billing': 1 if fd['paperless'] == "Yes" else 0,
                'Monthly Charge': fd['monthly_charge'],
                'Total Charges': fd['total_charges'],
                'Total Refunds': fd['total_refunds'],
                'Total Extra Data Charges': fd['total_extra'],
                'Total Long Distance Charges': fd['total_ld'],
                'Total Revenue': fd['total_revenue'],               # derived: charges - refunds + extra + ld
                'TotalServices': services_count
            }])

            X = prepare_input(input_df)
            _, probs = predict_all(X, selected_models)

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            # Per-model gauges
            n = len(probs)
            gauge_cols = st.columns(n)
            for i, (name, prob) in enumerate(probs.items()):
                with gauge_cols[i]:
                    st.plotly_chart(gauge(prob[0], name), use_container_width=True, config={'displayModeBar': False})
                    label, badge_cls = risk_label(prob[0])
                    st.markdown(f"<div style='text-align:center'><span class='{badge_cls}'>{label}</span></div>", unsafe_allow_html=True)

            # Ensemble
            avg_prob = float(np.mean(list(probs.values())))
            label, badge_cls = risk_label(avg_prob)

            st.markdown("---")
            _, ec, _ = st.columns([1, 2, 1])
            with ec:
                st.markdown(f"<div style='text-align:center; font-family:Syne; font-size:1rem; color:#9ba3c0; margin-bottom:0.5rem;'>Ensemble ({len(probs)} models)</div>", unsafe_allow_html=True)
                st.plotly_chart(gauge(avg_prob, "Ensemble", height=300), use_container_width=True, config={'displayModeBar': False})

            # Result box
            if avg_prob >= 0.7:
                box_cls = "result-high"
                heading = "⚠️ High Churn Risk"
                head_color = "#fca5a5"
                actions = ["Offer personalized retention bundle", "Schedule satisfaction callback",
                           "Review & optimize service plan", "Grant loyalty reward / discount",
                           "Flag for proactive support queue"]
            elif avg_prob >= 0.35:
                box_cls = "result-medium"
                heading = "🔶 Moderate Churn Risk"
                head_color = "#fcd34d"
                actions = ["Monitor usage patterns over next 30 days",
                           "Send targeted upsell for add-on services",
                           "Enroll in loyalty milestone program",
                           "Offer paperless billing incentive"]
            else:
                box_cls = "result-low"
                heading = "✅ Low Churn Risk"
                head_color = "#6ee7b7"
                actions = ["Maintain regular engagement cadence",
                           "Leverage for referral program",
                           "Upsell premium streaming bundle",
                           "Recognize tenure milestone"]

            actions_html = "".join(f"<li style='color:#c8cde0; margin:4px 0;'>{a}</li>" for a in actions)
            st.markdown(f"""
            <div class='{box_cls}'>
                <h3 style='color:{head_color}; font-family:Syne; margin-bottom:0.5rem;'>{heading}</h3>
                <p style='color:#c8cde0; font-size:1.05rem;'>
                    Churn probability: <strong style='color:{head_color};'>{avg_prob*100:.1f}%</strong>
                    &nbsp;·&nbsp; Services active: <strong>{services_count}</strong>
                    &nbsp;·&nbsp; Tenure: <strong>{fd['tenure']} months</strong>
                </p>
                <p style='color:#9ba3c0; font-size:0.85rem; margin-top:1rem; margin-bottom:0.4rem; 
                          font-weight:600; text-transform:uppercase; letter-spacing:0.06em;'>Recommended Actions</p>
                <ul style='margin:0; padding-left:1.2rem;'>{actions_html}</ul>
            </div>
            """, unsafe_allow_html=True)

            # Reset wizard
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄  New Prediction"):
                st.session_state.wizard_step = 1
                st.session_state.form_data = {}
                st.rerun()

# ═══════════════════════════════════════════════════════
# PAGE 2 — BULK PREDICTION
# ═══════════════════════════════════════════════════════
elif page == "📂  Bulk Prediction":
    page_header("📂", "Bulk Prediction", "Batch Processing")

    # Template download
    with st.expander("📥 Download CSV Template", expanded=False):
        sample = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Age': [35, 52],
            'Married': ['Yes', 'No'],
            'Number of Dependents': [2, 0],
            'Zip Code': [90001, 94105],
            'Latitude': [34.0522, 37.7749],
            'Longitude': [-118.2437, -122.4194],
            'Number of Referrals': [3, 0],
            'Tenure in Months': [24, 6],
            'Phone Service': ['Yes', 'Yes'],
            'Avg Monthly Long Distance Charges': [15.5, 5.0],
            'Multiple Lines': ['Yes', 'No'],
            'Internet Service': ['Fiber optic', 'DSL'],
            'Avg Monthly GB Download': [25.0, 8.0],
            'Online Security': ['No', 'Yes'],
            'Online Backup': ['Yes', 'No'],
            'Device Protection Plan': ['No', 'No'],
            'Premium Tech Support': ['Yes', 'No'],
            'Streaming TV': ['Yes', 'No'],
            'Streaming Movies': ['Yes', 'No'],
            'Streaming Music': ['No', 'No'],
            'Unlimited Data': ['Yes', 'No'],
            'Paperless Billing': ['Yes', 'No'],
            'Monthly Charge': [89.5, 45.0],
            'Total Charges': [2146.0, 270.0],
            'Total Refunds': [0.0, 15.0],
            'Total Extra Data Charges': [0, 50],
            'Total Long Distance Charges': [252.0, 30.0],
            'Total Revenue': [2398.0, 335.0],
        })
        csv_bytes = sample.to_csv(index=False).encode()
        b64 = base64.b64encode(csv_bytes).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="churn_template.csv" style="color:#3b82f6;">⬇ churn_template.csv</a>', unsafe_allow_html=True)
        st.dataframe(sample, use_container_width=True)

    uploaded = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(f"✅ **{len(df_raw):,} records** loaded from `{uploaded.name}`")

            tab1, tab2 = st.tabs(["📋 Preview", "⚠️ Data Quality"])
            with tab1:
                st.dataframe(df_raw.head(10), use_container_width=True)
            with tab2:
                missing = df_raw.isnull().sum()
                missing = missing[missing > 0]
                if missing.empty:
                    st.success("No missing values detected.")
                else:
                    st.warning(f"{len(missing)} columns have missing values — will be imputed automatically.")
                    st.dataframe(missing.rename("Missing Count").to_frame(), use_container_width=True)

            missing_cols = set(FEATURE_NAMES) - {'TotalServices'} - set(df_raw.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(sorted(missing_cols))}")
                st.stop()

            if st.button("🔮  Generate Predictions", use_container_width=True):
                with st.spinner("Running models..."):
                    df_proc = df_raw.copy()

                    # Impute
                    for col in df_proc.select_dtypes(['float64','int64']).columns:
                        df_proc[col].fillna(df_proc[col].median(), inplace=True)
                    for col in df_proc.select_dtypes(['object']).columns:
                        mode = df_proc[col].mode()
                        df_proc[col].fillna(mode[0] if len(mode) else 'Unknown', inplace=True)

                    # Feature engineering
                    svc_cols = ['Online Security','Online Backup','Device Protection Plan',
                                'Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music']
                    df_proc['TotalServices'] = sum((df_proc[c]=='Yes').astype(int) for c in svc_cols if c in df_proc.columns)

                    # Encode
                    df_proc['Gender'] = (df_proc['Gender'] == 'Male').astype(int)
                    for col in ['Married','Phone Service','Unlimited Data','Paperless Billing']:
                        if col in df_proc.columns:
                            df_proc[col] = (df_proc[col] == 'Yes').astype(int)
                    for col in ['Multiple Lines','Online Security','Online Backup','Device Protection Plan',
                                'Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music']:
                        if col in df_proc.columns:
                            df_proc[col] = (df_proc[col] == 'Yes').astype(int)
                    if 'Internet Service' in df_proc.columns:
                        df_proc['Internet Service'] = (df_proc['Internet Service'] != 'No').astype(int)
                    for col in FEATURE_NAMES:
                        if col in df_proc.columns:
                            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)

                    X_bulk = prepare_input(df_proc)
                    _, probs = predict_all(X_bulk, selected_models)

                    results = df_raw.copy()
                    for name, p in probs.items():
                        results[f'{name}_Prob'] = np.round(p, 4)
                        results[f'{name}_Pred'] = (p > 0.5).astype(int)
                    if probs:
                        results['Ensemble_Prob'] = np.round(np.mean(list(probs.values()), axis=0), 4)
                        results['Ensemble_Pred'] = (results['Ensemble_Prob'] > 0.5).astype(int)
                        results['Risk_Level'] = results['Ensemble_Prob'].apply(
                            lambda x: 'High Risk' if x >= 0.7 else ('Medium Risk' if x >= 0.35 else 'Low Risk'))

                st.success("✅ Predictions complete!")

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                total  = len(results)
                high   = (results['Risk_Level']=='High Risk').sum()
                medium = (results['Risk_Level']=='Medium Risk').sum()
                low    = (results['Risk_Level']=='Low Risk').sum()
                c1.metric("Total Customers", f"{total:,}")
                c2.metric("🔴 High Risk",   f"{high:,}",   f"{high/total*100:.1f}%")
                c3.metric("🟡 Medium Risk", f"{medium:,}", f"{medium/total*100:.1f}%")
                c4.metric("🟢 Low Risk",    f"{low:,}",    f"{low/total*100:.1f}%")

                # Charts
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_pie = px.pie(
                        values=[high, medium, low],
                        names=['High Risk', 'Medium Risk', 'Low Risk'],
                        color_discrete_map={'High Risk':'#ef4444','Medium Risk':'#f59e0b','Low Risk':'#10b981'},
                        hole=0.45
                    )
                    dark_chart_layout(fig_pie, "Risk Distribution")
                    fig_pie.update_traces(textfont_color='#e8eaf0')
                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

                with col_chart2:
                    fig_hist = px.histogram(
                        results, x='Ensemble_Prob', nbins=30,
                        color_discrete_sequence=['#3b82f6'],
                        labels={'Ensemble_Prob': 'Churn Probability'}
                    )
                    dark_chart_layout(fig_hist, "Probability Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

                # Results table
                st.markdown("### 📋 Results Table")
                default_cols = ['Ensemble_Prob','Ensemble_Pred','Risk_Level']
                display_cols = st.multiselect("Columns to show:", results.columns.tolist(), default=default_cols)
                if display_cols:
                    st.dataframe(
                        results[display_cols].style.background_gradient(
                            subset=['Ensemble_Prob'] if 'Ensemble_Prob' in display_cols else [],
                            cmap='RdYlGn_r'
                        ), use_container_width=True
                    )

                # Downloads
                st.markdown("### 💾 Download Results")
                dc1, dc2 = st.columns(2)
                with dc1:
                    csv_out = results.to_csv(index=False).encode()
                    b64c = base64.b64encode(csv_out).decode()
                    st.markdown(f'<a href="data:file/csv;base64,{b64c}" download="predictions.csv" style="color:#3b82f6; font-weight:600;">📥 Download CSV</a>', unsafe_allow_html=True)
                with dc2:
                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
                        results.to_excel(w, sheet_name='Predictions', index=False)
                    b64x = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64x}" download="predictions.xlsx" style="color:#8b5cf6; font-weight:600;">📥 Download Excel</a>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ═══════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════
elif page == "📊  Model Insights":
    page_header("📊", "Model & Data Insights", "Analytics")

    tab_perf, tab_feat, tab_data, tab_recs = st.tabs([
        "🏆 Model Performance", "🔑 Feature Importance", "📋 Dataset Overview", "💡 Recommendations"
    ])

    # ── TAB 1: Performance ──
    with tab_perf:
        st.markdown("### Model Comparison Metrics")

        # Best model banner
        best = model_performance['ROC-AUC'].idxmax()
        best_auc = model_performance.loc[best, 'ROC-AUC']
        st.markdown(f"""
        <div class='cs-card-accent'>
            <div style='display:flex; align-items:center; gap:1rem;'>
                <div style='font-size:2rem;'>🥇</div>
                <div>
                    <div style='font-family:Syne; font-size:1.2rem; font-weight:700; color:#e8eaf0;'>{best}</div>
                    <div style='color:#7b8ec8; font-size:0.85rem;'>Best model by ROC-AUC · Score: <strong style='color:#3b82f6;'>{best_auc:.4f}</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards
        metrics = model_performance.columns.tolist()
        mc = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            best_val = model_performance[m].max()
            mc[i].metric(m, f"{best_val:.4f}", f"Best: {model_performance[m].idxmax()}")

        # Radar chart
        st.markdown("### Radar Comparison")
        fig_radar = go.Figure()
        for idx, (model_name, row) in enumerate(model_performance.iterrows()):
            color = list(MODEL_COLORS.values())[idx % len(MODEL_COLORS)]
            # Convert hex to rgba for fillcolor
            h = color.lstrip('#')
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            fill_rgba = f'rgba({r},{g},{b},0.15)'
            fig_radar.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=metrics + [metrics[0]],
                fill='toself',
                fillcolor=fill_rgba,
                line=dict(color=color, width=2),
                name=model_name
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(20,22,40,0.8)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1e2235', tickfont=dict(color='#5a637a')),
                angularaxis=dict(gridcolor='#1e2235', tickfont=dict(color='#9ba3c0'))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            height=420,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ba3c0')),
            font=dict(color='#9ba3c0')
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

        # Bar subplots
        st.markdown("### Per-Metric Bar Charts")
        fig_bars = make_subplots(rows=2, cols=3, subplot_titles=metrics,
                                  vertical_spacing=0.18, horizontal_spacing=0.08)
        colors_list = list(MODEL_COLORS.values())
        for idx, metric in enumerate(metrics):
            row, col = idx // 3 + 1, idx % 3 + 1
            fig_bars.add_trace(go.Bar(
                x=model_performance.index,
                y=model_performance[metric],
                marker_color=colors_list,
                text=model_performance[metric].round(3),
                textposition='outside',
                textfont=dict(color='#9ba3c0', size=10),
                showlegend=False
            ), row=row, col=col)
        fig_bars.update_layout(
            height=550, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,22,40,0.6)',
            font=dict(family='DM Sans', color='#9ba3c0'),
            margin=dict(l=10, r=10, t=60, b=10)
        )
        for i in range(1, 7):
            r, c = (i-1) // 3 + 1, (i-1) % 3 + 1
            fig_bars.update_xaxes(tickfont=dict(size=9), row=r, col=c, gridcolor='#1e2235')
            fig_bars.update_yaxes(range=[0.5, 1.0], gridcolor='#1e2235', row=r, col=c)
        st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})

        st.markdown("### Raw Performance Table")
        st.dataframe(
            model_performance.style.format("{:.4f}"),
            use_container_width=True
        )

    # ── TAB 2: Feature Importance ──
    with tab_feat:
        st.markdown("### Real Feature Importances (from trained models)")
        avg_fi, rf_fi, gb_fi = get_feature_importances()

        fi_view = st.radio("View", ["Ensemble Average", "Random Forest", "Gradient Boosting"], horizontal=True)
        fi_data = {'Ensemble Average': avg_fi, 'Random Forest': rf_fi, 'Gradient Boosting': gb_fi}[fi_view]

        top_n = st.slider("Top N features", 5, 30, 15)
        fi_plot = fi_data.head(top_n).reset_index()
        fi_plot.columns = ['Feature', 'Importance']

        fig_fi = px.bar(fi_plot, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Blues',
                        text=fi_plot['Importance'].round(3))
        dark_chart_layout(fig_fi, f"Top {top_n} Features — {fi_view}", height=max(350, top_n * 28))
        fig_fi.update_traces(textfont_color='#e8eaf0', textposition='outside')
        fig_fi.update_coloraxes(showscale=False)
        fig_fi.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

        # Key insight callouts
        top3 = fi_data.head(3).index.tolist()
        st.markdown("#### Top 3 Churn Drivers")
        t1, t2, t3 = st.columns(3)
        icons = ["🥇","🥈","🥉"]
        for i, (col_obj, feat) in enumerate(zip([t1,t2,t3], top3)):
            col_obj.markdown(f"""
            <div class='cs-card' style='text-align:center;'>
                <div style='font-size:1.5rem;'>{icons[i]}</div>
                <div style='font-family:Syne; font-weight:700; font-size:0.95rem; color:#e8eaf0; margin-top:0.3rem;'>{feat}</div>
                <div style='color:#3b82f6; font-size:1.1rem; font-weight:600; margin-top:0.3rem;'>{fi_data[feat]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: Dataset Overview ──
    with tab_data:
        st.markdown("### Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", "7,043")
        c2.metric("Features", "30")
        c3.metric("Churn Rate", "26.5%")
        c4.metric("Avg Tenure", "32.4 mo")

        if dataset is not None:
            col_vis1, col_vis2 = st.columns(2)

            with col_vis1:
                churn_counts = dataset['Customer Status'].value_counts()
                fig_churn = px.pie(values=churn_counts.values, names=churn_counts.index,
                                   color_discrete_sequence=['#3b82f6','#ef4444','#10b981'],
                                   hole=0.4, title="Customer Status Distribution")
                dark_chart_layout(fig_churn, height=350)
                fig_churn.update_traces(textfont_color='#e8eaf0')
                st.plotly_chart(fig_churn, use_container_width=True, config={'displayModeBar': False})

            with col_vis2:
                fig_tenure = px.histogram(dataset, x='Tenure in Months', nbins=30,
                                          color='Customer Status',
                                          color_discrete_map={'Stayed':'#3b82f6','Churned':'#ef4444','Joined':'#10b981'},
                                          barmode='overlay', opacity=0.75)
                dark_chart_layout(fig_tenure, "Tenure by Customer Status", height=350)
                st.plotly_chart(fig_tenure, use_container_width=True, config={'displayModeBar': False})

            col_vis3, col_vis4 = st.columns(2)
            with col_vis3:
                fig_age = px.box(dataset, x='Customer Status', y='Age',
                                 color='Customer Status',
                                 color_discrete_map={'Stayed':'#3b82f6','Churned':'#ef4444','Joined':'#10b981'})
                dark_chart_layout(fig_age, "Age Distribution by Status", height=350)
                fig_age.update_layout(showlegend=False)
                st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})

            with col_vis4:
                fig_charge = px.box(dataset, x='Customer Status', y='Monthly Charge',
                                    color='Customer Status',
                                    color_discrete_map={'Stayed':'#3b82f6','Churned':'#ef4444','Joined':'#10b981'})
                dark_chart_layout(fig_charge, "Monthly Charge by Status", height=350)
                fig_charge.update_layout(showlegend=False)
                st.plotly_chart(fig_charge, use_container_width=True, config={'displayModeBar': False})

            # Churn by Internet Service
            if 'Internet Service' in dataset.columns:
                churn_by_internet = dataset.groupby(['Internet Service','Customer Status']).size().reset_index(name='count')
                fig_internet = px.bar(churn_by_internet, x='Internet Service', y='count',
                                      color='Customer Status',
                                      color_discrete_map={'Stayed':'#3b82f6','Churned':'#ef4444','Joined':'#10b981'},
                                      barmode='group')
                dark_chart_layout(fig_internet, "Churn by Internet Service Type", height=350)
                st.plotly_chart(fig_internet, use_container_width=True, config={'displayModeBar': False})

    # ── TAB 4: Recommendations ──
    with tab_recs:
        st.markdown("### Business Recommendations")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
            <div class='cs-card'>
                <div style='font-family:Syne; font-size:1rem; font-weight:700; color:#ef4444; margin-bottom:0.8rem;'>
                    🔴 High-Risk Interventions
                </div>
                <ul style='color:#c8cde0; line-height:2;'>
                    <li>Offer multi-service retention bundles</li>
                    <li>Proactive outreach for customers &lt; 12 months tenure</li>
                    <li>Review pricing for high monthly-charge accounts</li>
                    <li>Free premium tech support trials (30-day)</li>
                    <li>Incentivize long-term contract switch</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='cs-card'>
                <div style='font-family:Syne; font-size:1rem; font-weight:700; color:#f59e0b; margin-bottom:0.8rem;'>
                    🟡 Medium-Risk Actions
                </div>
                <ul style='color:#c8cde0; line-height:2;'>
                    <li>Monitor monthly usage for 30-day anomalies</li>
                    <li>Targeted upsell for add-on services</li>
                    <li>Enroll in loyalty milestone program</li>
                    <li>Offer paperless billing discount</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div class='cs-card'>
                <div style='font-family:Syne; font-size:1rem; font-weight:700; color:#10b981; margin-bottom:0.8rem;'>
                    🟢 Retention & Growth
                </div>
                <ul style='color:#c8cde0; line-height:2;'>
                    <li>Build referral incentive programs</li>
                    <li>Improve onboarding for first-90-day customers</li>
                    <li>Celebrate tenure milestones with rewards</li>
                    <li>Personalized streaming bundle recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='cs-card'>
                <div style='font-family:Syne; font-size:1rem; font-weight:700; color:#3b82f6; margin-bottom:0.8rem;'>
                    📊 Key Insight Summary
                </div>
                <ul style='color:#c8cde0; line-height:2;'>
                    <li><strong>Tenure</strong> is the #1 churn predictor — protect early months</li>
                    <li><strong>Referral customers</strong> churn significantly less</li>
                    <li><strong>Fiber optic</strong> users show higher churn than DSL</li>
                    <li>Customers with <strong>3+ add-ons</strong> show strong retention</li>
                    <li>High monthly charge alone is a strong churn signal</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#2a3155; font-size:0.78rem; padding:1rem 0;'>
    ChurnWave · Telecom Customer Intelligence · Built with Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
