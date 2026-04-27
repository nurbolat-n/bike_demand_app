import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# ── Custom transformers (required for joblib to load the model) ────────────────
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])

class InteractionAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'workingday' in X.columns and 'hr' in X.columns:
            X['workingday_x_hr'] = X['workingday'] * X['hr']
        return X

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col_periods):
        self.col_periods = col_periods
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col, period in self.col_periods.items():
            if col in X.columns:
                X[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / period)
                X[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / period)
                X = X.drop(columns=[col])
        return X

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Demand Predictor",
    page_icon="🚲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page background */
.stApp {
    background: #f7f9fc;
    color: #1f2937;
}

/* Main container */
.block-container {
    max-width: 760px !important;
    padding: 2.5rem 2rem 4rem 2rem !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}
.app-header h1 {
    font-size: 2.6rem;  /* было 2.2 */
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.03em;
    margin: 0;
}
.app-header p {
    font-size: 1rem;  /* было 0.92 */
    color: #6b7280;
    margin: 0.4rem 0 0 0;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-size: 0.85rem;  /* было 0.7 */
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2563eb;
    margin: 1.8rem 0 0.8rem 0;
}

/* Card */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #eef2ff 0%, #ffffff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1.2rem 0;
}
.result-number {
    font-family: 'DM Mono', monospace;
    font-size: 3.8rem;  /* было 3.5 */
    font-weight: 500;
    color: #2563eb;
    line-height: 1;
}
.result-unit {
    font-size: 0.95rem;
    color: #6b7280;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
}
.result-context {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.8rem;
}

/* Slider labels */
.stSlider label {
    font-size: 0.95rem !important;  /* увеличили */
    color: #374151 !important;
    font-weight: 400 !important;
}

/* Selectbox labels */
.stSelectbox label {
    font-size: 0.95rem !important;
    color: #374151 !important;
    font-weight: 400 !important;
}

/* Button */
.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.25) !important;
}

/* Divider */
hr {
    border-color: #e5e7eb !important;
    margin: 1.5rem 0 !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.8rem;
    color: #9ca3af;
    padding-top: 2rem;
    line-height: 1.7;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* SHAP plot background */
.stImage img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🚲 Bike Demand Predictor</h1>
    <p>Hourly rental forecast · Capital Bikeshare, Washington D.C.</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Time & Date</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    hr = st.slider("Hour of day", 0, 23, 8, help="0 = midnight · 17 = evening peak")
with col2:
    mnth = st.slider("Month", 1, 12, 6)
with col3:
    yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")

col4, col5 = st.columns(2)
with col4:
    weekday = st.selectbox(
        "Day of week",
        options=[0, 1, 2, 3, 4, 5, 6],
        index=1,
        format_func=lambda x: ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][x]
    )
with col5:
    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1:"Winter (Jan–Mar)", 2:"Spring (Apr–Jun)", 3:"Summer (Jul–Sep)", 4:"Autumn (Oct–Dec)"}[x]
    )

st.markdown('<div class="section-label">Weather Conditions</div>', unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    weathersit = st.selectbox(
        "Weather situation",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1:"Clear / Partly cloudy", 2:"Mist / Cloudy", 3:"Light rain or snow", 4:"Heavy rain / Thunderstorm"}[x]
    )
with col7:
    temp = st.slider("Temperature (normalised)", 0.0, 1.0, 0.5, step=0.01,
                     help="0 → −8 °C · 0.5 → 20 °C · 1.0 → 39 °C")

col8, col9 = st.columns(2)
with col8:
    hum = st.slider("Humidity (normalised)", 0.0, 1.0, 0.6, step=0.01)
with col9:
    windspeed = st.slider("Wind speed (normalised)", 0.0, 1.0, 0.2, step=0.01)

st.markdown('<div class="section-label">Day Type</div>', unsafe_allow_html=True)
col10, col11 = st.columns(2)

with col10:
    workingday = st.radio(
        "Working day?",
        options=[1, 0],
        horizontal=True,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col11:
    holiday = st.radio(
        "Public holiday?",
        options=[1, 0],
        horizontal=True,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.markdown("")  # small spacer

# ── Predict button ─────────────────────────────────────────────────────────────
predict = st.button("Predict demand →", use_container_width=True, type="primary")

if predict:
    input_df = pd.DataFrame([{
        'hr':         hr,
        'season':     season,
        'yr':         yr,
        'mnth':       mnth,
        'holiday':    holiday,
        'workingday': workingday,
        'weathersit': weathersit,
        'temp':       temp,
        'atemp':      temp,
        'hum':        hum,
        'windspeed':  windspeed,
        'weekday':    weekday,
    }])

    pred_log = model.predict(input_df)[0]
    pred_cnt = int(np.expm1(pred_log))

    if workingday == 1 and hr in [7, 8, 9, 17, 18, 19]:
        context = "Rush hour · working day"
    elif workingday == 0:
        context = "Weekend / holiday"
    else:
        context = "Regular working hours"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-number">{pred_cnt}</div>
        <div class="result-unit">bikes / hour</div>
        <div class="result-context">{context}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP explanation ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model explanation (SHAP)</div>', unsafe_allow_html=True)
    st.caption("Each bar shows how much a feature pushed the prediction up (+) or down (−) from the baseline.")

    try:
        numerical_features = ['yr', 'holiday', 'workingday', 'weathersit',
                               'temp', 'hum', 'windspeed',
                               'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos',
                               'workingday_x_hr']
        categorical_features = ['season', 'weekday']

        ct = model.named_steps['preprocessor']
        cat_ohe = ct.named_transformers_['cat'].named_steps['onehot']
        cat_names = cat_ohe.get_feature_names_out(categorical_features).tolist()
        post_ct_names = numerical_features + cat_names

        selector = model.named_steps['selector']
        support = selector.get_support()
        feature_names = np.array(post_ct_names)[support].tolist()

        from sklearn.pipeline import Pipeline as SKPipeline
        prep_steps = SKPipeline(model.steps[:-1])
        X_transformed = prep_steps.transform(input_df)

        final_estimator = model.named_steps['model']
        explainer = shap.TreeExplainer(final_estimator)
        explanation = shap.Explanation(
            values=explainer.shap_values(X_transformed)[0],
            base_values=explainer.expected_value,
            data=X_transformed[0],
            feature_names=feature_names
        )

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#1a1d27')
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.gcf().set_facecolor('#1a1d27')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"Could not render SHAP chart: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MSc Machine Learning · Team 14<br>
    Dataset: UCI Bike Sharing — Fanaee-T &amp; Gama, 2014 · Model: XGBoost · R² = 0.9553
</div>
""", unsafe_allow_html=True)
