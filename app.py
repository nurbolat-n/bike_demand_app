import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Demand Predictor",
    page_icon="🚲",
    layout="centered"
)

# ── Modern UI CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #f7f9fc;
}

/* Container */
.block-container {
    max-width: 780px;
    padding: 2rem 1.5rem;
}

/* Header */
.app-header h1 {
    font-size: 2.6rem;
    font-weight: 600;
    color: #111827;
    text-align: center;
    margin-bottom: 0.3rem;
}
.app-header p {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
}

/* Section title */
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2563eb;
    margin: 2rem 0 0.7rem 0;
}

/* Card */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

/* Inputs */
.stSlider label, .stSelectbox label {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

/* Button */
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 1rem;
    font-weight: 500;
    transition: 0.2s;
}
.stButton > button:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
}

/* Result */
.result-box {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-top: 1rem;
}
.result-number {
    font-size: 3.8rem;
    font-weight: 600;
    color: #111827;
}
.result-unit {
    color: #6b7280;
}
.result-context {
    margin-top: 0.7rem;
    font-size: 0.9rem;
    color: #2563eb;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 3rem;
}

/* Hide streamlit junk */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🚲 Bike Demand Predictor</h1>
    <p>Predict hourly bike rentals using ML</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs wrapped in cards ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Time & Date</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        hr = st.slider("Hour", 0, 23, 8)
    with col2:
        mnth = st.slider("Month", 1, 12, 6)
    with col3:
        yr = st.selectbox("Year", [0,1], format_func=lambda x: "2011" if x==0 else "2012")

    col4, col5 = st.columns(2)
    with col4:
        weekday = st.selectbox("Weekday", list(range(7)),
            format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x])
    with col5:
        season = st.selectbox("Season", [1,2,3,4],
            format_func=lambda x: ["Winter","Spring","Summer","Autumn"][x-1])

    st.markdown('</div>', unsafe_allow_html=True)

# Weather
st.markdown('<div class="section-label">Weather</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    weathersit = st.selectbox("Weather", [1,2,3,4])
with col7:
    temp = st.slider("Temperature", 0.0, 1.0, 0.5)

col8, col9 = st.columns(2)
with col8:
    hum = st.slider("Humidity", 0.0, 1.0, 0.6)
with col9:
    windspeed = st.slider("Wind speed", 0.0, 1.0, 0.2)

st.markdown('</div>', unsafe_allow_html=True)

# Day type
st.markdown('<div class="section-label">Day Type</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col10, col11 = st.columns(2)
with col10:
    workingday = st.selectbox("Working day", [1,0])
with col11:
    holiday = st.selectbox("Holiday", [0,1])

st.markdown('</div>', unsafe_allow_html=True)

# Button
predict = st.button("Predict 🚀", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict:
    input_df = pd.DataFrame([{
        'hr': hr, 'season': season, 'yr': yr, 'mnth': mnth,
        'holiday': holiday, 'workingday': workingday,
        'weathersit': weathersit, 'temp': temp, 'atemp': temp,
        'hum': hum, 'windspeed': windspeed, 'weekday': weekday
    }])

    pred_log = model.predict(input_df)[0]
    pred_cnt = int(np.expm1(pred_log))

    st.markdown(f"""
    <div class="result-box">
        <div class="result-number">{pred_cnt}</div>
        <div class="result-unit">bikes / hour</div>
        <div class="result-context">Predicted demand</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
ML Project · Bike Sharing Dataset
</div>
""", unsafe_allow_html=True)
