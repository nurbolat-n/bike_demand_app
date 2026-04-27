import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Custom transformers (required for loading the model with joblib) ──────────
from sklearn.base import BaseEstimator, TransformerMixin

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

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚲 Bike Demand Predictor", layout="centered")
st.title("🚲 Bike Demand Predictor")
st.markdown("**Hourly bike rental prediction** — Capital Bikeshare, Washington D.C.")
st.markdown("Select the conditions below and click **Predict**.")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("### 🌿 Season")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

if "season" not in st.session_state:
    st.session_state.season = 1

selected = st.session_state.season

with col_s1:
    if st.button("❄️ Winter", type="primary" if selected == 1 else "secondary"):
        st.session_state.season = 1

with col_s2:
    if st.button("🌸 Spring", type="primary" if selected == 2 else "secondary"):
        st.session_state.season = 2

with col_s3:
    if st.button("☀️ Summer", type="primary" if selected == 3 else "secondary"):
        st.session_state.season = 3

with col_s4:
    if st.button("🍂 Autumn", type="primary" if selected == 4 else "secondary"):
        st.session_state.season = 4

season = st.session_state.season

st.divider()
col1, col2 = st.columns(2)

with col1:
    hr = st.slider("🕐 Hour of the day", min_value=0, max_value=23, value=8,
                   help="0 = midnight, 8 = morning, 17 = evening peak")

    weathersit = st.selectbox("🌤 Weather", options=[1, 2, 3, 4],
                             format_func=lambda x: {
                                 1: "Clear / Cloudy",
                                 2: "Mist",
                                 3: "Light rain/snow",
                                 4: "Heavy rain / thunderstorm"
                             }[x])

    temp = st.slider("🌡 Temperature (normalized)", 0.0, 1.0, 0.5, step=0.01,
                     help="0 = -8°C, 0.5 = 20°C, 1.0 = 39°C")

with col2:
    yr = st.selectbox("📅 Year", options=[0, 1],
                      format_func=lambda x: "2011" if x == 0 else "2012")

    mnth = st.slider("📆 Month", min_value=1, max_value=12, value=6)

    workingday = st.radio("💼 Working day?", options=[1, 0],
                          format_func=lambda x: "Yes" if x == 1 else "No (weekend/holiday)")

    holiday = st.radio("🎉 Holiday?", options=[0, 1],
                       format_func=lambda x: "No" if x == 0 else "Yes")

    hum = st.slider("💧 Humidity (normalized)", 0.0, 1.0, 0.6, step=0.01)
    windspeed = st.slider("💨 Windspeed (normalized)", 0.0, 1.0, 0.2, step=0.01)

weekday = st.select_slider("📅 Day of the week",
                           options=[0, 1, 2, 3, 4, 5, 6],
                           value=1,
                           format_func=lambda x: [
                               "Sunday", "Monday", "Tuesday",
                               "Wednesday", "Thursday", "Friday", "Saturday"
                           ][x])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔮 Predict demand", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([{
        'hr': hr,
        'season': season,
        'yr': yr,
        'mnth': mnth,
        'holiday': holiday,
        'workingday': workingday,
        'weathersit': weathersit,
        'temp': temp,
        'atemp': temp,
        'hum': hum,
        'windspeed': windspeed,
        'weekday': weekday,
    }])

    # Log prediction → convert back
    pred_log = model.predict(input_df)[0]
    pred_cnt = int(np.expm1(pred_log))

    # Context
    if workingday == 1 and hr in [7, 8, 9, 17, 18, 19]:
        context = "🏢 Peak hour — working day"
    elif workingday == 0:
        context = "🌳 Weekend"
    else:
        context = "📋 Regular working hours"

    st.success(f"**Prediction: {pred_cnt} bikes/hour**")
    st.info(f"Context: {context}")

    # ── SHAP explanation ─────────────────────────────────────────────────────
    st.subheader("📊 Why did the model predict this? (SHAP)")
    st.caption("Each bar shows how a feature increased (+) or decreased (–) the prediction.")

    try:
        prep_pipeline = model[:-1] if hasattr(model, '__getitem__') else \
                        __import__('sklearn.pipeline', fromlist=['Pipeline']).Pipeline(model.steps[:-1])

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

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"Could not generate SHAP plot: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("MSc Machine Learning — Team 14 | Dataset: UCI Bike Sharing (Fanaee-T & Gama, 2014) | Model: XGBoost | R² = 0.9553")
