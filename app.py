import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Кастомные трансформеры (нужны чтобы joblib мог загрузить модель) ──────────
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

# ── Загрузка модели ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ── Интерфейс ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚲 Bike Demand Predictor", layout="centered")
st.title("🚲 Bike Demand Predictor")
st.markdown("**Прогноз почасовой аренды велосипедов** — Capital Bikeshare, Washington D.C.")
st.markdown("Выберите условия ниже и нажмите кнопку **Предсказать**.")
st.divider()

# ── Входные данные ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    hr = st.slider("🕐 Час дня", min_value=0, max_value=23, value=8,
                   help="0 = полночь, 8 = утро, 17 = вечерний пик")

    season = st.selectbox("🌿 Сезон", options=[1, 2, 3, 4],
                          format_func=lambda x: {1:"Зима (Янв–Март)",
                                                  2:"Весна (Апр–Июнь)",
                                                  3:"Лето (Июль–Сент)",
                                                  4:"Осень (Окт–Дек)"}[x])

    weathersit = st.selectbox("🌤 Погода", options=[1, 2, 3, 4],
                               format_func=lambda x: {1:"Ясно / Облачно",
                                                       2:"Туман",
                                                       3:"Лёгкий дождь/снег",
                                                       4:"Сильный дождь/гроза"}[x])

    temp = st.slider("🌡 Температура (нормализованная)", 0.0, 1.0, 0.5, step=0.01,
                     help="0 = -8°C, 0.5 = 20°C, 1.0 = 39°C")

with col2:
    yr = st.selectbox("📅 Год", options=[0, 1],
                      format_func=lambda x: "2011" if x == 0 else "2012")

    mnth = st.slider("📆 Месяц", min_value=1, max_value=12, value=6)

    workingday = st.radio("💼 Рабочий день?", options=[1, 0],
                          format_func=lambda x: "Да" if x == 1 else "Нет (выходной/праздник)")

    holiday = st.radio("🎉 Праздник?", options=[0, 1],
                       format_func=lambda x: "Нет" if x == 0 else "Да")

    hum = st.slider("💧 Влажность (норм.)", 0.0, 1.0, 0.6, step=0.01)
    windspeed = st.slider("💨 Скорость ветра (норм.)", 0.0, 1.0, 0.2, step=0.01)

weekday = st.select_slider("📅 День недели",
                            options=[0, 1, 2, 3, 4, 5, 6],
                            value=1,
                            format_func=lambda x: ["Воскресенье","Понедельник","Вторник",
                                                    "Среда","Четверг","Пятница","Суббота"][x])

st.divider()

# ── Предсказание ───────────────────────────────────────────────────────────────
if st.button("🔮 Предсказать спрос", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([{
        'hr':         hr,
        'season':     season,
        'yr':         yr,
        'mnth':       mnth,
        'holiday':    holiday,
        'workingday': workingday,
        'weathersit': weathersit,
        'temp':       temp,
        'atemp':      temp,       # будет удалён пайплайном (r=0.99 с temp)
        'hum':        hum,
        'windspeed':  windspeed,
        'weekday':    weekday,
    }])

    # Прогноз на log-шкале → обратное преобразование в кол-во велосипедов
    pred_log = model.predict(input_df)[0]
    pred_cnt = int(np.expm1(pred_log))

    # Определяем контекст поездки
    if workingday == 1 and hr in [7, 8, 9, 17, 18, 19]:
        context = "🏢 Час пик — рабочий день"
    elif workingday == 0:
        context = "🌳 Выходной день"
    else:
        context = "📋 Обычное рабочее время"

    st.success(f"**Прогноз: {pred_cnt} велосипедов/час**")
    st.info(f"Контекст: {context}")

    # ── SHAP объяснение ────────────────────────────────────────────────────────
    st.subheader("📊 Почему модель так предсказала? (SHAP)")
    st.caption("Каждая полоска показывает, насколько данный признак увеличил (+) или уменьшил (–) прогноз.")

    try:
        # Прогоняем вход через все шаги пайплайна, кроме финальной модели
        prep_pipeline = model[:-1] if hasattr(model, '__getitem__') else \
                        __import__('sklearn.pipeline', fromlist=['Pipeline']).Pipeline(model.steps[:-1])

        # Получаем имена признаков
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

        # Трансформируем вход
        from sklearn.pipeline import Pipeline as SKPipeline
        prep_steps = SKPipeline(model.steps[:-1])
        X_transformed = prep_steps.transform(input_df)

        # SHAP waterfall
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
        st.warning(f"SHAP-график не удалось построить: {e}")

# ── Подвал ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("MSc Machine Learning — Team 14 | Dataset: UCI Bike Sharing (Fanaee-T & Gama, 2014) | Model: XGBoost | R² = 0.9553")
