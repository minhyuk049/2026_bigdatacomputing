
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

def set_korean_font():
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family']        = 'NanumGothic'
    else:

        plt.rcParams['font.family']        = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


st.set_page_config(page_title="기대수명 예측 서비스", layout="wide")
st.title(" WHO 기대수명 예측 AI 서비스")
st.markdown(
    "수업 파이프라인 **(Linear / Poly / Ridge)** 을 활용한 "
    "**다중 특성 회귀 분석** 대시보드"
)

@st.cache_resource
def load_all():
    m_linear = joblib.load('model_linear.pkl')
    m_poly   = joblib.load('model_poly.pkl')
    m_ridge  = joblib.load('model_ridge.pkl')
    stats    = joblib.load('feature_stats.pkl')
    results  = joblib.load('model_results.pkl')
    return m_linear, m_poly, m_ridge, stats, results

model_linear, model_poly, model_ridge, feature_stats, model_results = load_all()

models_dict = {
    "Linear (1차 선형 회귀)":         model_linear,
    "Poly (3차 다항 회귀, 규제없음)":  model_poly,
    "Ridge (3차 다항 + Ridge 규제)":   model_ridge,
}

FEATURES = feature_stats['features']

st.sidebar.header("📋 입력값 설정")

selected_model_name = st.sidebar.selectbox(
    " 예측 모델 선택",
    list(models_dict.keys())
)
selected_model = models_dict[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.subheader(" 특성값 조절")


slider_configs = {
    'Adult mortality': ("Adult mortality (성인 사망률)", 1.0,    723.0,  168.0),
    'BMI':             ("BMI (체질량지수)",              2.0,     77.0,   38.0),
    'GDP':             ("GDP (달러)",                    1.0, 120000.0, 5566.0),
    'Alcohol':         ("Alcohol (알코올 소비량)",        0.0,     18.0,    4.5),
}

user_inputs = {}
for feat in FEATURES:
    label, mn, mx, default = slider_configs[feat]
    user_inputs[feat] = st.sidebar.slider(label, mn, mx, default)


input_array = np.array([[user_inputs[f] for f in FEATURES]])
prediction  = selected_model.predict(input_array)[0]


col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 예측 결과")
    st.metric(
        label=f"모델: {selected_model_name.split('(')[0].strip()}",
        value=f"{prediction:.2f} 세",
        delta=f"{prediction - 69.3:.2f} (세계 평균 대비)"
    )
    st.markdown("---")
    st.write("** 입력된 특성값 요약**")
    input_df = pd.DataFrame({
        '특성':  FEATURES,
        '입력값': [round(user_inputs[f], 2) for f in FEATURES]
    })
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader(" 입력값 상대 수준")

    f_min  = np.array([slider_configs[f][1] for f in FEATURES])
    f_max  = np.array([slider_configs[f][2] for f in FEATURES])
    norm_v = (np.array([user_inputs[f] for f in FEATURES]) - f_min) / (f_max - f_min)


    fig1, ax1 = plt.subplots(figsize=(5, 2.8), dpi=100)
    bars1 = ax1.barh(FEATURES, norm_v, color='steelblue', alpha=0.75)
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("Normalized value (0~1)")
    ax1.set_title("Input Feature Relative Level")
    for bar, val in zip(bars1, norm_v):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', fontsize=9)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

st.markdown("---")
st.subheader(" 모델 성능 비교")

perf_df = pd.DataFrame(model_results).T.reset_index()
perf_df.columns = ['Model', 'Train R2', 'Test R2', 'Train MSE', 'Test MSE', 'Complexity']
st.dataframe(perf_df, use_container_width=True)


st.subheader(" Test R² 점수 비교 (Bar Chart)")

model_names  = list(model_results.keys())
display_r2_vals = [max(model_results[m]['Test R2'], -5)for m in model_names]
colors       = ['#4C72B0', '#DD8452', '#55A868']


fig2, ax2 = plt.subplots(figsize=(6, 3), dpi=100)

bars2 = ax2.bar(model_names,display_r2_vals,color=colors,alpha=0.85,edgecolor='black',width=0.5)

ax2.set_ylim(min(display_r2_vals) - abs(min(display_r2_vals)) * 0.1 - 0.5, 1.2)
ax2.set_ylabel("Test R2 Score")
ax2.set_title("3-Model Test R2 Comparison\n"
              "(Higher is better / Negative = Overfitting)")
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.0)

for bar, real_val, display_val in zip(bars2,display_r2_vals,display_r2_vals):

    ax2.text(bar.get_x() + bar.get_width()/2,display_val + 0.1,f'{real_val:.4f}',ha='center')

fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.info(
    "**Model Interpretation**\n\n"
    "- **Linear**: Stable but simple → possible underfitting\n"
    "- **Poly (no regularization)**: Train R2≈1 but Test R2 negative → **Overfitting**\n"
    "- **Ridge (regularized)**: Ridge penalty suppresses overfitting → best generalization"
)

st.markdown("---")
st.caption(
    "Assignment | WHO Life Expectancy Dataset | "
    "scikit-learn Pipeline + Streamlit"
)
