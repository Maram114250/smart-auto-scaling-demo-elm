
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Auto-Scaling with AI (Up & Down)", layout="wide")

st.title("⚡️ Smart Auto-Scaling with AI (Up & Down)")
st.caption("نموذج يتنبأ بالضغط ويتخذ قرارات توسعة أو تقليص حسب الحاجة.")

st.sidebar.header("إعدادات المحاكاة")
upper_threshold = st.sidebar.slider("📈 حد التوسعة للأعلى (Upper Threshold)", 150, 300, 210)
lower_threshold = st.sidebar.slider("📉 حد التوسعة للأسفل (Lower Threshold)", 50, 149, 100)
st.sidebar.caption("🔍 النظام يتخذ القرار بناءً على هذه الحدود. إذا تجاوز التوقع العلوي يتم التوسعة، وإذا انخفض دون الحد السفلي يتم تقليل الموارد.")

pattern = st.sidebar.selectbox("نمط الضغط", ["تصاعدي", "متذبذب", "عشوائي"])

np.random.seed(42)
timestamps = [datetime(2025, 4, 22, 10, 0) + timedelta(minutes=i) for i in range(30)]

if pattern == "تصاعدي":
    actual_requests = [160 + int(i * 2.5) + np.random.randint(-2, 4) for i in range(30)]
elif pattern == "متذبذب":
    actual_requests = [180 + int(10 * np.sin(i / 2)) + np.random.randint(-5, 5) for i in range(30)]
else:
    actual_requests = [160 + np.random.randint(-30, 30) for i in range(30)]

df = pd.DataFrame({'Time': timestamps, 'Requests': actual_requests})
df['Minute'] = list(range(len(df)))
X = df[['Minute']]
y = df['Requests']

model = LinearRegression()
model.fit(X, y)

future_minutes = np.array([[len(df) + i] for i in range(1, 4)])
predicted_values = model.predict(future_minutes)
predicted_peak = max(predicted_values)

initial_pods = 3
pods = initial_pods

scaling_time = df['Time'].iloc[-1] + timedelta(minutes=1)
actual_now = df['Requests'].iloc[-1]

tab1, tab2 = st.tabs(["🔬 المحاكاة", "📊 لوحة الإدارة"])

with tab1:
    st.subheader("نتائج التنبؤ")

    if predicted_peak > upper_threshold:
        pods = 5
        st.error("▲ Scale Up: زيادة عدد الـ Pods (3 ➜ 5)")
        st.write(f"**Actual:** {actual_now} — **Predicted Peak:** {int(predicted_peak)} > Upper Threshold {upper_threshold}")
        decision = "Scaled Up"
        st.markdown(f"""
        #### لماذا تم اتخاذ القرار؟
        - التوقعات القادمة كانت: {', '.join(str(int(v)) for v in predicted_values)}.
        - أحد هذه التوقعات ({int(predicted_peak)}) تجاوز الحد الأعلى ({upper_threshold}).
        - تم اتخاذ قرار التوسعة لتفادي الضغط المتوقع.
        """)
    elif predicted_peak < lower_threshold:
        pods = 2
        st.warning("▼ Scale Down: تقليل عدد الـ Pods (3 ➜ 2)")
        st.write(f"**Actual:** {actual_now} — **Predicted Peak:** {int(predicted_peak)} < Lower Threshold {lower_threshold}")
        decision = "Scaled Down"
        st.markdown(f"""
        #### لماذا تم اتخاذ القرار؟
        - التوقعات القادمة كانت: {', '.join(str(int(v)) for v in predicted_values)}.
        - جميعها أقل من الحد الأدنى ({lower_threshold}).
        - يمكن تقليل الموارد دون التأثير على الأداء.
        """)
    else:
        st.success("✅ لا حاجة للتوسعة أو التقليص")
        decision = "Not Scaled"
        st.markdown(f"""
        #### لماذا لم يتم اتخاذ أي إجراء؟
        - التوقعات القادمة كانت: {', '.join(str(int(v)) for v in predicted_values)}.
        - لم تتجاوز أي توقع الحد الأعلى أو تنخفض دون الحد الأدنى.
        - النظام مستقر ولا حاجة للتعديل حالياً.
        """)

    st.markdown(f"### 🕒 تم اتخاذ القرار عند: **{scaling_time.strftime('%Y-%m-%d %H:%M:%S')}**")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Time'], df['Requests'], label='Actual Requests', color='blue')
    ax.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')
    ax.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold')
    ax.scatter([scaling_time + timedelta(minutes=i) for i in range(3)], predicted_values,
               color='orange', label='Predicted (AI)', zorder=5)
    ax.scatter([df['Time'].iloc[-1]], [actual_now], color='black', label=f'Actual = {actual_now}', zorder=5)
    ax.set_title("Traffic Simulation with AI Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Requests")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

    with st.expander("📊 حركة الطلبات"):
        st.dataframe(df.drop(columns=["Minute"])[::-1].reset_index(drop=True))

with tab2:
    st.header("📊 لوحة معلومات الإدارة")
    st.markdown("### ⚙️ ملخص")
    col1, col2, col3 = st.columns(3)
    col1.metric("الطلب الحالي", actual_now)
    col2.metric("أعلى توقع قادم", int(predicted_peak))
    col3.metric("عدد الـ Pods الحالي", pods)
    st.metric("الحد الأعلى", upper_threshold)
    st.metric("الحد الأدنى", lower_threshold)

    st.markdown("---")
    st.markdown("### ✅ التوصية")
    if decision == "Scaled Up":
        st.success("توصية: جرب النظام الذكي في بيئة الإنتاج خلال أوقات الذروة.")
    elif decision == "Scaled Down":
        st.warning("توصية: يمكن تقليل الموارد لتوفير الاستهلاك حسب التوقعات المنخفضة.")
    else:
        st.info("توصية: لا حاجة لاتخاذ أي إجراء حالياً.")
