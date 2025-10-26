# app.py
import streamlit as st
from PIL import Image
import torch  # для YOLOv11
import pandas as pd
import io

st.set_page_config(page_title="WeldoneAI", layout="wide")
st.title("WeldoneAI - Анализ сварных швов")

# 1️⃣ Загрузка снимка
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Исходное изображение", use_column_width=True)

    # 2️⃣ Загрузка модели YOLOv11
    @st.cache_resource  # кэшируем модель для ускорения повторных запусков
    def load_model():
        model = torch.hub.load('ultralytics/yolov11', 'custom', path='best_model.pt')
        return model

    model = load_model()

    # 3️⃣ Получение предсказаний
    results = model(image)

    # 4️⃣ Визуализация предсказаний
    results.render()  # модифицирует image
    st.image(results.imgs[0], caption="Результат модели", use_column_width=True)

    # 5️⃣ Создание отчёта
    df = results.pandas().xyxy[0]  # DataFrame с координатами и метками
    st.subheader("Детали дефектов")
    st.dataframe(df)

    # 6️⃣ Кнопка для скачивания CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Скачать отчет CSV",
        data=csv,
        file_name="weldone_report.csv",
        mime="text/csv"
    )