import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import pandas as pd
import io

st.set_page_config(page_title="WeldoneAI", layout="wide")
st.title("WeldoneAI - Анализ сварных швов")

# 1️⃣ Загружаем модель (кэшируем для ускорения)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # путь к твоей модели
    return model

model = load_model()

st.write("Модель загружена и готова к работе!")

# 2️⃣ Загрузка изображения пользователем
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Исходное изображение", use_column_width=True)

    # 3️⃣ Получение предсказаний
    results = model.predict(image, imgsz=640, device=0, half=True)

    # 4️⃣ Визуализация предсказаний
    results.render()  # рисует рамки и маски прямо на image
    st.image(results.imgs[0], caption="Результат модели", use_column_width=True)

    # 5️⃣ Создание таблицы с предсказаниями
    df = results.pandas().xyxy[0]  # DataFrame с координатами и метками
    st.subheader("Детали дефектов")
    st.dataframe(df)

    # 6️⃣ Скачивание CSV отчета
    csv = df.to_csv(index=False)
    st.download_button(
        label="Скачать отчет CSV",
        data=csv,
        file_name="weldone_report.csv",
        mime="text/csv"
    )