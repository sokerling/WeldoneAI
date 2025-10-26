import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# ----------------------------
# Заголовок приложения
# ----------------------------
st.title("WeldoneAI - Прототип анализа сварных швов")
st.write("Загружайте рентгеновские снимки, и модель YOLO покажет дефекты!")

# ----------------------------
# Загрузка модели
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(path="best.pt"):
    model = YOLO(path)
    return model

model = load_model()
st.success("Модель загружена!")

# ----------------------------
# Загрузка изображения
# ----------------------------
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Исходное изображение", use_container_width=True)

    # Инференс на CPU
    with st.spinner("Модель обрабатывает изображение..."):
        results = model.predict(image, imgsz=640, device='cpu', half=False)
        result = results[0]  # первый результат
        annotated_image = result.plot()  # возвращает np.ndarray с визуализацией

    st.image(annotated_image, caption="Результат модели", use_container_width=True)

    # Таблица с результатами
    df = result.boxes.data if result.boxes is not None else []
    if df != []:
        df = pd.DataFrame(df.cpu().numpy(), columns=["x1","y1","x2","y2","confidence","class"])
        st.subheader("Обнаруженные дефекты")
        st.dataframe(df)

        # Кнопка скачивания CSV
        csv = df.to_csv(index=False)
        st.download_button("Скачать CSV отчёт", csv, file_name="weldone_report.csv")
    else:
        st.info("Дефекты не обнаружены")