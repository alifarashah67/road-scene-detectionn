import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="Road Scene Detection Demo", layout="wide")

st.title("Road Scene Detection Demo")
st.write("Upload a road image and run object detection with YOLOv8.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(image)
    annotated = results[0].plot()

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Detection Result",
        use_container_width=True
    )
else:
    st.info("Please upload a road scene image to test the model.")
