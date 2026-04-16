import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Road Scene Detection Demo", layout="centered")
st.title("Road Scene Detection Demo")
st.write("Upload an image and run object detection using YOLOv8.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
    else:
        results = model(image)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
else:
    st.info("Please upload a road image to test the demo.")
