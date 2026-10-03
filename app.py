import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Traffic Sign & Road Feature Detector",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign & Road Feature Detector")
st.markdown("""
This Deep Learning web app uses a **Convolutional Neural Network (MobileNetV2)** 
to classify uploaded traffic signs, road signals, and automotive obstacles in real time.
""")

# 2. Load Pretrained CNN Model (Cached in memory)
@st.cache_resource
def load_cnn_model():
    # Pre-trained deep neural network backbone
    model = MobileNetV2(weights="imagenet")
    return model

with st.spinner("Loading Deep Learning Model..."):
    model = load_cnn_model()

# 3. Image Uploader Component
st.subheader("Upload an Image")
uploaded_file = st.file_uploader(
    "Choose a traffic sign or road image (JPG, PNG, JPEG)", 
    type=["jpg", "png", "jpeg"]
)

# 4. Image Processing & Model Prediction
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Road Image", use_column_width=True)

    # Preprocess image for MobileNetV2 (Resizing to 224x224 RGB tensor)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    if st.button("Classify Image with CNN", type="primary"):
        with st.spinner("Running forward pass through neural network..."):
            predictions = model.predict(img_preprocessed)
            decoded = decode_predictions(predictions, top=3)[0]

        st.subheader("Top Model Predictions")
        for i, (imagenet_id, label, prob) in enumerate(decoded):
            clean_label = label.replace("_", " ").title()
            confidence = prob * 100
            st.write(f"**{i+1}. {clean_label}** — Confidence: `{confidence:.2f}%`")
            st.progress(float(prob))
