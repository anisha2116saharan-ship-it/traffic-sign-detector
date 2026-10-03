# Traffic Sign & Road Feature Detector

A Deep Learning computer vision application that classifies road signs and urban driving features using a Convolutional Neural Network (CNN).

## Project Overview
* **Domain:** Computer Vision / Intelligent Transportation Systems
* **Architecture:** MobileNetV2 (Deep Convolutional Neural Network with Inverted Residuals)
* **Frameworks:** Streamlit, TensorFlow, NumPy, Pillow
* **Input Resolution:** 224x224 RGB images

## Features
* Drag-and-drop image file uploader supporting standard web image formats.
* Real-time normalization and forward pass inference.
* Top-3 classification outputs with visual confidence distribution bars.

## How to Run Locally
```bash
git clone [https://github.com/](https://github.com/)<your-username>/traffic-sign-detector.git
cd traffic-sign-detector
pip install -r requirements.txt
streamlit run app.py
