import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Brain Tumor Classification + MRI Detector", layout="centered")
st.title("🧠 Brain Tumor Classification + MRI Detector")
st.write("""
Upload MRI images to detect tumors and classify them as Brain MRI / Other MRI / Not MRI.
⚠ Note: Predictions are based on trained models and may not be 100% accurate. Always consult a medical professional.
""")

# === Paths to models ===
MRI_MODEL_PATH = "mri_detector_retrained.tflite"
TUMOR_MODEL_PATH = "tumor_classifier_roi.tflite"

# === MRI classes ===
MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]

# === Load TFLite model function ===
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load models
mri_interpreter = load_tflite_model(MRI_MODEL_PATH)
tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# === Prediction functions ===
def predict_tflite(interpreter, image, input_size=(224, 224)):
    img = image.resize(input_size).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

# === File uploader ===
uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # MRI prediction
    mri_pred = predict_tflite(mri_interpreter, img)
    mri_class = MRI_CLASSES[np.argmax(mri_pred)]
    st.write(f"**MRI Prediction:** {mri_class} (Confidence: {np.max(mri_pred):.2f})")

    # Tumor prediction only if Brain MRI
    if mri_class == "Brain MRI":
        tumor_pred = predict_tflite(tumor_interpreter, img)
        tumor_class = TUMOR_CLASSES[np.argmax(tumor_pred)]
        st.write(f"**Tumor Prediction:** {tumor_class} (Confidence: {np.max(tumor_pred):.2f})")
    else:
        st.write("**No tumor prediction. Image not recognized as Brain MRI.**")
