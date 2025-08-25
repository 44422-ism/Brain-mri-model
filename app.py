import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# =======================
# Helper functions
# =======================
@st.cache_data
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, target_size=(224,224)):
    img = image.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    if len(img_array.shape) == 2:  # grayscale to RGB
        img_array = np.stack((img_array,)*3, axis=-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# =======================
# Model paths
# =======================
MRI_MODEL_PATH = "models/mri_detector_retrained (1).tflite"
TUMOR_MODEL_PATH = "models/tumor_classifier_roi (2).tflite"

# =======================
# Load models
# =======================
mri_interpreter = load_tflite_model(MRI_MODEL_PATH)
tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]

# =======================
# Streamlit UI
# =======================
st.title("🧠 Brain Tumor Classification + MRI Detector")
st.write("Upload MRI images to detect tumors and classify them as Brain MRI / Other MRI / Not MRI.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Predict MRI type
    mri_pred = predict_tflite(mri_interpreter, img_array)
    mri_index = np.argmax(mri_pred)
    mri_conf = mri_pred[mri_index]
    mri_type = MRI_CLASSES[mri_index]

    st.write(f"**MRI Prediction:** {mri_type} (Confidence: {mri_conf:.2f})")

    # Predict tumor only if Brain MRI, else force prediction
    force_tumor = False
    if mri_type != "Brain MRI":
        st.warning("No tumor prediction. Image not recognized as Brain MRI. Forcing tumor prediction...")
        force_tumor = True

    tumor_pred = predict_tflite(tumor_interpreter, img_array)
    tumor_index = np.argmax(tumor_pred)
    tumor_conf = tumor_pred[tumor_index]
    tumor_type = TUMOR_CLASSES[tumor_index]

    if force_tumor:
        st.info(f"**Tumor Prediction (forced):** {tumor_type} (Confidence: {tumor_conf:.2f})")
    elif mri_type == "Brain MRI":
        st.write(f"**Tumor Prediction:** {tumor_type} (Confidence: {tumor_conf:.2f})")

st.write("⚠ Note: Predictions are based on trained models and may not be 100% accurate. Always consult a medical professional.")
