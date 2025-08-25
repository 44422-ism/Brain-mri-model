import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

# === PAGE CONFIG ===
st.set_page_config(page_title="Brain Tumor Classification + MRI Detector", layout="wide")
st.title("🧠 Brain Tumor Classification + MRI Detector (Debug Mode)")
st.markdown("""
Upload MRI images to classify them as Brain MRI / Other MRI / Not MRI and detect tumor type.

⚠ **Note:** Predictions are based on trained models and may **not be 100% accurate**. Always consult a medical professional.
""")

# === SETTINGS ===
MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ["Tumour"]  # Add more if needed
IMG_SIZE = 224
STRIDE = 112
MRI_CONF_THRESHOLD = st.slider("MRI confidence threshold", 0.5, 0.95, 0.85, 0.05)

# === MODEL PATHS ===
MRI_MODEL_PATH = "multi_class_mri_detector.tflite"
TUMOR_MODEL_PATH = "tumor_classifier_roi.tflite"

# Check models exist
if not os.path.exists(MRI_MODEL_PATH):
    st.error(f"MRI model not found: {MRI_MODEL_PATH}")
if not os.path.exists(TUMOR_MODEL_PATH):
    st.error(f"Tumor model not found: {TUMOR_MODEL_PATH}")

# === LOAD MODELS ===
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

mri_interpreter = load_tflite_model(MRI_MODEL_PATH)
tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# === PATCH EXTRACTION (PIL only) ===
def extract_patches_pil(img, patch_size=IMG_SIZE, stride=STRIDE):
    w, h = img.size
    if w < patch_size or h < patch_size:
        img = img.resize((patch_size, patch_size))
        return [np.array(img)]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(np.array(patch))
    return patches

# === PREPROCESS IMAGE ===
def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype('float32')
    # Zero-mean, unit-variance normalization
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    arr = np.expand_dims(arr, axis=0)
    return arr

# === PREDICTION ===
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("Upload MRI images", type=["jpg","jpeg","png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"📂 {file.name}")
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # === MRI TYPE PREDICTION ===
        img_array = preprocess_image(img)
        mri_pred = predict_tflite(mri_interpreter, img_array)
        mri_conf = mri_pred.max()
        mri_label = MRI_CLASSES[np.argmax(mri_pred)]
        st.write(f"MRI Prediction: {mri_label} (Confidence: {mri_conf:.2f})")

        if mri_label == "Brain MRI" and mri_conf >= MRI_CONF_THRESHOLD:
            # === PATCH-BASED TUMOR PREDICTION ===
            patches = extract_patches_pil(img)
            patch_preds = []
            for i, patch in enumerate(patches):
                patch_arr = np.expand_dims(patch.astype('float32') / 255.0, axis=0)
                pred = predict_tflite(tumor_interpreter, patch_arr)
                patch_preds.append(pred)
                st.write(f"Patch {i+1}/{len(patches)} prediction: {pred}")

            patch_preds = np.array(patch_preds)
            mean_conf = patch_preds.mean(axis=0)
            tumor_label = TUMOR_CLASSES[np.argmax(mean_conf)]
            tumor_conf = mean_conf.max()
        else:
            mri_label = "Unknown / Not MRI"
            tumor_label = "N/A"
            tumor_conf = 0.0

        results.append({
            "File Name": file.name,
            "MRI Type": mri_label,
            "MRI Confidence": round(float(mri_conf), 2),
            "Tumor Type": tumor_label,
            "Tumor Confidence": round(float(tumor_conf), 2)
        })

    # === SUMMARY TABLE ===
    st.subheader("📊 Summary Table")
    df = pd.DataFrame(results)
    st.table(df)

else:
    st.warning("Please upload one or more MRI images to get predictions.")
