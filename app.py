import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

# === PAGE CONFIG ===
st.set_page_config(page_title="Brain Tumor Classification + MRI Detector", layout="wide")
st.title("🧠 Brain Tumor Classification + MRI Detector")
st.markdown("""
Upload MRI images to classify them as Brain MRI / Other MRI / Not MRI and detect tumor type.

⚠ **Note:** Predictions are based on trained models and may **not be 100% accurate**. Always consult a medical professional.
""")

# === LABELS ===
MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ["Tumour"]  # Update if multiple tumor classes exist

# === LOAD TFLITE MODELS ===
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

mri_interpreter = load_tflite_model("multi_class_mri_detector.tflite")
tumor_interpreter = load_tflite_model("tumor_classifier_roi.tflite")

# === PATCH EXTRACTION USING PIL ONLY ===
IMG_SIZE = 224
STRIDE = 112

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

# === TFLITE PREDICTION ===
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def preprocess_image(img):
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("Upload MRI images", type=["jpg","jpeg","png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"📂 {file.name}")
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # === MRI TYPE PREDICTION WITH HIGH-CONFIDENCE FILTER ===
        img_array = preprocess_image(img)
        mri_pred = predict_tflite(mri_interpreter, img_array)
        mri_conf = mri_pred.max()
        mri_label = MRI_CLASSES[np.argmax(mri_pred)]

        if mri_label == "Brain MRI" and mri_conf >= 0.85:
            # === PATCH-BASED TUMOR PREDICTION ===
            patches = extract_patches_pil(img)
            patch_preds = []
            for patch in patches:
                patch = np.expand_dims(patch.astype('float32') / 255.0, axis=0)
                pred = predict_tflite(tumor_interpreter, patch)
                patch_preds.append(pred)
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
