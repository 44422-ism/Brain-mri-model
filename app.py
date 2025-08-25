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
TUMOR_CLASSES = ["Tumour"]  # Add more classes if needed
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
def preprocess_image(img, interpreter):
    """Resize, normalize, and match the TFLite model input shape."""
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']  # e.g., [1,224,224,3]
    h, w, c = input_shape[1], input_shape[2], input_shape[3]

    # Convert grayscale to RGB if needed
    if img.mode != "RGB" and c == 3:
        img = img.convert("RGB")
    elif img.mode != "L" and c == 1:
        img = img.convert("L")

    img = img.resize((w, h))
    arr = np.array(img).astype('float32')

    # Normalize to zero-mean, unit-variance
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)

    # Add batch dimension if missing
    if len(arr.shape) == 3:
        arr = np.expand_dims(arr, axis=0)

    # If model expects 3 channels but array has 1, replicate channels
    if arr.shape[-1] != c:
        arr = np.repeat(arr, c, axis=-1)

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
        img_array = preprocess_image(img, mri_interpreter)
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
                # replicate channels if needed
                if patch_arr.shape[-1] == 1 and tumor_interpreter.get_input_details()[0]['shape'][-1] == 3:
                    patch_arr = np.repeat(patch_arr, 3, axis=-1)
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
