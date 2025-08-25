import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import pandas as pd

# === PAGE CONFIG ===
st.set_page_config(page_title="Brain Tumor Classification + MRI Detector", layout="wide")
st.title("🧠 Brain Tumor Classification + MRI Detector")
st.markdown("""
Upload MRI images to classify them as Brain MRI / Other MRI / Not MRI, detect tumor type, and visualize tumor location.

⚠ **Note:** Predictions are based on trained models and may **not be 100% accurate**. Always consult a medical professional.
""")

# === MODEL PATHS ===
MRI_MODEL_PATH = "multi_class_mri_detector.tflite"
TUMOR_MODEL_PATH = "tumor_classifier_with_unknown.tflite"

# === LABELS ===
MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor", "Unknown"]

# === LOAD TFLITE MODELS ===
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

mri_interpreter = load_tflite_model(MRI_MODEL_PATH)
tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# === IMAGE PREPROCESSING ===
def preprocess_image(img, interpreter):
    input_shape = interpreter.get_input_details()[0]['shape']
    h, w, c = input_shape[1], input_shape[2], input_shape[3]

    if img.mode != "RGB" and c == 3:
        img = img.convert("RGB")
    elif img.mode != "L" and c == 1:
        img = img.convert("L")

    img = img.resize((w,h))
    arr = np.array(img).astype('float32')
    arr = arr / 255.0

    if len(arr.shape) == 3:
        arr = np.expand_dims(arr, axis=0)

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

# === SLIDING WINDOW PATCHES ===
def extract_patches(img, patch_size=224, stride=112):
    w, h = img.size
    patches = []
    positions = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x,y))
    if len(patches) == 0:
        # if image smaller than patch, use whole image
        patches = [img.resize((patch_size, patch_size))]
        positions = [(0,0)]
    return patches, positions

# === HEATMAP GENERATION ===
def generate_heatmap(img, positions, confidences, patch_size=224):
    heat = Image.new("RGBA", img.size)
    for (x,y), conf in zip(positions, confidences):
        overlay = Image.new("RGBA", (patch_size, patch_size), (255,0,0,int(conf*150)))
        heat.paste(overlay, (x,y), overlay)
    return Image.alpha_composite(img.convert("RGBA"), heat)

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("Upload MRI images", type=["jpg","jpeg","png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"📂 {file.name}")
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # === MRI DETECTION ===
        img_array = preprocess_image(img, mri_interpreter)
        mri_pred = predict_tflite(mri_interpreter, img_array)
        mri_conf = mri_pred.max()
        mri_label = MRI_CLASSES[np.argmax(mri_pred)]
        st.write(f"MRI Prediction: {mri_label} (Confidence: {mri_conf:.2f})")

        # === TUMOR DETECTION (always run) ===
        patches, positions = extract_patches(img)
        patch_confidences = []
        patch_preds = []
        for patch in patches:
            patch_arr = preprocess_image(patch, tumor_interpreter)
            pred = predict_tflite(tumor_interpreter, patch_arr)
            patch_preds.append(pred)
            patch_confidences.append(pred.max())
        patch_preds = np.array(patch_preds)
        mean_conf = patch_preds.mean(axis=0)
        tumor_label = TUMOR_CLASSES[np.argmax(mean_conf)]
        tumor_conf = mean_conf.max()

        st.write(f"Tumor Prediction: {tumor_label} (Confidence: {tumor_conf:.2f})")

        # === HEATMAP ===
        heatmap_img = generate_heatmap(img, positions, patch_confidences)
        st.image(heatmap_img, caption="Tumor Heatmap Overlay", use_column_width=True)

        # === RECORD RESULTS ===
        results.append({
            "File Name": file.name,
            "MRI Type": mri_label,
            "MRI Confidence": round(float(mri_conf),2),
            "Tumor Type": tumor_label,
            "Tumor Confidence": round(float(tumor_conf),2)
        })

    # === SUMMARY TABLE ===
    st.subheader("📊 Summary Table")
    df = pd.DataFrame(results)
    st.table(df)

else:
    st.warning("Please upload one or more MRI images to get predictions.")
