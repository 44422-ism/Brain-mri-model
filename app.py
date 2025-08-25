import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# -----------------------------
# Paths to your TFLite models
# -----------------------------
MRI_MODEL_PATH = "mri_detector_retrained.tflite"
TUMOR_MODEL_PATH = "tumor_classifier_roi.tflite"

TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]

# -----------------------------
# Load TFLite models
# -----------------------------
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

mri_interpreter = load_tflite_model(MRI_MODEL_PATH)
tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# -----------------------------
# Tumor Prediction
# -----------------------------
def predict_tumor(img):
    img_array = preprocess_image(img)
    tumor_pred = predict_tflite(tumor_interpreter, img_array)
    tumor_index = np.argmax(tumor_pred)
    tumor_label = TUMOR_CLASSES[tumor_index]
    tumor_conf = float(tumor_pred[tumor_index])
    
    # Summary table
    summary_df = pd.DataFrame({
        "Class": TUMOR_CLASSES,
        "Probability": [round(float(p), 4) for p in tumor_pred]
    }).sort_values(by="Probability", ascending=False)
    
    return tumor_label, tumor_conf, summary_df

# -----------------------------
# MRI Prediction
# -----------------------------
def predict_mri(img):
    img_array = preprocess_image(img)
    mri_pred = predict_tflite(mri_interpreter, img_array)
    mri_index = np.argmax(mri_pred)
    mri_classes = ["Brain MRI", "Other MRI", "Not MRI"]
    mri_label = mri_classes[mri_index]
    mri_conf = float(mri_pred[mri_index])
    return mri_label, mri_conf

# -----------------------------
# Sidebar Widgets
# -----------------------------
st.sidebar.header("User Widgets")

# Scan history
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

st.sidebar.subheader("Scan History")
for record in st.session_state.scan_history:
    st.sidebar.write(f"{record['timestamp']}: {record['tumor_label']} ({record['tumor_conf']:.2f})")

# Tumor info
st.sidebar.subheader("Tumor Information")
tumor_info_dict = {
    "Glioma": "Gliomas are tumors that arise from glial cells in the brain...",
    "Meningioma": "Meningiomas are tumors that arise from the meninges...",
    "Pituitary": "Pituitary tumors occur in the pituitary gland, affecting hormone levels..."
}
selected_tumor = st.sidebar.selectbox("Select Tumor Type", TUMOR_CLASSES)
st.sidebar.write(tumor_info_dict[selected_tumor])

# Hospital recommendation
st.sidebar.subheader("Find Hospitals")
location_input = st.sidebar.text_input("Enter your location for hospital recommendation")
if st.sidebar.button("Show Hospitals"):
    if location_input:
        # Placeholder hospitals
        hospitals = [f"Hospital {i} - {location_input}" for i in range(1,4)]
        for h in hospitals:
            st.sidebar.write(h)

# -----------------------------
# Main Panel
# -----------------------------
st.title("🧠 Brain Tumor Detection App")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # MRI prediction
    mri_label, mri_conf = predict_mri(img)
    st.write(f"**MRI Prediction:** {mri_label} (Confidence: {mri_conf:.2f})")
    
    # Only predict tumor if Brain MRI with confidence ≥ 0.8
    if mri_label == "Brain MRI" and mri_conf >= 0.8:
        tumor_label, tumor_conf, summary_df = predict_tumor(img)
        st.write(f"✅ **Tumor Prediction:** {tumor_label} (Confidence: {tumor_conf:.2f})")
        st.write("### Tumor Class Probabilities")
        st.dataframe(summary_df)
        
        # Save to scan history
        st.session_state.scan_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tumor_label": tumor_label,
            "tumor_conf": tumor_conf
        })
    else:
        st.warning("No tumor prediction. Image not recognized as Brain MRI.")
