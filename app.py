import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

# -------------------------------
# Constants
# -------------------------------
TUMOR_MODEL_PATH = "tumor_classifier_roi (2).tflite"
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

TUMOR_INFO = {
    "Glioma": "Gliomas are tumors that arise from glial cells in the brain or spine.",
    "Meningioma": "Meningiomas develop from the meninges, the protective membranes covering the brain and spinal cord.",
    "Pituitary": "Pituitary tumors form in the pituitary gland at the base of the brain.",
    "No Tumor": "No tumor detected in this scan."
}

CONF_THRESHOLD = 0.70  # Minimum confidence to accept prediction

# -------------------------------
# Load TFLite Model
# -------------------------------
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)
input_details = tumor_interpreter.get_input_details()
output_details = tumor_interpreter.get_output_details()

# -------------------------------
# Predict Tumor
# -------------------------------
def predict_tumor(image: Image.Image):
    img = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img_array = np.array(img, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:  # Grayscale image
        img_array = np.stack((img_array,)*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    tumor_interpreter.set_tensor(input_details[0]['index'], img_array)
    tumor_interpreter.invoke()
    tumor_pred = tumor_interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Normalize probabilities to mitigate bias
    tumor_pred = tumor_pred / np.sum(tumor_pred)
    
    max_index = np.argmax(tumor_pred)
    tumor_conf = float(tumor_pred[max_index])
    
    # Confidence thresholding
    if tumor_conf < CONF_THRESHOLD:
        tumor_label = "Uncertain – Needs further review"
    else:
        tumor_label = TUMOR_CLASSES[max_index]
    
    # Summary Table
    summary_df = pd.DataFrame({
        "Class": TUMOR_CLASSES,
        "Probability": [round(float(p), 4) for p in tumor_pred]
    }).sort_values(by="Probability", ascending=False)
    
    return tumor_label, tumor_conf, summary_df

# -------------------------------
# Session State for Scan History
# -------------------------------
if "scan_history" not in st.session_state:
    st.session_state["scan_history"] = []

# -------------------------------
# App UI
# -------------------------------
st.title("🧠 Brain Tumor Classification")
st.markdown("""
Upload MRI scans to detect tumors. Predictions are **not 100% accurate** and should be verified by a medical professional.
""")

uploaded_file = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)
    
    # Tumor Prediction
    tumor_label, tumor_conf, summary_df = predict_tumor(image)
    
    st.markdown(f"### ✅ Tumor Prediction: {tumor_label} (Confidence: {round(tumor_conf,2)})")
    
    # Display Summary Table
    st.markdown("### Probability Summary")
    st.dataframe(summary_df)
    
    # Visualization (optional)
    st.bar_chart(summary_df.set_index("Class"))

    # Save scan history
    st.session_state.scan_history.append({
        "Filename": uploaded_file.name,
        "Predicted Tumor": tumor_label,
        "Confidence": round(tumor_conf, 2),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# -------------------------------
# Scan History
# -------------------------------
if st.session_state.scan_history:
    st.markdown("### 🗂 Scan History")
    history_df = pd.DataFrame(st.session_state.scan_history)
    st.dataframe(history_df)

# -------------------------------
# Tumor Info
# -------------------------------
st.markdown("### 📚 Tumor Information")
for tumor, info in TUMOR_INFO.items():
    st.markdown(f"**{tumor}:** {info}")
