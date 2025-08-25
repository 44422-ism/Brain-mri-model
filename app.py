import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os

# ----------------------
# App Config
# ----------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

TUMOR_MODEL_PATH = "tumor_classifier_roi (2).tflite"  # exact filename
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]  # adjust to your model

# Sidebar storage for scan history
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

# Tumor info for sidebar
TUMOR_INFO = {
    "Glioma": "Gliomas are tumors that start in the glial cells of the brain.",
    "Meningioma": "Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord.",
    "Pituitary": "Pituitary tumors develop in the pituitary gland and can affect hormone levels."
}

# Hospital recommendations by location (dummy example)
HOSPITALS = {
    "New York": ["NY Brain Center", "Mount Sinai Hospital", "NYC Medical College"],
    "London": ["London Brain Institute", "St Thomas Hospital", "Royal London Hospital"],
    "Tokyo": ["Tokyo Neurology Center", "St Luke's Hospital", "Tokyo Medical University Hospital"]
}

# ----------------------
# Load TFLite model
# ----------------------
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# ----------------------
# Prediction function
# ----------------------
def predict_tumor(image: Image):
    img = image.resize((224, 224))  # adjust if your model has different input size
    img_array = np.array(img, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:  # grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    input_data = np.expand_dims(img_array, axis=0)

    input_details = tumor_interpreter.get_input_details()
    output_details = tumor_interpreter.get_output_details()
    tumor_interpreter.set_tensor(input_details[0]['index'], input_data)
    tumor_interpreter.invoke()
    tumor_pred = tumor_interpreter.get_tensor(output_details[0]['index'])[0]

    # Find top prediction
    top_idx = int(np.argmax(tumor_pred))
    tumor_label = TUMOR_CLASSES[top_idx]
    tumor_conf = float(tumor_pred[top_idx])

    # Create summary table
    summary_df = pd.DataFrame({
        "Class": TUMOR_CLASSES,
        "Probability": [round(float(p), 4) for p in tumor_pred]
    }).sort_values(by="Probability", ascending=False)

    return tumor_label, tumor_conf, summary_df

# ----------------------
# Main App Layout
# ----------------------
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan to detect tumor type. ⚠ Note: Predictions are based on trained models and may not be 100% accurate.")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    tumor_label, tumor_conf, summary_df = predict_tumor(image)

    # Save to scan history
    st.session_state.scan_history.append({
        "filename": uploaded_file.name,
        "prediction": tumor_label,
        "confidence": round(tumor_conf, 4)
    })

    # Display results
    st.subheader("Prediction Result")
    st.write(f"✅ Tumor Prediction: **{tumor_label}** (Confidence: {round(tumor_conf, 2)})")
    st.subheader("Probability Summary")
    st.dataframe(summary_df, use_container_width=True)

# ----------------------
# Sidebar Widgets
# ----------------------
st.sidebar.header("User Panel")

# Scan History
st.sidebar.subheader("📋 Scan History")
if st.session_state.scan_history:
    for i, scan in enumerate(st.session_state.scan_history[::-1]):
        st.sidebar.write(f"{i+1}. {scan['filename']} → {scan['prediction']} ({scan['confidence']})")
else:
    st.sidebar.write("No scans yet.")

# Tumor Info
st.sidebar.subheader("ℹ Tumor Information")
selected_tumor = st.sidebar.selectbox("Select Tumor Type", TUMOR_CLASSES)
st.sidebar.write(TUMOR_INFO[selected_tumor])

# Hospital Recommendations
st.sidebar.subheader("🏥 Hospital Recommendations")
location = st.sidebar.text_input("Enter Your Location")
if location and location in HOSPITALS:
    st.sidebar.write("Recommended Hospitals:")
    for h in HOSPITALS[location]:
        st.sidebar.write(f"- {h}")
elif location:
    st.sidebar.write("No hospitals found for this location.")
