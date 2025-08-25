import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

# ----------------------
# Constants
# ----------------------
TUMOR_MODEL_PATH = "tumor_classifier_roi (2).tflite"
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]  # Adjust if different

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
    img = image.resize((224, 224))  # adjust to your model input size
    img_array = np.array(img, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:  # grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    input_data = np.expand_dims(img_array, axis=0)

    input_details = tumor_interpreter.get_input_details()
    output_details = tumor_interpreter.get_output_details()
    tumor_interpreter.set_tensor(input_details[0]['index'], input_data)
    tumor_interpreter.invoke()
    tumor_pred = tumor_interpreter.get_tensor(output_details[0]['index'])[0]

    # Handle mismatch
    if len(tumor_pred) != len(TUMOR_CLASSES):
        st.warning(f"Model output length ({len(tumor_pred)}) does not match number of classes ({len(TUMOR_CLASSES)}).")
        min_len = min(len(tumor_pred), len(TUMOR_CLASSES))
        tumor_pred = tumor_pred[:min_len]
        classes = TUMOR_CLASSES[:min_len]
    else:
        classes = TUMOR_CLASSES

    top_idx = int(np.argmax(tumor_pred))
    tumor_label = classes[top_idx]
    tumor_conf = float(tumor_pred[top_idx])

    summary_df = pd.DataFrame({
        "Class": classes,
        "Probability": [round(float(p), 4) for p in tumor_pred]
    }).sort_values(by="Probability", ascending=False)

    return tumor_label, tumor_conf, summary_df

# ----------------------
# Sidebar: Scan history
# ----------------------
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

st.sidebar.header("Scan History")
for i, entry in enumerate(st.session_state.scan_history[::-1], 1):
    st.sidebar.write(f"{i}. {entry}")

# ----------------------
# Sidebar: Tumor Info
# ----------------------
st.sidebar.header("Tumor Info")
tumor_info = {
    "Glioma": "Gliomas are brain tumors that originate from glial cells...",
    "Meningioma": "Meningiomas arise from the meninges and are usually benign...",
    "Pituitary": "Pituitary tumors develop in the pituitary gland and can affect hormone levels..."
}
st.sidebar.write("Select tumor type to see info:")
selected_tumor = st.sidebar.selectbox("Tumor Type", TUMOR_CLASSES)
st.sidebar.write(tumor_info.get(selected_tumor, "Info not available."))

# ----------------------
# Sidebar: Hospital Recommendations
# ----------------------
st.sidebar.header("Hospital Recommendations")
user_location = st.sidebar.text_input("Enter your city or location:")
if user_location:
    st.sidebar.write(f"Recommended hospitals in {user_location}:")
    # Placeholder, ideally replace with actual API or dataset
    st.sidebar.write("- National Brain Center")
    st.sidebar.write("- Advanced Neuro Hospital")
    st.sidebar.write("- City Medical Institute")

# ----------------------
# Main App
# ----------------------
st.title("🧠 Brain Tumor Classification")
st.write("Upload your MRI scan to detect tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    tumor_label, tumor_conf, summary_df = predict_tumor(image)
    st.success(f"Tumor Prediction: {tumor_label} (Confidence: {tumor_conf:.2f})")
    st.write("Prediction Summary:")
    st.dataframe(summary_df)

    # Add to scan history
    st.session_state.scan_history.append(f"{tumor_label} - {tumor_conf:.2f}")
