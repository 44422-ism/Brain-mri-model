import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detector", layout="wide")
MODEL_PATH = "tumor_classifier_roi (2).tflite"

# Tumor information for info widget
TUMOR_INFO = {
    "Glioma": "Gliomas are tumors that start in the glial cells of the brain or spinal cord.",
    "Meningioma": "Meningiomas are tumors that form in the membranes that surround the brain and spinal cord.",
    "Pituitary": "Pituitary tumors are abnormal growths in the pituitary gland, located at the base of the brain.",
    "No Tumor": "No tumor detected in the scan."
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

tumor_interpreter = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image, input_shape):
    img = image.resize((input_shape[2], input_shape[1]))
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_tumor(image):
    input_details = tumor_interpreter.get_input_details()
    output_details = tumor_interpreter.get_output_details()

    img_array = preprocess_image(image, input_details[0]['shape'])
    tumor_interpreter.set_tensor(input_details[0]['index'], img_array)
    tumor_interpreter.invoke()
    tumor_pred = tumor_interpreter.get_tensor(output_details[0]['index'])[0]

    if len(tumor_pred) == 1:  # Binary Model
        prob = float(tumor_pred[0])
        tumor_label = "Tumor Detected" if prob >= 0.5 else "No Tumor Detected"
        confidence = round(prob if prob >= 0.5 else 1 - prob, 4)
        summary_df = pd.DataFrame({
            "Prediction": [tumor_label],
            "Confidence": [confidence]
        })
        return tumor_label, confidence, summary_df
    else:  # Multi-class Model
        classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        tumor_label = classes[np.argmax(tumor_pred)]
        tumor_conf = float(np.max(tumor_pred))
        summary_df = pd.DataFrame({
            "Class": classes,
            "Probability": [round(float(p), 4) for p in tumor_pred.tolist()]
        }).sort_values(by="Probability", ascending=False)
        return tumor_label, tumor_conf, summary_df

# -----------------------------
# SCAN HISTORY STORAGE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# APP UI
# -----------------------------
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload MRI images to detect tumors. ⚠ Predictions may not be 100% accurate. Consult a medical professional.")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Analyzing image..."):
        tumor_label, tumor_conf, summary_df = predict_tumor(image)

    # Display Results
    st.subheader("✅ Prediction Result")
    st.write(f"**Tumor Prediction:** {tumor_label} (Confidence: {tumor_conf:.2f})")

    st.subheader("📊 Prediction Summary")
    st.dataframe(summary_df)

    if len(summary_df) > 1:
        st.bar_chart(summary_df.set_index("Class"))

    # Save to history
    st.session_state.history.append({
        "Image": uploaded_file.name,
        "Prediction": tumor_label,
        "Confidence": round(tumor_conf, 4)
    })

# -----------------------------
# HISTORY SECTION
# -----------------------------
st.sidebar.header("📜 Scan History")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(hist_df)
else:
    st.sidebar.write("No scans yet.")

# -----------------------------
# TUMOR INFORMATION SECTION
# -----------------------------
st.sidebar.header("ℹ Tumor Information")
selected_info = st.sidebar.selectbox("Select Tumor Type", list(TUMOR_INFO.keys()))
st.sidebar.write(TUMOR_INFO[selected_info])
