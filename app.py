import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- CONFIG --------------------
st.title("🧠 Brain Tumor Classification")
st.write("Upload MRI images to classify tumors: **Glioma**, **Meningioma**, or **Pituitary**.")
st.warning("⚠ Predictions are based on trained models and may not be 100% accurate. Always consult a medical professional.")

# Model path
TUMOR_MODEL_PATH = "tumor_classifier_roi (2).tflite"

# -------------------- LOAD TFLITE MODEL --------------------
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# -------------------- FUNCTIONS --------------------
def preprocess_image(image, input_shape):
    """Resize and normalize image for TFLite model."""
    img = image.convert("RGB")
    img_resized = img.resize((input_shape[1], input_shape[2]))  # e.g., (224, 224)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tumor(image):
    input_details = tumor_interpreter.get_input_details()
    output_details = tumor_interpreter.get_output_details()
    
    img_array = preprocess_image(image, input_details[0]['shape'])
    tumor_interpreter.set_tensor(input_details[0]['index'], img_array)
    tumor_interpreter.invoke()
    tumor_pred = tumor_interpreter.get_tensor(output_details[0]['index'])[0]
    
    classes = ["Glioma", "Meningioma", "Pituitary"]
    tumor_label = classes[np.argmax(tumor_pred)]
    tumor_conf = float(np.max(tumor_pred))
    return tumor_label, tumor_conf

# -------------------- UI --------------------
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Tumor"):
        with st.spinner("Analyzing..."):
            tumor_label, tumor_conf = predict_tumor(image)
            st.success(f"✅ Tumor Prediction: **{tumor_label}** (Confidence: {tumor_conf:.2f})")
