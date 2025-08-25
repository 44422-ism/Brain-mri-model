import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# === Model paths ===
MRI_MODEL_PATH = "mri_detector_retrained (1).tflite"
TUMOR_MODEL_PATH = "tumor_classifier_roi (2).tflite"

# === Load TFLite models ===
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

mri_interpreter = load_tflite_model(MRI_MODEL_PATH)
tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]

# === Prediction function ===
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize input if needed
    input_shape = input_details[0]['shape']
    if img_array.shape != tuple(input_shape):
        img_array = np.resize(img_array, input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# === Streamlit App UI ===
st.title("🧠 Brain Tumor Classification + MRI Detector")
st.write("""
Upload MRI images to detect tumors and classify them as Brain MRI / Other MRI / Not MRI.  
⚠ Predictions are based on trained models and may not be 100% accurate. Always consult a medical professional.
""")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image for model
    img_resized = image.resize((224, 224))  # assuming your models use 224x224 input
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    
    # MRI Prediction
    mri_pred = predict_tflite(mri_interpreter, img_array)
    mri_label = MRI_CLASSES[np.argmax(mri_pred)]
    st.write(f"**MRI Prediction:** {mri_label} (Confidence: {np.max(mri_pred):.2f})")
    
    # Only run tumor prediction if Brain MRI
    if mri_label == "Brain MRI":
        tumor_pred = predict_tflite(tumor_interpreter, img_array)
        tumor_label = TUMOR_CLASSES[np.argmax(tumor_pred)]
        st.write(f"**Tumor Prediction:** {tumor_label} (Confidence: {np.max(tumor_pred):.2f})")
    else:
        st.write("No tumor prediction. Image not recognized as Brain MRI.")
