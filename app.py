<<<<<<< HEAD
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
=======
# ==========================
# Brain Tumor + MRI Detection App
# ==========================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import streamlit as st
import pandas as pd
>>>>>>> f206414 (Initial commit for Streamlit app)

# -------------------------
# Constants
# -------------------------
<<<<<<< HEAD
TUMOR_MODEL_PATH = "best_model.tflite"
MRI_MODEL_PATH = "mri_detector_model.tflite"
IMG_SIZE = (224, 224)
CLASS_LABELS = ["Glioma", "Meningioma", "Pituitary"]

# -------------------------
# Load TFLite Models
# -------------------------
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

tumor_interpreter, tumor_input, tumor_output = load_tflite_model(TUMOR_MODEL_PATH)
mri_interpreter, mri_input, mri_output = load_tflite_model(MRI_MODEL_PATH)

st.success("✅ TFLite models loaded successfully!")

# -------------------------
# Helper Functions
# -------------------------
def preprocess_image(img: Image.Image):
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_tflite(interpreter, input_details, output_details, img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]
=======
TUMOR_MODEL_PATH = "best_model.h5"           # Brain tumor classifier
MRI_DETECTOR_PATH = "mri_detector_model.h5"  # MRI detector
IMG_SIZE = (224, 224)
CLASS_LABELS = ["Glioma", "Meningioma", "Pituitary"]  # Brain tumor classes
>>>>>>> f206414 (Initial commit for Streamlit app)

# -------------------------
# Streamlit UI
# -------------------------
st.title("Brain Tumor MRI Classification + MRI Detector")
<<<<<<< HEAD
st.write("Upload multiple images. Non-MRI images will be skipped automatically.")

uploaded_files = st.file_uploader(
    "Upload Image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True
)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            img_array = preprocess_image(img)

            # MRI Detection
            mri_pred = float(predict_tflite(mri_interpreter, mri_input, mri_output, img_array)[0])
            if mri_pred > 0.5:  # MRI detected
                # Tumor Prediction
                tumor_pred = predict_tflite(tumor_interpreter, tumor_input, tumor_output, img_array)
                class_idx = int(np.argmax(tumor_pred))
                confidence = float(tumor_pred[class_idx])
                st.success(f"Predicted Tumor Type: {CLASS_LABELS[class_idx]}")
                st.info(f"Confidence: {confidence:.2f}")
                results.append((uploaded_file.name, "MRI", CLASS_LABELS[class_idx], confidence))
            else:
                st.warning(f"⚠ {uploaded_file.name}: Non-MRI / Random object")
                results.append((uploaded_file.name, "Non-MRI", "N/A", 0.0))

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            results.append((uploaded_file.name, "Error", "N/A", 0.0))

    # Summary Table
    if results:
        df = pd.DataFrame(results, columns=["File Name", "MRI Status", "Tumor Type", "Confidence"])
        st.subheader("Summary of Predictions")
        st.dataframe(df)
=======
st.write(
    "Upload multiple images (brain MRI, other body MRI, non-MRI, random objects) "
    "for detection and classification."
)

# -------------------------
# Load Models
# -------------------------
model_loaded = False
if os.path.exists(TUMOR_MODEL_PATH) and os.path.exists(MRI_DETECTOR_PATH):
    tumor_model = tf.keras.models.load_model(TUMOR_MODEL_PATH)
    mri_detector = tf.keras.models.load_model(MRI_DETECTOR_PATH)
    st.success("✅ Models loaded successfully!")
    model_loaded = True
else:
    st.error(
        "❌ Model files not found. Upload 'best_model.h5' and 'mri_detector_model.h5' "
        "in the same directory as this script."
    )

# -------------------------
# Helper Function
# -------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    img_resized = img.resize(IMG_SIZE)
    img_array = img_to_array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------------
# Batch Upload & Prediction
# -------------------------
if model_loaded:
    uploaded_files = st.file_uploader(
        "Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                img_array = preprocess_image(img)

                # MRI Detection
                mri_pred = float(mri_detector.predict(img_array)[0][0])
                if mri_pred > 0.5:
                    st.success(f"✅ {uploaded_file.name}: MRI detected")

                    # Brain Tumor Classification
                    tumor_pred = tumor_model.predict(img_array)[0]
                    class_idx = int(np.argmax(tumor_pred))
                    confidence = float(tumor_pred[class_idx])

                    st.success(f"Predicted Tumor Type: {CLASS_LABELS[class_idx]}")
                    st.info(f"Confidence: {confidence:.2f}")
                    results.append((uploaded_file.name, CLASS_LABELS[class_idx], confidence))
                else:
                    st.warning(f"⚠ {uploaded_file.name}: Not an MRI")
                    results.append((uploaded_file.name, "Not MRI", mri_pred))

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                results.append((uploaded_file.name, "Error", 0.0))

        # Summary Table
        if results:
            df = pd.DataFrame(results, columns=["File Name", "Prediction", "Confidence"])
            st.subheader("Summary of Predictions")
            st.dataframe(df)

# -------------------------
# Note for Users
# -------------------------
st.write(
    "ℹ Note: The model's predictions are based on training data and may not be 100% accurate."
)
>>>>>>> f206414 (Initial commit for Streamlit app)
