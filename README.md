This project builds a deep learning pipeline to classify brain MRI scans into four tumor categories using both a Custom CNN and Transfer Learning (ResNet50). It includes EDA, model evaluation, and a Streamlit web app for real-time predictions.

📂 Dataset Overview
Source: MRI brain scan dataset (folder-based)

Classes:

glioma

meningioma

pituitary

no_tumor

Image Format: JPG/PNG

Split: train/, valid/, test/

Image Shape: Resized to (224, 224, 3)

📊 Exploratory Data Analysis (EDA)
Visualized class distribution using bar plots

Displayed sample images from each class

Checked for:

Empty folders

Corrupted/unreadable images

Duplicate filenames

Image brightness and area consistency

🧠 Models Implemented
1️⃣ Custom CNN (from scratch)
3 convolutional blocks + dense layers

Dropout & BatchNormalization

Accuracy: ~baseline performance

2️⃣ Transfer Learning (ResNet50)
Pretrained on ImageNet

Fine-tuned top layers

Significantly higher validation accuracy

📈 Model Evaluation
Metrics: Accuracy, Classification Report, Confusion Matrix

Statistical Testing:

✅ Chi-Square – No significant class imbalance

✅ Levene’s Test – Image area variance is consistent

✅ ANOVA – Brightness consistency across classes

🧪 Technologies Used
Python, TensorFlow/Keras

Matplotlib, Seaborn, NumPy, PIL

Scikit-learn, SciPy

Google Colab

Streamlit (for Web App)

🚀 Streamlit Web App
Upload your brain MRI and get real-time tumor classification with confidence scores.

bash
Copy
Edit
streamlit run streamlit_app.py
🧠 Model Used: best_transfer_model.h5
📸 Input: MRI Image (JPG/PNG)
📊 Output: Tumor class + confidence score

📌 Project Structure
bash
Copy
Edit
📁 brain_tumor_dataset/
    └── Tumour/
        ├── train/
        ├── valid/
        └── test/
📄 brain_tumor_classification.ipynb
📄 streamlit_app.py
📄 best_transfer_model.h5
📄 README.md
✅ Conclusion
Transfer Learning with ResNet50 showed superior results and is recommended for deployment. This project demonstrates how deep learning can assist in accurate and real-time tumor detection from MRI scans.

✍️ Author
Mehek Pathan
Deep Learning & Computer Vision Enthusiast
