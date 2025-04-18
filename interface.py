import streamlit as st
import joblib
from keras.models import load_model
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from PIL import Image
import os
import glob

# -----------------------
# Load models/resources
# -----------------------
@st.cache_resource
def load_text_model():
    return joblib.load("fake_news_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def load_image_model():
    return load_model("img_model.keras")

@st.cache_resource
def load_esrgan_model():
    return hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

# -----------------------
# Image preprocessing functions
# -----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dimension = (256, 256)

def srmodel(img, model):
    image_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(img, 0, 0, image_size[0], image_size[1])
    preprocessed_image = tf.cast(cropped_image, tf.float32) / 255.0
    preprocessed_image = tf.expand_dims(preprocessed_image, 0)
    new_image = model(preprocessed_image)
    return tf.squeeze(new_image)

def preprocess_single_image(image_path, sr_model):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        raise ValueError("No faces detected.")

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    if w >= 256 and h >= 256:
        resized_face = cv2.resize(face_rgb, dimension, interpolation=cv2.INTER_AREA)
    else:
        face_tensor = tf.convert_to_tensor(face_rgb, dtype=tf.float32)
        sr_face = srmodel(face_tensor, sr_model)
        sr_face = (sr_face.numpy() * 255).astype(np.uint8)
        resized_face = cv2.resize(sr_face, dimension, interpolation=cv2.INTER_CUBIC)

    resized_face = resized_face.astype(np.float32) / 255.0
    return np.expand_dims(resized_face, axis=0)  # Shape: (1, 256, 256, 3)

# -----------------------
# Prediction functions
# -----------------------
def predict_news(text, model, vectorizer):
    text_input = vectorizer.transform([text])
    prob = model.predict_proba(text_input)[0][1]
    label = "Fake" if prob > 0.5 else "Real"
    return label, prob

def predict_image(image_path, model, sr_model):
    try:
        img_array = preprocess_single_image(image_path, sr_model)
        pred = model.predict(img_array)
        prob = pred[0][1]  # assumes softmax; change to [0] if sigmoid
        label = "Fake" if prob > 0.5 else "Real"
        return label, prob
    except Exception as e:
        return "Error", 0.0

# -----------------------
# Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title="Real or Fake Detector", layout="centered")
    st.title("🧠 Real or Fake? Multimodal Detector")

    # News input
    news_text = st.text_area("📰 Enter News Text")

    # Image input (Local path browser)
    st.markdown("### 📂 Choose Image From Local Folder")
    folder_path = st.text_input("Enter image folder path", value=os.getcwd())
    image_path = None

    if os.path.isdir(folder_path):
        image_files = glob.glob(os.path.join(folder_path, "*.jp*g"))
        if image_files:
            image_path = st.selectbox("Select an image:", image_files)
        else:
            st.warning("No JPG/JPEG images found in this folder.")

    if st.button("🔍 Predict"):
        if not news_text or not image_path:
            st.warning("Please provide both news text and select a valid image.")
            return

        # Load models
        text_model = load_text_model()
        vectorizer = load_vectorizer()
        image_model = load_image_model()
        sr_model = load_esrgan_model()

        # Predict
        news_label, news_prob = predict_news(news_text, text_model, vectorizer)
        img_label, img_prob = predict_image(image_path, image_model, sr_model)

        # Display Results
        st.subheader("📰 News Prediction")
        st.write(f"Prediction: **{news_label}** ({news_prob:.2%} fake probability)")

        st.subheader("🖼️ Image Prediction")
        if img_label == "Error":
            st.error("Face not detected or image could not be processed.")
        else:
            st.write(f"Prediction: **{img_label}** ({img_prob:.2%} fake probability)")
            st.image(image_path, caption="Selected Image", use_column_width=True)
if __name__ == "__main__":
    main()
