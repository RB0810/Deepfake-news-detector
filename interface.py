import streamlit as st
import joblib
import tempfile
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
def preprocessing(img):
    """Prepare image for ESRGAN"""
    if img.shape[0] < 4 or img.shape[1] < 4:
        raise ValueError("Image too small for ESRGAN")

    image_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(img, 0, 0, image_size[0], image_size[1])
    preprocessed_image = tf.cast(cropped_image, tf.float32) / 255.0
    return tf.expand_dims(preprocessed_image, 0)

def srmodel(img):
    """Apply super resolution using ESRGAN"""
    preprocessed_image = preprocessing(img)
    esgran = load_esrgan_model()
    new_image = esgran(preprocessed_image)
    return tf.squeeze(new_image)

def preprocess_single_image(image_path, save_path=None):
    """Preprocess one image and optionally save it"""
    print("Debug line 1")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        raise ValueError("No faces detected.")

    # Select largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    dimension = (256, 256)

    try:
        if w >= 256 and h >= 256:
            resized_face = cv2.resize(face_rgb, dimension, interpolation=cv2.INTER_AREA)
        else:
            face_tensor = tf.convert_to_tensor(face_rgb, dtype=tf.float32)
            sr_face = srmodel(face_tensor)
            sr_face = (sr_face.numpy() * 255).astype(np.uint8)
            resized_face = cv2.resize(sr_face, dimension, interpolation=cv2.INTER_CUBIC)

        print("Debug line 2")
        if save_path:
            Image.fromarray(resized_face).save(save_path)
            print(f"Saved preprocessed image to {save_path}")
        return resized_face

    except Exception as e:
        raise RuntimeError(f"Error during preprocessing: {e}")
# -----------------------
# Prediction functions
# -----------------------
def predict_news(text, model, vectorizer):
    text_input = vectorizer.transform([text])
    prob = model.predict_proba(text_input)[0][1]
    label = "Fake" if prob > 0.5 else "Real"
    return label, prob

def predict_image(image_path, model):
    try:
        img_array = preprocess_single_image(image_path)
        img_input = np.expand_dims(img_array, axis=0)  # shape: (1, 256, 256, 3)
        img_input = img_input / 255.0  # normalize

        print("Debug line 3")
        image_pred = model.predict(img_input)
        predicted_class = np.argmax(image_pred[0])
        confidence = image_pred[0][predicted_class]

        label = "Real" if predicted_class == 0 else "Fake"
        print(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")

        return label, confidence * 100
    except Exception as e:
        return "Error", 0.0

# -----------------------
# Streamlit App
# -----------------------

def main():
    st.set_page_config(page_title="Real or Fake Detector", layout="centered")
    st.title("📰 Real or Fake News?")

    # News input
    news_text = st.text_area("Enter News Text")

    # Image input using file uploader
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if st.button("🔍 Predict"):
        if not news_text or not uploaded_image:
            st.warning("Please provide both news text and upload a valid image.")
            return

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_image.read())
            image_path = tmp_file.name

        # Load models
        text_model = load_text_model()
        vectorizer = load_vectorizer()
        image_model = load_image_model()

        # Predict
        news_label, news_prob = predict_news(news_text, text_model, vectorizer)
        img_label, img_prob = predict_image(image_path, image_model)

        # Display Results
        st.subheader("News Prediction")
        st.write(f"Prediction: **{news_label}** ({news_prob:.2%} fake probability)")

        st.subheader("Image Prediction")
        if img_label == "Error":
            st.error("Face not detected or image could not be processed.")
        else:
            st.write(f"Prediction: **{img_label}** ({img_prob:.2%} probability)")

if __name__ == "__main__":
    main()
