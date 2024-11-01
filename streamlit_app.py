import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import os

@st.cache_resource  # Updated decorator for caching resources like models
def load_model():
    try:
        model_url = "https://raw.githubusercontent.com/AlpharafGitHub/PCBDetection/main/PCB_Multi_Label_Classifier.h5"
        response = requests.get(model_url)
        response.raise_for_status()  # Checks for request errors
        model_path = "PCB_Multi_Label_Classifier.h5"
        
        # Save model file locally
        with open(model_path, "wb") as file:
            file.write(response.content)
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model and check if it was successful
model = load_model()
if model is None:
    st.stop()  # Stops the app if the model failed to load

st.title("PCB Defect Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Ensure the model exists before prediction
    if model:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        st.write(f"Predicted class: {predicted_class}")
