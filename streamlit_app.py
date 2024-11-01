import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os

# Class mapping (update this with your actual PCB classes)
class_mapping = {
    0: 'Capacitor_SMD',
    1: 'Diode_SMD',
    2: 'IC_Chip',
    3: 'Inductor_SMD',
    4: 'Resistor_SMD',
    }

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/AlpharafGitHub/PCBDetection/raw/main/PCB_Multi_Label_Classifier.h5"
    model_path = tf.keras.utils.get_file("pcb_model.h5", origin=model_url, cache_subdir=os.path.abspath("."))
    model_path = tf.keras.utils.get_file("PCB_Multi_Label_Classifier.h5", origin=model_url, cache_subdir=os.path.abspath("."))
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess and make predictions
def predict(image, model):
    # Preprocess the image
    image = image.resize((224, 224))  # Adjust this size if your model requires a different input size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class with the highest probability
    predicted_class = class_mapping[np.argmax(predictions[0])]
    return predicted_class

# Streamlit app
st.title("PCB Image Classifier")
uploaded_file = st.file_uploader("Upload an image of a PCB for classification.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
