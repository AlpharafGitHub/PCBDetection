import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Class mapping for PCB components (update this with your actual PCB classes)
class_mapping = {
    0: 'Capacitor_SMD',
    1: 'Diode_SMD',
    2: 'IC_Chip',
    3: 'Inductor_SMD',
    4: 'Resistor_SMD',
}

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model_url = "https://raw.githubusercontent.com/AlpharafGitHub/PCBDetection/main/PCB_Multi_Label_Classifier.h5"
        model_path = "PCB_Multi_Label_Classifier.h5"
        
        if not os.path.exists(model_path):
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)

        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize the model
model = load_model()
if model is None:
    st.stop()

# Function to preprocess and make predictions
def predict(image, model):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_mapping[np.argmax(predictions[0])]
    return predicted_class

# Streamlit app layout
st.title("PCB Image Classifier")
st.subheader("Upload an image of a PCB for classification.")

uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

# Display the file size and type limit
st.markdown("Limit 200MB per file • JPG, JPEG, PNG")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded PCB Image", use_column_width=True)
    st.write("Classifying...")

    predicted_class = predict(image, model)
    st.write(f"Predicted Class: **{predicted_class}**")
