import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('PCB_Multi_Label_Classifier_V2.h5')

# Streamlit interface
st.title("PCB Multi-Class Detection")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (224, 224)) / 255.0
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    predictions = model.predict(np.expand_dims(image_resized, axis=0))
    classes = ['Capacitor_SMD', 'Diode_SMD', 'IC_Chip', 'Inductor_SMD', 'Resistor_SMD']
    st.write("Predictions:")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {predictions[0][i]:.2f}")
