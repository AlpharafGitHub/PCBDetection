import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the .h5 model
def load_model():
    model = tf.keras.models.load_model("PCB_Multi_Label_Classifier_V2.h5")
    return model

# Initialize the model
model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to RGB if it is in a different mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image to 224x224 and normalize to [0, 1]
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0).astype(np.float32)

# Function to run inference on the model
def predict(image):
    predictions = model.predict(image)
    return predictions

# Function to calculate detected classes and their percentages
def calculate_classes(predictions, threshold=0.5):
    classes = ['Capacitor_SMD', 'Diode_SMD', 'IC_Chip', 'Inductor_SMD', 'Resistor_SMD']
    detected_classes = []

    # Threshold the predictions to detect classes
    for i, cls in enumerate(classes):
        if predictions[0][i] >= threshold:
            detected_classes.append(cls)

    # Count occurrences of each class
    class_counts = {cls: 0 for cls in classes}
    for cls in detected_classes:
        class_counts[cls] += 1

    # Calculate total detections and percentage
    total_detected = len(detected_classes)
    class_percentages = {}
    if total_detected > 0:
        for cls in classes:
            class_percentages[cls] = (class_counts[cls] / total_detected) * 100

    return class_counts, class_percentages

# Streamlit UI
st.title("PCB Multi-Class Semiconductor Classifier for SMD")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Run prediction
    predictions = predict(processed_image)
    
    # Calculate detected classes and their percentages
    class_counts, class_percentages = calculate_classes(predictions)

    # Display annotations with detected classes, counts, and percentages
    st.write("### Detected Classes with Counts and Percentages:")
    for cls in class_counts:
        st.write(f"**{cls}**: Count = {class_counts[cls]}, Percentage = {class_percentages.get(cls, 0.0):.2f}%")
