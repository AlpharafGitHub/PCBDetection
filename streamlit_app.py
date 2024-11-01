import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load TFLite model using TensorFlow's interpreter
def load_model():
    interpreter = tf.lite.Interpreter(model_path="PCB_Multi_Label_Classifier_V2.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Initialize the model
interpreter = load_model()

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to RGB if it is in a different mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image to 224x224 and normalize to [0, 1]
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0).astype(np.float32)

# Function to run inference on the TFLite model
def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    return predictions

# Streamlit UI
st.title("PCB Multi-Class Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Run prediction
    predictions = predict(processed_image)
    
    # Define class labels
    classes = ['Capacitor_SMD', 'Diode_SMD', 'IC_Chip', 'Inductor_SMD', 'Resistor_SMD']
    
    # Display predictions
    st.write("Predictions:")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {predictions[0][i]:.2f}")
