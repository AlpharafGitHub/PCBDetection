import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model
def load_model():
    interpreter = tflite.Interpreter(model_path="PCB_Multi_Label_Classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Make a prediction
def make_prediction(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess input and set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit App Interface
st.title("PCB Multi-Label Classifier")
st.write("Upload an image and get predictions from the model.")

# Load model once when the app starts
interpreter = load_model()

# Placeholder for input data
input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Dummy data, replace as needed

# Button to make a prediction
if st.button("Make Prediction"):
    prediction = make_prediction(interpreter, input_data)
    st.write("Prediction:", prediction)
