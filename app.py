import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the trained model
model = load_model("enhanced_corn_cotton_disease_model.h5")

# Class names (update based on your training dataset structure)
class_names = sorted(os.listdir("data/train"))  # Dynamically fetch class names

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input size
    image = img_to_array(image)      # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0            # Normalize to 0-1 range
    return image

# Streamlit app
st.title("Corn and Cotton Disease Prediction")
st.write("Upload an image of a corn or cotton leaf to detect diseases.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Display class probabilities (optional)
    st.write("**Class Probabilities:**")
    probabilities = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.json(probabilities)
