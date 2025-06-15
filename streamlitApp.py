import streamlit as st
import io
import cv2
import numpy as np
import os
from PIL import Image
import tensorflow as tf # Import TensorFlow for Teachable Machine model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly - Bottle Anomaly", page_icon=":camera:")

st.title("InspectorsAlly - Bottle Anomaly Detection")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App for Bottles"
)

st.write(
    "Try clicking a bottle image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg") # Assuming you have a relevant image for bottles here
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections for bottles. With InspectorsAlly, companies can ensure that their bottle products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )

    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as cracks, deformities, foreign objects, and more on Bottle Product Images."
    )

# Define the functions to load images
def load_uploaded_image(file):
    img = Image.open(file)
    return img

# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

# Check which input method was selected
uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# --- Teachable Machine Model Integration ---
# Path to your Teachable Machine Keras H5 model
# Make sure to replace this with the actual path to your exported model
MODEL_PATH = "./teachable_machine_models/bottle_anomaly_model.h5"
# Make sure these class names match the order in your Teachable Machine model
CLASS_NAMES = ["Anomaly", "Good"] # Adjust based on your Teachable Machine output order


@st.cache_resource
def load_teachable_machine_model():
    """Loads the Teachable Machine Keras H5 model."""
    try:
        model = tf.keras.models.load_model("./keras_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading Teachable Machine model: {e}")
        st.stop()

model = load_teachable_machine_model()

def preprocess_image_for_teachable_machine(image_pil):
    """
    Preprocesses a PIL image for a Teachable Machine model.
    Teachable Machine models typically expect images scaled to 224x224 and normalized.
    """
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = image_pil.resize((224, 224))
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 # Normalize to [-1, 1]
    data[0] = normalized_image_array
    return data


def Anomaly_Detection_TeachableMachine(image_pil):
    """
    Given a PIL image and a loaded Teachable Machine model, returns the predicted class.
    """
    if model is None:
        return "Model not loaded. Please check the model path."

    # Preprocess the image
    processed_image = preprocess_image_for_teachable_machine(image_pil)

    # Get the model's predictions
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    prediction_sentence = ""
    if predicted_class == "Good":
        prediction_sentence = f"Congratulations! Your bottle product has been classified as a **'Good'** item with no anomalies detected in the inspection images. (Confidence: {confidence_score:.2f})"
    else: # Assuming "Anomaly" is the other class
        prediction_sentence = f"We're sorry to inform you that our AI-based visual inspection system has detected an **anomaly** in your bottle product. (Confidence: {confidence_score:.2f})"

    return prediction_sentence

# --- End Teachable Machine Model Integration ---

submit = st.button(label="Submit a Bottle Product Image")
if submit:
    st.subheader("Output")
    img_to_predict = None
    if input_method == "File Uploader" and uploaded_file_img is not None:
        img_to_predict = uploaded_file_img
    elif input_method == "Camera Input" and camera_file_img is not None:
        img_to_predict = camera_file_img
    else:
        st.warning("Please upload or capture an image first.")

    if img_to_predict is not None:
        with st.spinner(text="Analyzing image for anomalies..."):
            prediction = Anomaly_Detection_TeachableMachine(img_to_predict)
            st.write(prediction)