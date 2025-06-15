import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf # Keep tensorflow, but we'll use tf.lite

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
    # Ensure this path is correct for your deployed app
    img = Image.open("./docs/overview_dataset.jpg")
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

# --- Teachable Machine TFLite Model Integration ---
# Paths to your Teachable Machine TFLite model and labels
TFLITE_MODEL_PATH = "./model_unquant.tflite" # Update this path!
LABELS_PATH = "./labels.txt"     # Update this path!

CLASS_NAMES = []
try:
    with open(LABELS_PATH, "r") as f:
        # Labels.txt typically has "0 Anomaly", "1 Good". We want just "Anomaly", "Good"
        CLASS_NAMES = [line.strip().split(" ", 1)[1] for line in f.readlines()]
except FileNotFoundError:
    st.error(f"Labels file not found at {LABELS_PATH}. Please ensure it's in your repository.")
    st.stop()
except Exception as e:
    st.error(f"Error reading labels file: {e}")
    st.stop()

@st.cache_resource
def load_tflite_model_and_interpreter():
    """Loads the TensorFlow Lite model and initializes interpreter."""
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TensorFlow Lite model: {e}")
        st.stop()

tflite_interpreter = load_tflite_model_and_interpreter()

def preprocess_image_for_teachable_machine_tflite(image_pil):
    """
    Preprocesses a PIL image for a Teachable Machine TFLite model.
    It handles resizing to the model's expected input shape and normalization.
    """
    # Get input details from the interpreter to determine expected shape and type
    input_details = tflite_interpreter.get_input_details()
    input_shape = input_details[0]['shape'] # e.g., [1, 224, 224, 3]
    input_height, input_width = input_shape[1], input_shape[2]
    input_dtype = input_details[0]['dtype']

    # Resize the image
    image = image_pil.resize((input_width, input_height))
    image_array = np.asarray(image)

    # Normalize the image based on the expected input type
    if input_dtype == np.float32:
        # Teachable Machine's float32 models expect normalization to [-1, 1]
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1 # Corrected normalization if Teachable Machine uses 127.5
    elif input_dtype == np.uint8:
        # For quantized models, they often expect uint8 (0-255) directly
        normalized_image_array = image_array.astype(np.uint8)
    else:
        st.error(f"Unsupported input dtype for TFLite model: {input_dtype}")
        st.stop()

    # Add batch dimension (e.g., from [224, 224, 3] to [1, 224, 224, 3])
    input_data = np.expand_dims(normalized_image_array, axis=0)
    return input_data


def Anomaly_Detection_TeachableMachine_TFLite(image_pil):
    """
    Given a PIL image and a loaded TFLite interpreter, returns the predicted class.
    """
    if tflite_interpreter is None:
        return "Model interpreter not loaded. Please check the model path."

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    # Preprocess the image
    processed_image = preprocess_image_for_teachable_machine_tflite(image_pil)

    # Set the tensor
    tflite_interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    tflite_interpreter.invoke()

    # Get the output tensor
    output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0] # Get the predictions for the first (and only) image in the batch

    # If the model is quantized, the output might be integers and need dequantization
    # For Float32 models, output is usually probabilities directly.
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        prediction = (prediction.astype(np.float32) - zero_point) * scale

    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence_score = prediction[predicted_class_index]

    prediction_sentence = ""
    if predicted_class == "Good":
        prediction_sentence = f"Congratulations! Your bottle product has been classified as a **'Good'** item with no anomalies detected in the inspection images. (Confidence: {confidence_score:.2f})"
    else: # Assuming "Anomaly" is the other class
        prediction_sentence = f"We're sorry to inform you that our AI-based visual inspection system has detected an **anomaly** in your bottle product. (Confidence: {confidence_score:.2f})"

    return prediction_sentence

# --- End Teachable Machine TFLite Model Integration ---

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
            prediction = Anomaly_Detection_TeachableMachine_TFLite(img_to_predict)
            st.write(prediction)