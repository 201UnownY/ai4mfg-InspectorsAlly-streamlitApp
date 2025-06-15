import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the Teachable Machine model
model = tf.keras.models.load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").read().splitlines()

# Page config
st.set_page_config(page_title="InspectorsAlly", page_icon="🔍", layout="centered")

st.title("🔍 InspectorsAlly - Anomaly Detection")
st.caption("AI-powered visual quality control using Teachable Machine model.")

# Select input type
input_type = st.radio("Choose input method", ["Upload Image", "Camera Input"])

uploaded_image = None

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", width=300)

elif input_type == "Camera Input":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        uploaded_image = Image.open(camera_image)
        st.image(uploaded_image, caption="Camera Image", width=300)

# Preprocess and predict
if uploaded_image is not None:
    if st.button("Run Inspection"):
        with st.spinner("Inspecting..."):
            image = uploaded_image.resize((224, 224))
            image = np.asarray(image).astype(np.float32)
            image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            prediction = model.predict(image)
            index = np.argmax(prediction)
            predicted_class = class_names[index]
            confidence = prediction[0][index]

            st.subheader("🔎 Inspection Result:")
            st.write(f"**Class:** {predicted_class}")
            st.write(f"**Confidence:** {round(confidence * 100, 2)}%")

            if "good" in predicted_class.lower():
                st.success("✅ Product is GOOD - No defects detected.")
            else:
                st.error("❌ Anomaly Detected! This product may have defects.")
