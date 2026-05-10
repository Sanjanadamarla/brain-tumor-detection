import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("saved_models/best_model.keras")
    return model

model = load_model()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):
    # Convert to RGB and resize
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)

    # Get highest probability
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    return CLASS_NAMES[predicted_index], confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection Using Deep Learning")
st.write("Upload a brain MRI image to predict the tumor type.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing MRI image..."):
            label, confidence = predict_image(image)

        # Show results
        st.success("Prediction completed!")

        st.subheader("Prediction Result")
        st.write(f"**Tumor Type:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Friendly message
        if label == "notumor":
            st.info("No tumor detected.")
        else:
            st.warning(f"Detected tumor type: {label}.")