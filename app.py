import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Disease classes
class_names = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "septoria_spot"
]

# Disease descriptions
disease_info = {
    "early_blight": {
        "name": "Early Blight",
        "description": "Early Blight is a fungal disease that causes brown spots with concentric rings on tomato leaves.",
        "symptoms": "Yellowing leaves, dark circular spots, leaf drying."
    },

    "healthy": {
        "name": "Healthy Leaf",
        "description": "The tomato leaf appears healthy with no visible disease symptoms.",
        "symptoms": "Fresh green leaves with normal texture."
    },

    "late_blight": {
        "name": "Late Blight",
        "description": "Late Blight is a severe disease causing dark lesions and rapid plant decay.",
        "symptoms": "Water soaked spots, black patches, leaf wilting."
    },

    "leaf_mold": {
        "name": "Leaf Mold",
        "description": "Leaf Mold affects tomato leaves and creates yellow spots with mold growth underneath.",
        "symptoms": "Yellow patches, velvety mold formation."
    },

    "septoria_spot": {
        "name": "Septoria Leaf Spot",
        "description": "Septoria Leaf Spot causes multiple small circular spots on tomato leaves.",
        "symptoms": "Tiny dark spots with gray centers and yellow edges."
    }
}

# App title
st.title("🍅 Tomato Leaf Disease Detection System")

st.write("Upload a tomato leaf image to detect diseases using Deep Learning.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Tomato Leaf Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Disease details
    disease = disease_info[predicted_class]

    # Results
    st.subheader("Prediction Result")

    st.success(f"Disease: {disease['name']}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    st.write("### Description")
    st.write(disease["description"])

    st.write("### Symptoms")
    st.write(disease["symptoms"])