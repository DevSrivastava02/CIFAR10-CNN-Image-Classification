import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cifar10_cnn_model")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classification")
st.write("Upload an image (32x32) to classify")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((32, 32))
    image = np.array(image) / 255.0

    if image.shape[-1] != 3:
        st.error("Image must have 3 color channels (RGB)")
    else:
        image = np.expand_dims(image, axis=0)

        # Prediction
        predictions = model(image, training=False)
        predicted_class = tf.argmax(predictions[0]).numpy()

        st.success(f"Predicted Class: **{class_names[predicted_class]}**")
