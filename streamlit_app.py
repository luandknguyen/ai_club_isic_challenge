import streamlit as st
from streamlit.logger import get_logger
from keras import layers
from PIL import Image
import numpy as np
from model import UNet

WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
WEIGHTS_FILE = "saved/v1/weights.h5"

LOGGER = get_logger(__name__)

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).resize(SIZE)
    image_array = np.array(image) / 255.0  # Scale pixel values if required
    image_array = image_array.reshape((1, WIDTH, HEIGHT, 3))  # Add batch dimension and remove alpha channel if present
    return image_array

# Function to postprocess the prediction
def postprocess_prediction(prediction):
    mask = prediction > 0.8  # Threshold the predictions to get binary mask
    mask = mask.squeeze() * 255  # Remove batch dimension and convert to uint8
    return Image.fromarray(mask.astype(np.uint8))

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    model = UNet()
    inputs = layers.Input((WIDTH, HEIGHT, 3))
    model(inputs)
    model.load_weights(WEIGHTS_FILE)
    
    uploaded_file = st.file_uploader("Choose a skin image...", type="jpg")

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        st.image(image.squeeze(), caption='Uploaded Skin Image', use_column_width=True)
        prediction = model.predict(image)
        st.image(prediction)



if __name__ == "__main__":
    run()