import streamlit as st
import tensorflow as tf
import PIL.Image
from PIL import Image, ImageOps
import numpy as np

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('signlanguage.h5')
    return model

model = load_model()

# Function to read the HTML file
def read_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content

# Path to your HTML file
html_file_path = 'index.html'

# Read the HTML file
html_content = read_html(html_file_path)

# Render the HTML file in Streamlit
st.markdown(html_content, unsafe_allow_html=True)

# Additional Streamlit UI elements
file = st.file_uploader("Choose a hand gesture from the photos", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)  # Match the input size with the Google Colab code
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS)  # Use PIL.Image.LANCZOS for resizing
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)

    # Reshape input according to the model's input shape
    img_reshape = tf.image.resize(img, [64, 64])  # Resize to (64, 64)
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                   '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                   'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
