import streamlit as st
import tensorflow as tf
import PIL.Image
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('signlanguage.h5')
    return model

model = load_model()

# CSS for custom styling
st.markdown("""
    <style>
        .title {
            background-color: #001f3f;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .upload-section {
            background-color: #001f3f;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
        }
        .result-section {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 150px;
        }
        .result-box {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            color: white;
            height: 100%;
        }
        .red { background-color: #FF4136; }
        .green { background-color: #2ECC40; }
        .yellow { background-color: #FFDC00; }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title"><h1>GROUP 2</h1><h2>American Sign Language</h2></div>', unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="upload-section"><h3>Choose File:</h3></div>', unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "png"])

# Function for prediction
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

# Results Display Section
result_container = st.container()
with result_container:
    col1, col2, col3 = st.columns(3)
    
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                       '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                       'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                       'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        string = "OUTPUT : " + class_names[np.argmax(prediction)]
        with col1:
            st.markdown(f'<div class="result-box red">{string}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="result-box green"></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="result-box yellow"></div>', unsafe_allow_html=True)
    else:
        with col1:
            st.markdown('<div class="result-box red">No file uploaded</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="result-box green"></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="result-box yellow"></div>', unsafe_allow_html=True)
