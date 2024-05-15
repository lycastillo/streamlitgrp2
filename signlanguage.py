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

# Main layout
st.markdown("""
    <style>
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .title-section {
            width: 100%;
            text-align: center;
            background-color: #001f3f;
            color: white;
            padding: 20px 0;
            margin-bottom: 20px;
        }
        .title-section h1 {
            margin: 0;
        }
        .upload-section {
            background-color: #001f3f;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            width: 80%;
            max-width: 600px;
            margin-bottom: 20px;
        }
        .result-section {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 80%;
            max-width: 600px;
        }
        .result-box {
            flex: 1;
            height: 150px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            color: white;
        }
        .result-box.red { background-color: #FF4136; }
        .result-box.green { background-color: #2ECC40; }
        .result-box.yellow { background-color: #FFDC00; }
    </style>
    <div class="main">
        <div class="title-section">
            <h1>American Sign Language</h1>
        </div>
        <div class="upload-section">
            <h3>Choose File:</h3>
            <input type="file" id="file-uploader" accept=".jpg, .png">
        </div>
        <div class="result-section">
            <div class="result-box red" id="result-box-red"></div>
            <div class="result-box green" id="result-box-green"></div>
            <div class="result-box yellow" id="result-box-yellow"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg", "png"])

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

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                   '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                   'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.markdown(f"""
        <script>
            document.getElementById('result-box-red').innerText = '{string}';
            document.getElementById('result-box-green').innerText = '';
            document.getElementById('result-box-yellow').innerText = '';
        </script>
    """, unsafe_allow_html=True)
else:
    st.text("Please upload an image file")
