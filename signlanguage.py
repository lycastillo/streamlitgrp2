import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('signlanguage.h5')
  return model
model=load_model()
st.write("""
# American Sign Language Identifier"""
)
file=st.file_uploader("Choose a hand gesture from the photos",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['0', '1', '2', '3', '4', '5', '6', '7','8',
                  '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                  'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                  'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)