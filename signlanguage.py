import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('aslx.h5')
    return model

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))  
    img_array = np.array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Function to make predictions
def make_prediction(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Display the prediction result
def display_prediction(predictions):
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                   '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                   'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    confidence_percentage = np.max(predictions[0]) * 100
    return predicted_class, confidence_percentage

# Main function to run the Streamlit app
def main():
    # Load the model
    model = load_model()

    # Set up the Streamlit app layout
    st.title("Sign Language Classifier")
    st.write("Upload an image of a sign to classify it.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose a sign image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)

        # Make predictions
        predictions = make_prediction(model, img_array)

        # Display the prediction result
        predicted_class, confidence_percentage = display_prediction(predictions)
        st.success(f"The predicted sign is: {predicted_class} with {confidence_percentage:.2f}% confidence")

# Run the Streamlit app
if __name__ == "__main__":
    main()