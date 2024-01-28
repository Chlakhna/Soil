import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import os
import warnings
import base64


warnings.filterwarnings("ignore")

# Load model
def load_model():
    model_fp = os.getcwd() + '/' + 'my_model1.h5'
    model = tf.keras.models.load_model(model_fp)
    return model

def classify_percentage(image, model):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    img_batch = np.expand_dims(image_array, axis=0)
    predicted_value = model.predict(img_batch)
    class_names = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']
    class_probabilities = predicted_value[0]
    results = []
    for name, probability in zip(class_names, class_probabilities):
        result = {
            "name": name,
            "value": f"{probability:.4f}"  # Adjust decimal places as desired
        }
        results.append(result)
    max_index = np.argmax(class_probabilities)
    max_class = class_names[max_index]
    max_probability = class_probabilities[max_index]
    return max_class, max_probability, results

# Set some pre-defined configurations for the page
st.set_page_config(
    page_title="Soil Detection Using Machine Learning",
    page_icon=":Soil:",
    initial_sidebar_state='auto'
)
with st.spinner('Model is being loaded..'):
    model = load_model()

# Hide the part of the code for custom CSS styling
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.image('image/ams.jpg', width=100)
    with col2:
        st.markdown("<h2 style='font-family: Arial; font-size: 14px; color: lightblue;'>Applied Mathematics and Statistics(AMS)</h6>", unsafe_allow_html=True)
    st.image('image/soil(image).jpg', width=305)

# Main content
st.write("# Soil Detection Using Machine Learning")
file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file) 
    st.image(image, use_column_width=True)
    max_class, max_probability, results = classify_percentage(image, model)
    st.write("Predicted Class:", max_class)
    st.write("Probability:", max_probability * 100, "%")

    if max_class == 'Alluvial Soil':
        output = "rice, wheat, pulses, oilseeds, sugarcane, cotton, cauliflower, cabbage, and brinjal."
    elif max_class == 'Black Soil':
        output = " cotton, soybeans, peanuts, pigeon peas, black gram, green gram, sunflower, sorghum, castor, and sesame."
    elif max_class == 'Clay Soil':
        output = "millets, groundnuts, pulses, oilseeds, sweet potatoes, pomegranate, and grapes through proper soil management practices. "
    elif max_class == 'Red Soil':
        output = "rice, wheat, maize, sugarcane, pulses, oilseeds, vegetables, mangoes, and bananas."
    else:
        output = None

    # Display the output
    if output is not None:
        st.write("The Crops: ",output)
    else:
        st.write("Unable to determine the class")





def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('image/photo.jpg')