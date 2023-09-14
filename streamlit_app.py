
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    im = im.resize([300,500])
    image = np.array(im)
    return image

best_model = tf.keras.models.load_model('modelConv4_test.h5')
# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

st.title("Kidney Stone Detection from Coronal CT Images")
st.header("Upload a coronal CT image to be diagnosted")

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    st.image(img)
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''

    st.markdown(hide_img_fs, unsafe_allow_html=True)
    st.write("Image Uploaded Successfully")
    if st.button('Diagnosis'):
        X = Image.open(uploadFile) 
        X = X.resize([224,224])
        X = np.array(X)
        X = X / 255.0
        test = []
        test.append(X)
        test = np.array(test)
        prediction = best_model.predict(test)
        y_pred = np.argmax(prediction, axis=1)
        if(y_pred == 0):
            st.write(str("{:.2f}".format(prediction[0][0]*100)+"% of chance that this image contains a kidney stone"))
        if(y_pred == 1):
            st.write(str("{:.2f}".format(prediction[0][0]*100)+"% of chance that this image does not contains a kidney stone"))    
else:
    st.write("Make sure you image is in JPG/PNG Format.")
