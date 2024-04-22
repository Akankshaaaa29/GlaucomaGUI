import streamlit as st
from model import custom
import numpy as np
from tensorflow.keras.layers import Input
import cv2
from io import BytesIO
from preprocess import preprocess_image
SIZE_X = 224  # Set your desired size
SIZE_Y = 224  # Set your desired size
 
if "model1_computed" not in st.session_state:
    st.session_state.model1_computed = False

st.title("Glaucoma prediction using LWG-Net")
st.write("Upload fundus image to predict glaucoma.")

# File Upload Section
option = st.selectbox('Choose the type of Image',('Fundus', 'OCT'))
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

# Display Uploaded Image and Processing Section
if uploaded_file is not None:
    st.sidebar.title("Uploaded Image")
    
    # Save the uploaded image to a temporary file
    temp_file_path = 'temp_uploaded_image.png'
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    # Read the image using cv2
    uploaded_image = cv2.imread(temp_file_path)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)

    st.sidebar.image(uploaded_image, caption='Input', width=200)
    height, width, _ = uploaded_image.shape
    if height != width:
        size = min(width, height)
        left = (width - size)
        top = (height - size)
        uploaded_image = uploaded_image[500:1500, 350:1500]
    uploaded_image = cv2.resize(uploaded_image, (224, 224), interpolation=cv2.INTER_AREA)
    # After processing uploaded_image
    uploaded_image = uploaded_image.astype(np.float32)
    uploaded_image = (uploaded_image - np.min(uploaded_image)) / (np.max(uploaded_image) - np.min(uploaded_image))
    uploaded_image = cv2.bilateralFilter(uploaded_image, 2, 50, 50)
    uploaded_image = np.expand_dims(uploaded_image, axis=0)  # Add batch dimension

    if st.button('Predict'):
        # Build and Load Model
        input_layer = Input((SIZE_X, SIZE_Y, 3))
        model = custom()
        model.load_weights('custom_aug.h5')

            # Perform Prediction
        st.session_state.model1_output = model.predict(uploaded_image)
        st.session_state.model1_computed = True
            # Result Section
        st.sidebar.title("Model Output")
    if st.session_state.model1_computed:
        if np.round(st.session_state.model1_output) == 0:
            st.sidebar.error("Glaucoma detected!")
            st.error("It is the case of Glaucoma, consult to an opthalmologist at the earliest")
        elif np.round(st.session_state.model1_output) == 1:
            st.sidebar.success("Normal")
            st.success("Healthy Case")
del st.session_state.model1_computed

