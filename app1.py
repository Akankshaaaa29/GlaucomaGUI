import streamlit as st
from model import build_unet_plusresatten,build_unet_plusresattencup
import numpy as np
from isntddls import ISNT, DDLS, isglaucoma
from tensorflow.keras.layers import Input
import cv2 
from PIL import Image
from preprocess import preprocess_image

SIZE_X = 256 # Set your desired size
SIZE_Y = 256  # Set your desired size
 
if "model1_computed" not in st.session_state:
    st.session_state.model1_computed = False

st.title("Glaucoma prediction")
st.write("Upload fundus image to predict glaucoma.")

# File Upload Section
option = st.selectbox('Choose the type of Image',('Fundus'))
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

# Display Uploaded Image and Processing Section
if uploaded_file is not None:
    st.sidebar.title("Uploaded Image")
    
    # Save the uploaded image to a temporary file
    temp_file_path = 'temp_uploaded_image.png'
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
        uploaded_image = cv2.imread(temp_file_path)
        uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)

        st.sidebar.image(uploaded_image, width=200)
        uploaded_image = cv2.resize(uploaded_image, (SIZE_Y, SIZE_X))

    if st.button('Predict'):
        # Build and Load Model
        input_layer = Input((256, 256, 3))
        modeldisc = build_unet_plusresatten(input_layer, 'he_normal', 0.2)
        modelcup = build_unet_plusresattencup(input_layer, 'he_normal', 0.2)

        modeldisc.load_weights('model_unetresatten.h5')
        modelcup.load_weights('model_unetresattencup (1).h5')
            
        # Perform Prediction
        test_img_input = np.expand_dims(uploaded_image,0)
        test_pred1 = modeldisc.predict(test_img_input)
        test_prediction1 = np.argmax(test_pred1, axis=3)[0, :, :]
        preddisc = (test_prediction1 * 255).astype(np.uint8)

        # Display the final model output segmentation mask
        st.image(preddisc * 25, caption='Output Segmentation Disc', width=200, channels="GRAY")
        test_img_input1 = np.expand_dims(uploaded_image,0)
        test_pred1c = modelcup.predict(test_img_input1)
        test_prediction1 = np.argmax(test_pred1c, axis=3)[0, :, :]
        predc = (test_prediction1 * 255).astype(np.uint8)
        st.image(predc * 25, caption='Output Segmentation CUP', width=200, channels="GRAY")
        cup = cv2.normalize(predc, None, 0, 255, cv2.NORM_MINMAX)
        cup = cv2.resize(cup, (512, 512))
        disc = cv2.normalize(preddisc, None, 0, 255, cv2.NORM_MINMAX)
        disc= cv2.resize(disc, (512, 512))
        (thresh, disc) = cv2.threshold(disc, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('temp_disc.png', disc)
        (thresh, cup) = cv2.threshold(cup, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('temp_cup.png', cup)  # Saving cup image
        cup_img = Image.open('temp_cup.png')
        disc_img = Image.open('temp_disc.png')
        
        # Result Section
        isnt, cup_dias = ISNT(cup, disc, 'r')
        ddls, disc_dias, minrim, minang, minind = DDLS(cup_img, disc_img, 5)
        ver_cdr = cup_dias[0]/disc_dias[0]
        st.sidebar.title("Model Output")
        if isglaucoma(ver_cdr) == 'SEVERE':
            st.sidebar.error("Glaucoma detected!")
            st.error("It is the case of Glaucoma, consult to an opthalmologist at the earliest")
        else:
            st.sidebar.success("Normal")
            st.success("Healthy Case")
