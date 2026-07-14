import streamlit as st
import cv2 as cv
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'models/NSFW_predictor.keras')

model=load_model()

def preprocess_streamlit_image(image_path):
    file_bytes=np.asarray(bytearray(image_path.read()),dtype=np.uint8)

    img=cv.imdecode(file_bytes,1)

    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img=cv.resize(img,(214,214),interpolation=cv.INTER_AREA)
    img=img.astype(np.float32) / 255
    img=np.expand_dims(img,axis=0)

    return img

st.title("Image content filter")
st.write('upload an image to check if it contains NSFW(adult) content')

uploaded_image=st.file_uploader('choose an image...',type='image')

if uploaded_image is not None:
    st.image(uploaded_image,caption='image uploaded')

    if st.button('Scan image'):
        st.spinner('Analyzing.....')

        processed_image=preprocess_streamlit_image(uploaded_image)
        
        prediction_raw=model.predict(processed_image)
        sfw_score=float(prediction_raw[0][0])

        st.divider()
        st.subheader('Scan Results')
        st.write(f"**Raw probability score:** {1-sfw_score:.4f}")

        if sfw_score>0.5:
            st.success(f"✅ Verdict: Safe Content ({sfw_score * 100:.2f}%)")
        else:
            st.error(f"🚨 Verdict: NSFW Content Detected ({(1- sfw_score) * 100:.2f}%)")