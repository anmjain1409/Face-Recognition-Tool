import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🧑‍💻 Face Recognition App")

# Upload known face
known_file = st.file_uploader("Upload Known Face", type=["jpg", "jpeg", "png"])
# Upload test face
test_file = st.file_uploader("Upload Test Face", type=["jpg", "jpeg", "png"])

if known_file and test_file:
    known_img = face_recognition.load_image_file(known_file)
    test_img = face_recognition.load_image_file(test_file)

    known_encoding = face_recognition.face_encodings(known_img)
    test_encoding = face_recognition.face_encodings(test_img)

    if known_encoding and test_encoding:
        result = face_recognition.compare_faces([known_encoding[0]], test_encoding[0])
        st.image(test_img, caption="Test Image", use_column_width=True)
        if result[0]:
            st.success("✅ Faces Match!")
        else:
            st.error("❌ Faces Do Not Match.")
    else:
        st.warning("Could not detect a face in one of the images.")
