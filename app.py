import streamlit as st
from PIL import Image
import numpy as np
import face_recognition
import cv2

st.title("Face Recognition App")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and convert image to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Convert to numpy array
    image_np = np.array(image)

    # Detect face locations
    try:
        face_locations = face_recognition.face_locations(image_np)

        # Draw rectangles around detected faces
        for top, right, bottom, left in face_locations:
            cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show image with face boxes
        st.image(image_np, caption=f"Detected {len(face_locations)} face(s)", use_column_width=True)

    except RuntimeError as e:
        st.error(f"RuntimeError: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
