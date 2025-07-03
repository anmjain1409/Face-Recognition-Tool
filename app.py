import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import tempfile
import os
import cv2

st.set_page_config(page_title="🧑‍💻 Face Detector", layout="centered")
st.title("🧑‍💻 Upload Image for Face Detection")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load image using face_recognition's built-in function (PIL-based)
        image_np = face_recognition.load_image_file(tmp_path)

        # Debug details
        st.write(f"Image shape: {image_np.shape}")
        st.write(f"Image dtype: {image_np.dtype}")
        st.write(f"Image is contiguous: {image_np.flags['C_CONTIGUOUS']}")

        # Detect face locations (this should now work)
        face_locations = face_recognition.face_locations(image_np)

        if not face_locations:
            st.warning("😕 No faces detected.")
        else:
            # Draw rectangles using OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Convert back to RGB for display
            final_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.image(final_image, caption="✅ Faces Detected", use_column_width=True)
            st.success(f"✅ Detected {len(face_locations)} face(s).")

        # Clean up temp file
        os.remove(tmp_path)

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("📂 Please upload a face image.")
