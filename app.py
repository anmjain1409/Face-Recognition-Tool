import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🧑‍💻 Face Recognition", layout="centered")
st.title("🧑‍💻 Face Recognition App (Streamlit Version)")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image from upload
        img_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Ensure RGB mode
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert("RGB")

        # Convert to NumPy array (uint8)
        img = np.array(pil_img).astype('uint8')

        # Debug info (optional)
        st.write(f"Image shape: {img.shape}")
        st.write(f"Image dtype: {img.dtype}")

        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Face detection
        face_locations = face_recognition.face_locations(img_bgr)

        if not face_locations:
            st.warning("😕 No faces found in the image.")
        else:
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Convert back to RGB for Streamlit display
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Detected Faces", use_column_width=True)
            st.success(f"✅ {len(face_locations)} face(s) detected.")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("📂 Upload an image to get started.")
