import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🧑‍💻 Face Recognition (Streamlit)", layout="centered")
st.title("🧑‍💻 Upload Image to Detect Faces")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image as bytes and convert to PIL
        file_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")  # Force RGB

        # Convert PIL to proper NumPy array (RGB, uint8)
        image_np = np.array(pil_image).astype(np.uint8)

        # Validate shape and dtype
        if image_np.ndim != 3 or image_np.shape[2] != 3 or image_np.dtype != np.uint8:
            st.error("Image is not valid RGB 8-bit format.")
        else:
            # Convert to BGR for OpenCV processing
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Detect faces
            face_locations = face_recognition.face_locations(image_bgr)

            if not face_locations:
                st.warning("No faces detected in the image.")
            else:
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

                # Convert back to RGB for display
                final_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                st.image(final_image, caption="✅ Face(s) Detected", use_column_width=True)
                st.success(f"Detected {len(face_locations)} face(s).")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("📂 Please upload an image file.")
