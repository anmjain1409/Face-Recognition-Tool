import streamlit as st
import numpy as np
import cv2
import face_recognition

st.set_page_config(page_title="📸 Face Detection", layout="centered")
st.title("📸 Upload a Photo to Detect Faces")

# Load and verify image
def load_image(image_file):
    try:
        # Read image bytes as NumPy array
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        # Decode into image
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Convert BGR to RGB (face_recognition expects RGB)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except:
        return None

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = load_image(uploaded_file)

    if image is not None and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        # Detect face locations
        try:
            face_locations = face_recognition.face_locations(image)

            # Draw rectangles
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            st.subheader(f"✅ {len(face_locations)} Face(s) Detected")
            st.image(image, caption="Detected Faces", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Face detection failed: {e}")
    else:
        st.error("⚠️ Image format is not supported. Use standard JPG/PNG with RGB.")
else:
    st.info("📂 Please upload an image to start.")
