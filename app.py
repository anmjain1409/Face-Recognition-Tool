import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🧑‍💻 Face Recognition Tool")

# Upload known face
known_file = st.file_uploader("Upload Known Face", type=["jpg", "jpeg", "png"])
# Upload test face
test_file = st.file_uploader("Upload Test Face", type=["jpg", "jpeg", "png"])

def convert_image(img_file):
    try:
        image = Image.open(img_file).convert("RGB")             # Ensure RGB
        image_np = np.asarray(image, dtype=np.uint8)            # Convert to uint8 NumPy array
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            return image_np
        else:
            st.error("Image must be RGB.")
            return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

if known_file and test_file:
    known_img = convert_image(known_file)
    test_img = convert_image(test_file)

    if known_img is not None and test_img is not None:
        try:
            known_encodings = face_recognition.face_encodings(known_img)
            test_encodings = face_recognition.face_encodings(test_img)

            if known_encodings and test_encodings:
                result = face_recognition.compare_faces([known_encodings[0]], test_encodings[0])

                st.image(test_img, caption="Test Image", use_column_width=True)

                if result[0]:
                    st.success("✅ Faces Match!")
                else:
                    st.error("❌ Faces Do Not Match.")
            else:
                st.warning("😕 Couldn't detect a face in one of the images.")
        except Exception as e:
            st.error(f"⚠️ Error processing images: {e}")
