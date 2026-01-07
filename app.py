import streamlit as st
import cv2
import numpy as np
from PIL import Image

# App configuration
st.set_page_config(page_title="Face Identification App", layout="centered")

st.title("🧠 Human Face Identification")
st.write("Upload an image and the app will detect human faces.")

# Sidebar controls
st.sidebar.header("🔧 Model Parameters")
scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Image upload
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    st.subheader("✅ Face Detection Result")
    st.image(img_array, use_container_width=True)

    st.success(f"Number of faces detected: {len(faces)}")
else:
    st.info("Please upload an image to start face detection.")
