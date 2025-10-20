import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

# Set up page config
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")

# Title
st.title("DV Lottery Photo Editor")

# File uploader
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg"])

if uploaded_file:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image.convert('RGB'))  # Ensure the image is in RGB format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add image processing functions here (resize, background removal, etc.)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display processed image
    st.image(img_rgb, caption="Processed Image", use_column_width=True)

    # Convert image to BytesIO for download
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    # Download button
    st.download_button(
        label="Download Processed Image",
        data=img_bytes,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )
