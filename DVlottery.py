import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

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
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize image to 600x600 pixels
    img_resized = cv2.resize(img, (600, 600))
    st.image(img_resized, caption="Resized Image", use_column_width=True)

    # Convert back to RGB for download
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # Download button
    st.download_button(
        label="Download Processed Image",
        data=img_bytes,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )
