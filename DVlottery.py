import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import os

# Ensure the cascade file is present
CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.isfile(CASCADE_PATH):
    st.error(f"Missing cascade file: {CASCADE_PATH}. Please add it to your repo.")
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def replace_background_white(np_img):
    """Convert image to RGB, then replace background with white by thresholding."""
    # Convert to RGB if needed
    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    # Here we assume background is light and subject darker — simple thresholding
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)
    mask_inv = cv2.bitwise_not(mask)
    white_bg = np.full(np_img.shape, 255, dtype=np.uint8)
    fg = cv2.bitwise_and(np_img, np_img, mask=mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
    combined = cv2.add(fg, bg)
    return combined

def auto_crop_square(image: Image.Image, min_head_ratio=0.50, max_head_ratio=0.69, final_size=600) -> Image.Image:
    """Automatic crop to square around detected face and ensure head size ratio roughly."""
    image = image.convert("RGB")
    np_img = np.array(image)
    # convert from PIL (RGB) to OpenCV BGR
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Replace background with white
    processed = replace_background_white(bgr)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
    if len(faces) == 0:
        raise Exception("No face detected. Please upload a clear, frontal photo.")
    # choose largest face
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])

    height, width = processed.shape[:2]
    # Expand bounding box somewhat (to include hair/shoulders)
    top = max(0, y - int(0.3*h))
    bottom = min(height, y + h + int(0.3*h))
    left = max(0, x - int(0.2*w))
    right = min(width, x + w + int(0.2*w))

    crop_img = processed[top:bottom, left:right]

    # Now we crop to square: find largest dimension
    c_h, c_w = crop_img.shape[:2]
    if c_h > c_w:
        # tall: pad width
        padded = cv2.copyMakeBorder(crop_img, 0, 0, (c_h-c_w)//2, (c_h-c_w)-(c_h-c_w)//2, cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        # wide or equal: pad height
        padded = cv2.copyMakeBorder(crop_img, (c_w-c_h)//2, (c_w-c_h)-(c_w-c_h)//2, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

    # Resize to final_size x final_size
    resized = cv2.resize(padded, (final_size, final_size), interpolation=cv2.INTER_AREA)

    # Convert back to PIL
    final_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

# Streamlit UI
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — Auto Crop & White Background")

uploaded_file = st.file_uploader("Upload your photo (jpg/jpeg)", type=["jpg","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    try:
        processed = auto_crop_square(image)
        st.image(processed, caption="Processed Image (Ready for DV Lottery)", use_column_width=True)
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        st.download_button(
            label="Download Ready Photo",
            data=buf,
            file_name="dvlottery_photo.jpg",
            mime="image/jpeg"
        )
    except Exception as e:
        st.error(f"Could not process image: {e}")
