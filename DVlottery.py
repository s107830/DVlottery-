import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Config
CASCADE_PATH = "haarcascade_frontalface_default.xml"
FINAL_SIZE = 600  # 2x2 inches at 300 DPI

if not os.path.isfile(CASCADE_PATH):
    st.error(f"Missing cascade file: {CASCADE_PATH}")
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
segmentor = SelfiSegmentation()

def replace_background_white(np_img):
    """Use AI segmentation to isolate person and apply white background."""
    img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    segmented = segmentor.removeBG(img_rgb, (255,255,255), threshold=0.7)
    return cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)

def auto_crop_dv(image: Image.Image, final_size=FINAL_SIZE):
    """Auto crop photo to DV standard: 600x600px, head 50–69% of height."""
    image = image.convert("RGB")
    np_img = np.array(image)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Step 1: clean background
    processed = replace_background_white(bgr)

    # Step 2: detect face
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected. Upload a clear, frontal photo.")
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    height, width = processed.shape[:2]

    # Estimate top of head and chin (include hair)
    head_top = max(0, int(y - 0.25*h))
    chin_bottom = min(height, int(y + h + 0.10*h))
    head_height = chin_bottom - head_top

    # Desired head height ratio: 50–69% of image height
    target_ratio = 0.60
    target_crop_h = int(head_height / target_ratio)
    center_y = int((head_top + chin_bottom) / 2)
    top = max(0, center_y - target_crop_h // 2)
    bottom = min(height, top + target_crop_h)

    # Center horizontally and include shoulders
    center_x = x + w // 2
    crop_width = target_crop_h
    left = max(0, center_x - crop_width // 2)
    right = min(width, left + crop_width)

    crop_img = processed[top:bottom, left:right]

    # Pad to square with white if needed
    c_h, c_w = crop_img.shape[:2]
    if c_h > c_w:
        pad = (c_h - c_w) // 2
        crop_img = cv2.copyMakeBorder(crop_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])
    elif c_w > c_h:
        pad = (c_w - c_h) // 2
        crop_img = cv2.copyMakeBorder(crop_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

    resized = cv2.resize(crop_img, (final_size, final_size), interpolation=cv2.INTER_AREA)
    final_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

# Streamlit UI
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("🧠 DV Lottery Smart Photo Editor")
st.markdown("Automatically crop, center, and whiten your background per **U.S. Visa Photo Standards**.")

uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG)", type=["jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    try:
        processed = auto_crop_dv(image)
        st.image(processed, caption="✅ DV-Ready Photo (2x2 inch, 600x600 px)", use_column_width=True)
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        st.download_button(
            "📥 Download Ready Photo",
            buf,
            file_name="dvlottery_photo.jpg",
            mime="image/jpeg"
        )
    except Exception as e:
        st.error(str(e))
