import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import os
from rembg import remove

# ===============================
# Config
# ===============================
CASCADE_PATH = "haarcascade_frontalface_default.xml"
FINAL_SIZE = 600  # 2x2 inches @ 300 DPI

if not os.path.isfile(CASCADE_PATH):
    st.error(f"Missing cascade file: {CASCADE_PATH}. Please add it to your repo.")
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ===============================
# Helper functions
# ===============================
def replace_background_white(np_img):
    """Use rembg (AI) to remove background and fill with white."""
    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    # Remove background
    output = remove(img_pil)
    # Convert output with alpha (transparent) to white background
    rgba = np.array(output)
    if rgba.shape[2] == 4:  # has alpha
        alpha = rgba[:, :, 3] / 255.0
        white_bg = np.ones_like(rgba[:, :, :3], dtype=np.uint8) * 255
        composite = white_bg * (1 - alpha[:, :, None]) + rgba[:, :, :3] * alpha[:, :, None]
        result = composite.astype(np.uint8)
    else:
        result = rgba
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def auto_crop_dv(image: Image.Image, final_size=FINAL_SIZE):
    """Auto-crop to DV standard (600x600, head 50–69%)."""
    image = image.convert("RGB")
    np_img = np.array(image)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Step 1: clean background
    processed = replace_background_white(bgr)

    # Step 2: detect face
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected. Please upload a clear, frontal photo.")
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    height, width = processed.shape[:2]

    # Estimate top of head and chin
    head_top = max(0, int(y - 0.25*h))
    chin_bottom = min(height, int(y + h + 0.10*h))
    head_height = chin_bottom - head_top

    # Adjust crop for DV ratio
    target_ratio = 0.60
    target_crop_h = int(head_height / target_ratio)
    center_y = int((head_top + chin_bottom) / 2)
    top = max(0, center_y - target_crop_h // 2)
    bottom = min(height, top + target_crop_h)

    center_x = x + w // 2
    crop_width = target_crop_h
    left = max(0, center_x - crop_width // 2)
    right = min(width, left + crop_width)

    crop_img = processed[top:bottom, left:right]

    # Pad to square
    c_h, c_w = crop_img.shape[:2]
    if c_h > c_w:
        pad = (c_h - c_w) // 2
        crop_img = cv2.copyMakeBorder(crop_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif c_w > c_h:
        pad = (c_w - c_h) // 2
        crop_img = cv2.copyMakeBorder(crop_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    resized = cv2.resize(crop_img, (final_size, final_size), interpolation=cv2.INTER_AREA)
    final_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="DV Lottery Pro Photo Editor", layout="wide")
st.title("📸 DV Lottery Pro Photo Editor (AI Background Removal)")
st.markdown(
    """
    ✨ **Features**
    - AI-based background removal (studio-white)
    - Automatic head ratio (50–69% of image height)
    - 2×2 inch (600×600 px) per U.S. Visa standards
    """
)

uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG)", type=["jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    try:
        processed = auto_crop_dv(image)
        st.image(processed, caption="✅ DV-Ready Photo (2×2 inch, 600×600 px)", use_column_width=True)
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        st.download_button(
            "📥 Download Ready Photo",
            buf,
            file_name="dvlottery_photo.jpg",
            mime="image/jpeg",
        )
    except Exception as e:
        st.error(f"Could not process image: {e}")
