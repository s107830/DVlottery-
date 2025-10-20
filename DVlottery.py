import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import os
from rembg import remove

CASCADE_PATH = "haarcascade_frontalface_default.xml"
FINAL_SIZE = 600  # 2x2 inch @ 300 DPI
DPI = 300

if not os.path.isfile(CASCADE_PATH):
    st.error(f"Missing cascade file: {CASCADE_PATH}")
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def replace_background_white(np_img):
    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    output = remove(img_pil)
    rgba = np.array(output)
    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3] / 255.0
        white_bg = np.ones_like(rgba[:, :, :3], dtype=np.uint8) * 255
        composite = white_bg * (1 - alpha[:, :, None]) + rgba[:, :, :3] * alpha[:, :, None]
        result = composite.astype(np.uint8)
    else:
        result = rgba
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def auto_crop_dv(image: Image.Image, final_size=FINAL_SIZE):
    image = image.convert("RGB")
    np_img = np.array(image)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    processed = replace_background_white(bgr)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected.")
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    height, width = processed.shape[:2]

    head_top = max(0, int(y - 0.25*h))
    chin_bottom = min(height, int(y + h + 0.10*h))
    head_height = chin_bottom - head_top
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

def add_dv_guidelines(image: Image.Image):
    """Draw guidelines like sample photo with inch markers and measurements."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    inch_per_px = 2 / W

    face_top = int(H * 0.35)
    face_bottom = int(H * 0.85)
    eye_line = int(H * 0.55)

    face_height_inch = round((face_bottom - face_top) * inch_per_px, 2)
    eye_line_inch = round(eye_line * inch_per_px, 2)

    # Blue lines
    line_color = (0, 90, 255)
    draw.line([(0, face_top), (W, face_top)], fill=line_color, width=2)
    draw.line([(0, face_bottom), (W, face_bottom)], fill=line_color, width=2)
    draw.line([(0, eye_line), (W, eye_line)], fill=line_color, width=2)

    # Border
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline=(0, 100, 255), width=4)

    # Labels
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    draw.text((10, face_top - 25), f"Face: {face_height_inch} inch", fill="blue", font=font)
    draw.text((10, eye_line - 25), f"Eye-line: {eye_line_inch} inch", fill="blue", font=font)
    draw.text((W - 140, H - 30), "2 inch × 2 inch", fill="gray", font=font)
    return img

# Streamlit UI
st.set_page_config(page_title="DV Photo Studio", layout="wide")
st.title("📏 DV Lottery Photo Studio (with Guidelines)")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_column_width=True)

    try:
        processed = auto_crop_dv(image)
        preview = add_dv_guidelines(processed)

        st.image(preview, caption="📐 DV Guideline Preview (2×2 inches)", use_column_width=True)
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        st.download_button(
            "📥 Download Final Clean Photo (No Lines)",
            buf,
            file_name="dvlottery_ready.jpg",
            mime="image/jpeg",
        )
    except Exception as e:
        st.error(f"Error: {e}")
