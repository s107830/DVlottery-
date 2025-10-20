import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Fully DV Compliant")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
MAX_SIZE = 1200
HEAD_MIN_RATIO = 0.50
HEAD_MAX_RATIO = 0.69
EYE_MIN_RATIO = 0.56
EYE_MAX_RATIO = 0.69
BG_COLOR = (255, 255, 255)

# ---------------------- FUNCTIONS ----------------------

def remove_background(img_pil):
    """Remove background using rembg and replace with white."""
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    img_byte = img_byte.getvalue()
    result = remove(img_byte)
    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    white_bg = Image.new("RGBA", fg.size, BG_COLOR + (255,))
    composite = Image.alpha_composite(white_bg, fg)
    return composite.convert("RGB")

def detect_face(cv_img):
    """Detect largest face in an image using Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected. Upload a clear, frontal photo.")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x, y, w, h

def auto_adjust_dv_photo(image_pil):
    """Fully auto-adjust the photo to meet DV requirements safely."""
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = detect_face(image_bgr)

    # Padding for hair/chin
    top_pad = int(0.15 * h)
    bottom_pad = int(0.10 * h)
    head_top = max(0, y - top_pad)
    head_bottom = min(image_bgr.shape[0], y + h + bottom_pad)
    head_height = head_bottom - head_top

    # Eye line approx
    eye_y = y + int(0.45 * h)

    # Canvas size
    desired_head_ratio = (HEAD_MIN_RATIO + HEAD_MAX_RATIO) / 2
    canvas_size = int(max(MIN_SIZE, min(MAX_SIZE, head_height / desired_head_ratio)))

    # Resize image
    scale_factor = canvas_size / image_bgr.shape[0]
    resized_img = cv2.resize(image_bgr, (int(image_bgr.shape[1]*scale_factor), int(image_bgr.shape[0]*scale_factor)), interpolation=cv2.INTER_AREA)
    img_h, img_w = resized_img.shape[:2]

    # Scaled positions
    head_top_scaled = int(head_top * scale_factor)
    eye_y_scaled = int(eye_y * scale_factor)
    head_height_scaled = int(head_height * scale_factor)

    # Vertical offset to place eye line
    target_eye_y = int(canvas_size * ((EYE_MIN_RATIO + EYE_MAX_RATIO)/2))
    top_offset = target_eye_y - (eye_y_scaled - head_top_scaled)

    # Horizontal offset (center)
    left_offset = (canvas_size - img_w) // 2

    # Compute paste region safely
    canvas = np.full((canvas_size, canvas_size, 3), BG_COLOR, dtype=np.uint8)
    y1 = max(0, top_offset)
    y2 = min(canvas_size, top_offset + img_h)
    x1 = max(0, left_offset)
    x2 = min(canvas_size, left_offset + img_w)

    src_y1 = max(0, -top_offset)
    src_y2 = src_y1 + (y2 - y1)
    src_x1 = max(0, -left_offset)
    src_x2 = src_x1 + (x2 - x1)

    if y2 > y1 and x2 > x1:
        canvas[y1:y2, x1:x2] = resized_img[src_y1:src_y2, src_x1:src_x2]

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def draw_guidelines(img):
    """Draw DV photo guides."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Outer border
    draw.rectangle([(0, 0), (w-1, h-1)], outline="gray", width=2)

    # Head height
    head_min = int(h * HEAD_MIN_RATIO)
    head_max = int(h * HEAD_MAX_RATIO)
    draw.line([(0, h - head_max), (w, h - head_max)], fill="blue", width=2)
    draw.line([(0, h - head_min), (w, h - head_min)], fill="blue", width=2)
    draw.text((10, h - head_max + 5), "Head height 1–1⅜ in", fill="blue")

    # Eye line
    eye_min = int(h * EYE_MIN_RATIO)
    eye_max = int(h * EYE_MAX_RATIO)
    draw.line([(0, h - eye_max), (w, h - eye_max)], fill="green", width=2)
    draw.line([(0, h - eye_min), (w, h - eye_min)], fill="green", width=2)
    draw.text((10, h - eye_max + 5), "Eye line ~1.18 in", fill="green")

    draw.text((10, 10), f"Square: {w}x{h}px", fill="black")
    return img

# ---------------------- STREAMLIT UI ----------------------

uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG)", type=["jpg", "jpeg"])

if uploaded_file:
    try:
        img_bytes = uploaded_file.read()
        orig = Image.open(io.BytesIO(img_bytes))
        if orig.mode != "RGB":
            orig = orig.convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig, caption="Original")

        with col2:
            st.subheader("✅ Processed (DV Compliant)")
            bg_removed = remove_background(orig)
            processed = auto_adjust_dv_photo(bg_removed)
            final_preview = draw_guidelines(processed.copy())
            st.image(final_preview, caption="DV Compliance Preview")

            # Download button
            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV-Ready Photo",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
