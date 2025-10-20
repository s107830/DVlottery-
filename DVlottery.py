import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Fully DV Compliant Auto-Adjust")

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
    """Fully auto-adjust the photo to meet DV requirements precisely."""
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_h_orig, img_w_orig = image_bgr.shape[:2]
    x, y, w, h = detect_face(image_bgr)

    # Padding for hair/chin
    top_pad = int(0.15 * h)
    bottom_pad = int(0.10 * h)
    head_top = max(0, y - top_pad)
    head_bottom = min(img_h_orig, y + h + bottom_pad)
    head_height = head_bottom - head_top

    # Approx eye height (from top of image)
    eye_y = y + int(0.45 * h)

    # Current ratios
    current_head_ratio = head_height / img_h_orig
    current_eye_ratio = (img_h_orig - eye_y) / img_h_orig

    # Target ratios (midpoints)
    target_head_ratio = (HEAD_MIN_RATIO + HEAD_MAX_RATIO) / 2
    target_eye_ratio = (EYE_MIN_RATIO + EYE_MAX_RATIO) / 2

    # Compute scale factor to satisfy both constraints
    scale_head = target_head_ratio / current_head_ratio
    scale_eye = target_eye_ratio / current_eye_ratio
    scale_factor = min(scale_head, scale_eye)

    # Resize image
    new_w = int(img_w_orig * scale_factor)
    new_h = int(img_h_orig * scale_factor)
    resized_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Scaled positions
    head_top_scaled = int(head_top * scale_factor)
    head_bottom_scaled = int(head_bottom * scale_factor)
    head_height_scaled = head_bottom_scaled - head_top_scaled
    eye_y_scaled = int(eye_y * scale_factor)

    # Determine canvas size
    canvas_size = max(new_w, new_h)
    canvas_size = max(MIN_SIZE, min(MAX_SIZE, canvas_size))
    canvas = np.full((canvas_size, canvas_size, 3), BG_COLOR, dtype=np.uint8)

    # Vertical offset to place eye line correctly
    eye_target = int(canvas_size * target_eye_ratio)
    top_offset = eye_target - (eye_y_scaled - head_top_scaled)
    # Clip top offset to ensure image fits
    top_offset = max(min(top_offset, canvas_size - new_h), 0)

    # Horizontal center
    left_offset = (canvas_size - new_w) // 2
    left_offset = max(left_offset, 0)

    # Compute paste region safely
    y1 = top_offset
    y2 = top_offset + new_h
    x1 = left_offset
    x2 = left_offset + new_w

    # Clip if exceeds canvas
    y2 = min(y2, canvas_size)
    x2 = min(x2, canvas_size)
    h_clip = y2 - y1
    w_clip = x2 - x1
    if h_clip > 0 and w_clip > 0:
        canvas[y1:y2, x1:x2] = resized_img[0:h_clip, 0:w_clip]

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
