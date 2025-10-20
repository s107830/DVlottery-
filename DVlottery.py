import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
from rembg import remove

# Streamlit setup
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Auto Crop, White Background & Official Size Guide")

# Constants
FINAL_PX = 600            # 2x2 inch at 300 DPI = 600x600 pixels
HEAD_MIN_RATIO = 0.50     # 50% of image height (1 inch)
HEAD_MAX_RATIO = 0.69     # 69% of image height (~1 3/8 inch)
BG_COLOR = (255, 255, 255)

# ---------------------- FUNCTIONS ----------------------

def remove_background(img_pil):
    """Remove background using rembg and replace with pure white."""
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    img_byte = img_byte.getvalue()
    result = remove(img_byte)
    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    white_bg = Image.new("RGBA", fg.size, BG_COLOR + (255,))
    composite = Image.alpha_composite(white_bg, fg)
    return composite.convert("RGB")

def detect_face(cv_img):
    """Detect largest face in an image."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected. Please upload a clear, frontal photo.")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x, y, w, h

def crop_and_resize(image_pil):
    """Crop around face and resize to DV 2x2 inch ratio."""
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = detect_face(image_bgr)

    height, width = image_bgr.shape[:2]
    top = max(0, y - int(0.45 * h))
    bottom = min(height, y + h + int(0.35 * h))
    left = max(0, x - int(0.25 * w))
    right = min(width, x + w + int(0.25 * w))

    cropped = image_bgr[top:bottom, left:right]

    # Pad to square with white
    c_h, c_w = cropped.shape[:2]
    diff = abs(c_h - c_w)
    if c_h > c_w:
        pad = (diff // 2, diff - diff // 2)
        cropped = cv2.copyMakeBorder(cropped, 0, 0, pad[0], pad[1], cv2.BORDER_CONSTANT, value=BG_COLOR)
    elif c_w > c_h:
        pad = (diff // 2, diff - diff // 2)
        cropped = cv2.copyMakeBorder(cropped, pad[0], pad[1], 0, 0, cv2.BORDER_CONSTANT, value=BG_COLOR)

    resized = cv2.resize(cropped, (FINAL_PX, FINAL_PX), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

def draw_guidelines(img):
    """Draw official DV photo size guides on image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # outer 2x2 inch border
    draw.rectangle([(0, 0), (w-1, h-1)], outline="gray", width=2)

    # Head height (50–69%)
    head_min = int(h * (1 - HEAD_MAX_RATIO) / 2)
    head_max = int(h * (1 - HEAD_MIN_RATIO) / 2)
    top_y = head_min
    bottom_y = h - head_min

    # Guidelines
    draw.line([(w//2 - 150, top_y), (w//2 + 150, top_y)], fill="blue", width=2)
    draw.line([(w//2 - 150, h - top_y), (w//2 + 150, h - top_y)], fill="blue", width=2)
    draw.text((w//2 + 155, (top_y + h - top_y)//2 - 20), "Head height 1–1⅜ inch (25–35 mm)", fill="blue")

    # Eye line (approx 1.18 inch from bottom)
    eye_line_y = int(h - (1.18 / 2.0) * (h / 2.0))
    draw.line([(50, eye_line_y), (w - 50, eye_line_y)], fill="green", width=2)
    draw.text((60, eye_line_y + 5), "Eye line ~1.18 in from bottom", fill="green")

    # Inch markings along sides
    inch_px = FINAL_PX // 2  # 300 px = 1 inch
    for i in range(1, 2):
        y = i * inch_px
        draw.line([(0, y), (15, y)], fill="black", width=2)
        draw.text((20, y - 10), f"{i} inch", fill="black")

        draw.line([(y, 0), (y, 15)], fill="black", width=2)
        draw.text((y - 15, 20), f"{i} inch", fill="black")

    draw.text((10, 10), "2x2 inch (51x51 mm)", fill="black")
    return img

# ---------------------- STREAMLIT UI ----------------------

uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG)", type=["jpg", "jpeg"])

if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Original Photo")
        st.image(orig, use_container_width=True)

    with col2:
        st.subheader("✅ Processed (DV Compliant)")
        try:
            bg_removed = remove_background(orig)
            processed = crop_and_resize(bg_removed)
            final_preview = draw_guidelines(processed.copy())

            st.image(final_preview, use_container_width=True, caption="DV Compliance Preview")

            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV-Ready Photo (600x600)",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"❌ Could not process image: {e}")
