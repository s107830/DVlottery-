import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Fully Auto-Adjusted DV Photo")

# ---------------------- CONSTANTS ----------------------
FINAL_PX = 600            # 2x2 inch at 300 DPI
HEAD_MIN_RATIO = 0.50     # 50% of image height
HEAD_MAX_RATIO = 0.69     # 69% of image height
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

def auto_scale_and_center(image_pil):
    """Automatically scale and center the head to fit DV guidelines."""
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = detect_face(image_bgr)

    # Calculate desired head height in pixels
    target_head_min = int(FINAL_PX * HEAD_MIN_RATIO)
    target_head_max = int(FINAL_PX * HEAD_MAX_RATIO)
    target_head_height = (target_head_min + target_head_max) // 2

    # Scale factor to make detected head fit target height
    scale_factor = target_head_height / h
    new_w = int(image_bgr.shape[1] * scale_factor)
    new_h = int(image_bgr.shape[0] * scale_factor)
    resized_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Recalculate face position after scaling
    x_scaled = int(x * scale_factor)
    y_scaled = int(y * scale_factor)
    w_scaled = int(w * scale_factor)
    h_scaled = int(h * scale_factor)

    # Center the head vertically in the final 600x600 canvas
    head_center_y = y_scaled + h_scaled // 2
    canvas = np.full((FINAL_PX, FINAL_PX, 3), BG_COLOR, dtype=np.uint8)
    top = FINAL_PX // 2 - head_center_y
    left = FINAL_PX // 2 - (x_scaled + w_scaled // 2)

    # Compute placement coordinates on canvas
    y1 = max(0, top)
    y2 = min(FINAL_PX, top + resized_img.shape[0])
    x1 = max(0, left)
    x2 = min(FINAL_PX, left + resized_img.shape[1])

    # Compute corresponding coordinates from resized image
    src_y1 = max(0, -top)
    src_y2 = src_y1 + (y2 - y1)
    src_x1 = max(0, -left)
    src_x2 = src_x1 + (x2 - x1)

    canvas[y1:y2, x1:x2] = resized_img[src_y1:src_y2, src_x1:src_x2]

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def draw_guidelines(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Outer border
    draw.rectangle([(0, 0), (w-1, h-1)], outline="gray", width=2)

    # Head height guidelines
    head_min = int(h * (1 - HEAD_MAX_RATIO) / 2)
    head_max = int(h * (1 - HEAD_MIN_RATIO) / 2)
    draw.line([(50, head_min), (w-50, head_min)], fill="blue", width=2)
    draw.line([(50, h - head_min), (w-50, h - head_min)], fill="blue", width=2)
    draw.text((w-200, head_min+5), "Head height 1–1⅜ inch", fill="blue")

    # Eye line (approx)
    eye_line_y = int(h * 0.4)
    draw.line([(50, eye_line_y), (w-50, eye_line_y)], fill="green", width=2)
    draw.text((60, eye_line_y + 5), "Eye line ~1.18 in from bottom", fill="green")

    # Inch markings
    inch_px = FINAL_PX // 2
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
            processed = auto_scale_and_center(bg_removed)
            final_preview = draw_guidelines(processed.copy())
            st.image(final_preview, caption="DV Compliance Preview")

            # Download button
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
