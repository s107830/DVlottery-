import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from rembg import remove
import io

# ------------------------------
# Streamlit UI setup
# ------------------------------
st.set_page_config(page_title="DV Lottery Photo Editor", page_icon="📸", layout="centered")
st.title("📸 DV Lottery Photo Editor — Auto Background, Crop & Official Guidelines")
st.write("Upload a clear front-facing photo (JPG, JPEG, or PNG)")

# ------------------------------
# Constants
# ------------------------------
FINAL_SIZE = 600  # 600x600 pixels (2x2 inches)
BG_COLOR = (255, 255, 255)
HEAD_MIN = 300  # Top of head to chin
HEAD_MAX = 414
EYE_MIN = 336
EYE_MAX = 414

# ------------------------------
# Background Removal
# ------------------------------
def remove_background(img_pil: Image.Image) -> Image.Image:
    img_pil = img_pil.convert("RGBA")
    input_bytes = io.BytesIO()
    img_pil.save(input_bytes, format="PNG")
    input_bytes = input_bytes.getvalue()

    result = remove(input_bytes)
    result_img = Image.open(io.BytesIO(result)).convert("RGBA")

    # Composite on white background
    white_bg = Image.new("RGBA", result_img.size, (255, 255, 255, 255))
    white_bg.paste(result_img, mask=result_img.split()[3])
    return white_bg.convert("RGB")

# ------------------------------
# Face Detection + Crop/Resize
# ------------------------------
def detect_and_crop(img_pil: Image.Image):
    np_img = np.array(img_pil)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        raise Exception("No face detected! Please upload a clear, front-facing photo.")

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w // 2, y + h // 2
    crop_size = int(max(w, h) * 2.1)

    left = max(0, cx - crop_size // 2)
    top = max(0, cy - crop_size // 2)
    right = min(np_img.shape[1], cx + crop_size // 2)
    bottom = min(np_img.shape[0], cy + crop_size // 2)

    cropped = img_pil.crop((left, top, right, bottom))
    resized = cropped.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
    return resized, (x, y, w, h)

# ------------------------------
# Draw Official DV Guidelines
# ------------------------------
def draw_dv_guidelines(img: Image.Image):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Center vertical line
    draw.line([(w // 2, 0), (w // 2, h)], fill="gray", width=2)

    # Head height zone (300–414 px)
    draw.line([(0, HEAD_MIN), (w, HEAD_MIN)], fill="red", width=2)
    draw.line([(0, HEAD_MAX), (w, HEAD_MAX)], fill="red", width=2)
    draw.text((10, HEAD_MIN - 20), "Head height: 300–414 px", fill="red")

    # Eye position zone (336–414 px from bottom)
    eye_min_y = h - EYE_MIN
    eye_max_y = h - EYE_MAX
    draw.line([(0, eye_min_y), (w, eye_min_y)], fill="blue", width=2)
    draw.line([(0, eye_max_y), (w, eye_max_y)], fill="blue", width=2)
    draw.text((10, eye_min_y - 20), "Eye line: 336–414 px from bottom", fill="blue")

    # Outer border
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="gray", width=3)

    # Labels
    draw.text((10, 10), "2x2 inch (600x600 px)", fill="black")

    return img

# ------------------------------
# Streamlit Main UI
# ------------------------------
uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image_data = uploaded_file.read()
        orig = Image.open(io.BytesIO(image_data))
        if orig.mode != "RGB":
            orig = orig.convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig, caption="Uploaded Image")

        with col2:
            st.subheader("✅ Processed DV-Compliant Photo")

            bg_removed = remove_background(orig)
            cropped, _ = detect_and_crop(bg_removed)
            final_preview = draw_dv_guidelines(cropped.copy())

            st.image(final_preview, caption="DV 2x2 inch with Guidelines")

            # Download button
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                label="⬇️ Download DV-Ready Photo (600x600)",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
else:
    st.info("👆 Please upload a clear, front-facing photo.")
