import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from rembg import remove
import io

# --- Constants ---
FINAL_PX = 600
HEAD_MIN_PX = 300
HEAD_MAX_PX = 414
EYE_LINE_MIN_PX = 336
EYE_LINE_MAX_PX = 414

# --- Helper: Detect face using OpenCV Haar cascade ---
def detect_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        h, w = gray.shape
        return (int(w * 0.3), int(h * 0.25), int(w * 0.4), int(h * 0.5))
    else:
        # return the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]

# --- Background remover with professional cleanup ---
def remove_background_smooth(img_pil):
    """Remove background and fully clean edge halo."""
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    result = remove(img_byte.getvalue())

    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    fg_np = np.array(fg)
    alpha = fg_np[:, :, 3].astype(np.float32) / 255.0

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # Decontaminate edges (make them white)
    fg_rgb = fg_np[:, :, :3].astype(np.float32)
    bg_white = np.ones_like(fg_rgb) * 255.0
    clean_rgb = fg_rgb * alpha[..., None] + bg_white * (1 - alpha[..., None])

    clean_img = Image.fromarray(np.uint8(clean_rgb))
    return clean_img.convert("RGB")

# --- Draw DV photo guidelines dynamically ---
def draw_guidelines(img, face_box=None):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Outer border
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="gray", width=2)

    if face_box:
        (x, y, w_face, h_face) = face_box
        head_top = max(0, int(y - 0.25 * h_face))
        chin = int(y + h_face + 0.05 * h_face)

        # Blue head height lines
        draw.line([(50, head_top), (w - 50, head_top)], fill="blue", width=2)
        draw.line([(50, chin), (w - 50, chin)], fill="blue", width=2)

        # Eye line (mid-face)
        eye_y = y + int(h_face * 0.45)
        draw.line([(50, eye_y), (w - 50, eye_y)], fill="green", width=2)

        # Height display
        head_height_px = chin - head_top
        draw.text((60, head_top + 10), f"Head height ≈ {head_height_px}px", fill="blue")
        draw.text((60, eye_y + 10), "Eye line", fill="green")

        # Red alert if not within DV spec
        if head_height_px < HEAD_MIN_PX or head_height_px > HEAD_MAX_PX:
            st.error(f"⚠️ Head height ({head_height_px}px) is out of DV range (300–414px).")
        if eye_y < EYE_LINE_MIN_PX or eye_y > EYE_LINE_MAX_PX:
            st.warning(f"⚠️ Eye line ({eye_y}px) is outside 336–414px range.")
    else:
        draw.text((10, 10), "No face detected", fill="red")

    # Inch reference marks
    inch_px = FINAL_PX // 2
    for i in range(1, 2):
        y_line = i * inch_px
        draw.line([(0, y_line), (15, y_line)], fill="black", width=2)
        draw.text((20, y_line - 10), f"{i} inch", fill="black")
        draw.line([(y_line, 0), (y_line, 15)], fill="black", width=2)
        draw.text((y_line - 15, 20), f"{i} inch", fill="black")

    draw.text((10, 10), "2x2 inch (51x51 mm)", fill="black")
    return img

# --- Streamlit UI ---
st.set_page_config(page_title="DV Photo Auto Guideline", layout="wide")
st.title("🧍‍♂️ U.S. DV Photo Auto Guideline Generator")
st.write("Upload your photo — the system will auto-remove the background, detect face position, and overlay official DV guidelines (600×600 px).")

uploaded = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Processing photo..."):
        cleaned = remove_background_smooth(image)
        cleaned = cleaned.resize((FINAL_PX, FINAL_PX))

        # Detect face for guideline placement
        image_rgb = np.array(cleaned)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        face_box = detect_face(image_bgr)

        final_preview = draw_guidelines(cleaned.copy(), face_box=face_box)

    st.image(final_preview, caption="DV Photo with Guidelines", use_container_width=True)

    # Download option
    buf = io.BytesIO()
    final_preview.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button("⬇️ Download Final DV Photo", data=byte_im, file_name="dv_photo_guideline.jpg", mime="image/jpeg")
