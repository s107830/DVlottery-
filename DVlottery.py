import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import io
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="📸 DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Auto Background, Crop & Guidelines")

# ---------------------- CONSTANTS ----------------------
FINAL_PX = 600           # 2x2 inch at 300 DPI
HEAD_MIN_RATIO = 0.50    # 50% of image height
HEAD_MAX_RATIO = 0.69    # 69% of image height
BG_COLOR = (255, 255, 255)

# ---------------------- FUNCTIONS ----------------------

def remove_background_smooth(img_pil):
    """Remove background and clean edges (no halo)."""
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    result = remove(img_byte.getvalue())

    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    fg_np = np.array(fg)
    alpha = fg_np[:, :, 3].astype(np.float32) / 255.0

    # 1️⃣ Refine alpha mask
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # 2️⃣ Decontaminate edge color
    fg_rgb = fg_np[:, :, :3].astype(np.float32)
    bg_white = np.ones_like(fg_rgb) * 255.0
    clean_rgb = fg_rgb * alpha[..., None] + bg_white * (1 - alpha[..., None])

    # 3️⃣ Convert back
    clean_img = Image.fromarray(np.uint8(clean_rgb))
    return clean_img.convert("RGB")


def detect_face(cv_img):
    """Detect largest face in an image."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected. Upload a clear, frontal photo.")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x, y, w, h


def crop_and_resize(image_pil):
    """Crop around face and resize to 600x600 with DV ratios."""
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = detect_face(image_bgr)
    height, width = image_bgr.shape[:2]

    # Crop with padding around face
    top = max(0, y - int(0.45 * h))
    bottom = min(height, y + h + int(0.35 * h))
    left = max(0, x - int(0.25 * w))
    right = min(width, x + w + int(0.25 * w))
    cropped = image_bgr[top:bottom, left:right]

    # Pad to square
    c_h, c_w = cropped.shape[:2]
    diff = abs(c_h - c_w)
    if c_h > c_w:
        pad = (diff // 2, diff - diff // 2)
        cropped = cv2.copyMakeBorder(cropped, 0, 0, pad[0], pad[1], cv2.BORDER_CONSTANT, value=BG_COLOR)
    elif c_w > c_h:
        pad = (diff // 2, diff - diff // 2)
        cropped = cv2.copyMakeBorder(cropped, pad[0], pad[1], 0, 0, cv2.BORDER_CONSTANT, value=BG_COLOR)

    resized = cv2.resize(cropped, (FINAL_PX, FINAL_PX), interpolation=cv2.INTER_AREA)
    face_box = (x, y, w, h)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), face_box


def draw_guidelines(img, face_box=None):
    """Draw DV photo size & proportion guides."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Outer border
    draw.rectangle([(0, 0), (w-1, h-1)], outline="gray", width=2)

    # Head height guidelines
    head_min = int(h * (1 - HEAD_MAX_RATIO) / 2)
    head_max = int(h * (1 - HEAD_MIN_RATIO) / 2)
    draw.line([(50, head_min), (w-50, head_min)], fill="blue", width=2)
    draw.line([(50, h - head_min), (w-50, h - head_min)], fill="blue", width=2)
    draw.text((60, head_min + 5), "Head height range (1–1⅜ in)", fill="blue")

    # Eye line (approx)
    eye_line_y = int(h * 0.4)
    draw.line([(50, eye_line_y), (w-50, eye_line_y)], fill="green", width=2)
    draw.text((60, eye_line_y + 5), "Eye line ~1.18 in from bottom", fill="green")

    # Inch markers
    inch_px = FINAL_PX // 2
    draw.line([(0, inch_px), (20, inch_px)], fill="black", width=2)
    draw.text((25, inch_px - 10), "1 inch", fill="black")
    draw.line([(inch_px, 0), (inch_px, 20)], fill="black", width=2)
    draw.text((inch_px - 15, 25), "1 inch", fill="black")

    draw.text((10, 10), "2x2 inch (600x600 px)", fill="black")
    return img


# ---------------------- STREAMLIT UI ----------------------

uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img_bytes = uploaded_file.read()
        orig = Image.open(io.BytesIO(img_bytes))
        if orig.mode != "RGB":
            orig = orig.convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig, use_container_width=True)

        with col2:
            st.subheader("✅ Final DV-Compliant Photo")

            # Process
            orig_large = orig.resize((orig.width * 2, orig.height * 2), Image.LANCZOS)
            cleaned = remove_background_smooth(orig_large)
            cropped, face_box = crop_and_resize(cleaned)
            final_preview = draw_guidelines(cropped.copy(), face_box=face_box)

            if isinstance(final_preview, Image.Image):
                st.image(final_preview, caption="DV 2x2 in format", use_container_width=True)
            else:
                st.warning("⚠️ Could not render final preview. Check face detection or image type.")

            # Download button
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV-Ready Photo (600x600)",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
