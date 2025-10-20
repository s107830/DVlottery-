import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import io
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Auto Crop, Background & DV Guidelines")

# ---------------------- CONSTANTS ----------------------
FINAL_PX = 600            # 2x2 inch photo
HEAD_MIN_RATIO = 0.50     # 50% of image height (300 px)
HEAD_MAX_RATIO = 0.69     # 69% of image height (414 px)
EYE_MIN_RATIO = 0.56      # 56% of height from bottom
EYE_MAX_RATIO = 0.69      # 69% of height from bottom
BG_COLOR = (255, 255, 255)

# ---------------------- FUNCTIONS ----------------------

def remove_background_smooth(img_pil):
    """Remove background and smooth hair edges."""
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    img_byte = img_byte.getvalue()
    result = remove(img_byte)
    fg = Image.open(io.BytesIO(result)).convert("RGBA")

    alpha = fg.split()[3]
    alpha_np = np.array(alpha)
    kernel = np.ones((3, 3), np.uint8)
    alpha_np = cv2.dilate(alpha_np, kernel, iterations=1)
    alpha_np = cv2.GaussianBlur(alpha_np, (5, 5), 0)
    alpha = Image.fromarray(alpha_np)
    fg.putalpha(alpha)

    white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, fg)
    return composite.convert("RGB")

def detect_face(cv_img):
    """Detect face bounding box."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        raise Exception("No face detected. Please upload a clear, frontal photo.")
    return max(faces, key=lambda f: f[2] * f[3])  # largest face

def crop_and_resize(image_pil):
    """Crop around face and resize to 600x600."""
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = detect_face(image_bgr)

    height, width = image_bgr.shape[:2]
    top = max(0, y - int(0.45 * h))
    bottom = min(height, y + h + int(0.35 * h))
    left = max(0, x - int(0.25 * w))
    right = min(width, x + w + int(0.25 * w))
    cropped = image_bgr[top:bottom, left:right]

    # pad to square
    c_h, c_w = cropped.shape[:2]
    diff = abs(c_h - c_w)
    if c_h > c_w:
        pad = (diff // 2, diff - diff // 2)
        cropped = cv2.copyMakeBorder(cropped, 0, 0, pad[0], pad[1], cv2.BORDER_CONSTANT, value=BG_COLOR)
    elif c_w > c_h:
        pad = (diff // 2, diff - diff // 2)
        cropped = cv2.copyMakeBorder(cropped, pad[0], pad[1], 0, 0, cv2.BORDER_CONSTANT, value=BG_COLOR)

    resized = cv2.resize(cropped, (FINAL_PX, FINAL_PX), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), (x, y, w, h)

def draw_guidelines(img):
    """Draw official DV guideline overlay."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # center dashed vertical line
    for y in range(0, h, 10):
        draw.line([(w // 2, y), (w // 2, y + 5)], fill="gray", width=1)

    # Head height range (50–69%)
    head_top = int(h * (1 - HEAD_MAX_RATIO) / 2)
    head_bottom = h - head_top
    draw.line([(0, head_top), (w, head_top)], fill="blue", width=2)
    draw.line([(0, head_bottom), (w, head_bottom)], fill="blue", width=2)
    draw.text((10, head_top + 5), "Head height: 300–414 px (50–69%)", fill="blue")

    # Eye position range (56–69% from bottom)
    eye_min_y = h - int(h * EYE_MAX_RATIO)
    eye_max_y = h - int(h * EYE_MIN_RATIO)
    draw.line([(0, eye_min_y), (w, eye_min_y)], fill="green", width=2)
    draw.line([(0, eye_max_y), (w, eye_max_y)], fill="green", width=2)
    draw.text((10, eye_min_y - 15), "Eye line: 336–414 px from bottom", fill="green")

    # Border & label
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="gray", width=2)
    draw.text((10, 10), "2x2 inch (600x600 px)", fill="black")
    return img

def check_head_ratio(face_box, img_height):
    """Check if head height is within DV limits."""
    _, y, _, h = face_box
    ratio = h / img_height
    if ratio < HEAD_MIN_RATIO:
        return f"⚠️ Head too small ({ratio:.2f}), increase zoom or move closer."
    elif ratio > HEAD_MAX_RATIO:
        return f"⚠️ Head too large ({ratio:.2f}), move farther or crop less."
    return "✅ Head size within DV requirements."

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
            st.image(orig, use_column_width=True)

        with col2:
            st.subheader("✅ Processed (DV Compliant)")
            orig_large = orig.resize((orig.width * 2, orig.height * 2), Image.LANCZOS)
            bg_removed = remove_background_smooth(orig_large)
            processed, face_box = crop_and_resize(bg_removed)

            # 🔘 Toggle guidelines
            show_guidelines = st.toggle("Show DV Guidelines", value=True)

            if show_guidelines:
                final_preview = draw_guidelines(processed.copy())
                st.image(final_preview, caption="DV Compliance Preview", use_column_width=True)
            else:
                st.image(processed, caption="Clean Photo (No Guidelines)", use_column_width=True)

            # Head size check
            message = check_head_ratio(face_box, FINAL_PX)
            st.markdown(f"**{message}**")

            # Download
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
