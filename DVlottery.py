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
BG_COLOR = (255, 255, 255)

# --- Face detection ---
def detect_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    # largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(map(int, faces[0]))


# --- Background cleanup ---
def remove_background_smooth(img_pil):
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    result = remove(img_byte.getvalue())

    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    fg_np = np.array(fg)
    alpha = fg_np[:, :, 3].astype(np.float32) / 255.0

    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    fg_rgb = fg_np[:, :, :3].astype(np.float32)
    bg_white = np.ones_like(fg_rgb) * 255.0
    clean_rgb = fg_rgb * alpha[..., None] + bg_white * (1 - alpha[..., None])

    clean_img = Image.fromarray(np.uint8(clean_rgb))
    return clean_img.convert("RGB")


# --- Auto-center and crop for DV framing ---
def auto_center_crop(image_pil, face_box):
    img_np = np.array(image_pil)
    h, w, _ = img_np.shape

    if face_box is None:
        # fallback
        img_resized = cv2.resize(img_np, (FINAL_PX, FINAL_PX), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img_resized)

    x, y, fw, fh = face_box
    cx = x + fw // 2
    cy = y + int(fh * 0.55)  # focus more toward face center

    desired_head_height = 360  # midrange of 300–414
    scale = desired_head_height / fh

    # scale + center crop
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cx, cy = int(cx * scale), int(cy * scale)

    left = max(0, cx - FINAL_PX // 2)
    top = max(0, cy - int(FINAL_PX * 0.55))
    right = min(new_w, left + FINAL_PX)
    bottom = min(new_h, top + FINAL_PX)

    cropped = scaled[top:bottom, left:right]
    if cropped.shape[0] != FINAL_PX or cropped.shape[1] != FINAL_PX:
        cropped = cv2.copyMakeBorder(
            cropped,
            0,
            FINAL_PX - cropped.shape[0],
            0,
            FINAL_PX - cropped.shape[1],
            cv2.BORDER_CONSTANT,
            value=BG_COLOR,
        )

    return Image.fromarray(cropped)


# --- Draw DV photo guidelines ---
def draw_guidelines(img, face_box=None):
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Border
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="gray", width=2)

    # Approximate guideline positions
    head_top = h // 2 - int(HEAD_MAX_PX / 2)
    chin = head_top + HEAD_MAX_PX
    eye_y = h - EYE_LINE_MAX_PX

    draw.line([(50, head_top), (w - 50, head_top)], fill="blue", width=2)
    draw.line([(50, chin), (w - 50, chin)], fill="blue", width=2)
    draw.line([(50, eye_y), (w - 50, eye_y)], fill="green", width=2)

    draw.text((60, head_top + 5), "Head top", fill="blue")
    draw.text((60, chin - 20), "Chin line", fill="blue")
    draw.text((60, eye_y + 5), "Eye line", fill="green")

    draw.text((10, 10), "2x2 inch (51x51 mm)", fill="black")
    return img


# --- Streamlit UI ---
st.set_page_config(page_title="DV Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor – Auto Background, Crop & Guidelines")

uploaded = st.file_uploader("Upload your photo (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Processing... please wait"):
        bg_removed = remove_background_smooth(image)
        image_rgb = np.array(bg_removed)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        face_box = detect_face(image_bgr)
        centered = auto_center_crop(bg_removed, face_box)
        final_preview = draw_guidelines(centered)

    st.subheader("✅ Final DV-Compliant Photo")
    st.image(final_preview, caption="DV 2x2 in with guidelines", use_container_width=True)

    buf = io.BytesIO()
    final_preview.save(buf, format="JPEG", quality=95)
    st.download_button(
        "⬇️ Download Final DV Photo",
        data=buf.getvalue(),
        file_name="dv_lottery_photo.jpg",
        mime="image/jpeg",
    )
