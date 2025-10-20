import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor AI", layout="wide")
st.title("📸 DV Lottery Photo Editor — Fully Automated")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
MAX_SIZE = 1200
HEAD_MIN_RATIO = 0.50
HEAD_MAX_RATIO = 0.69
EYE_MIN_RATIO = 0.56
EYE_MAX_RATIO = 0.69
BG_COLOR = (255, 255, 255)

# ---------------------- AI FACE DETECTION ----------------------
mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face detected.")
        return results.multi_face_landmarks[0]

def get_head_eye_positions(landmarks, img_h, img_w):
    top_idx = 10
    chin_idx = 152
    left_eye_idx = 159
    right_eye_idx = 386

    top_y = int(landmarks.landmark[top_idx].y * img_h)
    chin_y = int(landmarks.landmark[chin_idx].y * img_h)
    left_eye_y = int(landmarks.landmark[left_eye_idx].y * img_h)
    right_eye_y = int(landmarks.landmark[right_eye_idx].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2

    return top_y, chin_y, eye_y

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    img_byte = img_byte.getvalue()
    result = remove(img_byte)
    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    white_bg = Image.new("RGBA", fg.size, BG_COLOR + (255,))
    composite = Image.alpha_composite(white_bg, fg)
    return composite.convert("RGB")

# ---------------------- AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil):
    image_rgb = np.array(image_pil)
    img_h, img_w = image_rgb.shape[:2]

    landmarks = get_face_landmarks(image_rgb)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, img_h, img_w)
    head_height = chin_y - top_y

    target_head_ratio = (HEAD_MIN_RATIO + HEAD_MAX_RATIO) / 2
    target_eye_ratio = (EYE_MIN_RATIO + EYE_MAX_RATIO) / 2

    # Scale factor based on head height
    scale_factor = target_head_ratio / (head_height / img_h)
    new_h = int(img_h * scale_factor)
    new_w = int(img_w * scale_factor)
    resized_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top_y_scaled = int(top_y * scale_factor)
    eye_y_scaled = int(eye_y * scale_factor)

    # Canvas size
    canvas_size = max(new_w, new_h)
    canvas_size = max(MIN_SIZE, min(MAX_SIZE, canvas_size))
    canvas = np.full((canvas_size, canvas_size, 3), BG_COLOR, dtype=np.uint8)

    # Vertical placement
    eye_target = int(canvas_size * target_eye_ratio)
    top_offset = eye_target - eye_y_scaled
    top_offset = max(min(top_offset, canvas_size - new_h), 0)

    # Horizontal center
    left_offset = (canvas_size - new_w) // 2
    left_offset = max(left_offset, 0)

    # Paste
    y1 = top_offset
    y2 = top_offset + new_h
    x1 = left_offset
    x2 = left_offset + new_w
    y2 = min(y2, canvas_size)
    x2 = min(x2, canvas_size)
    h_clip = y2 - y1
    w_clip = x2 - x1
    if h_clip > 0 and w_clip > 0:
        canvas[y1:y2, x1:x2] = resized_img[0:h_clip, 0:w_clip]

    return Image.fromarray(canvas)

# ---------------------- DRAW GUIDELINES ----------------------
def draw_guidelines(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    head_min = int(h * HEAD_MIN_RATIO)
    head_max = int(h * HEAD_MAX_RATIO)
    eye_min = int(h * EYE_MIN_RATIO)
    eye_max = int(h * EYE_MAX_RATIO)

    draw.rectangle([(0, 0), (w-1, h-1)], outline="gray", width=2)
    draw.line([(0, h - head_max), (w, h - head_max)], fill="blue", width=2)
    draw.line([(0, h - head_min), (w, h - head_min)], fill="blue", width=2)
    draw.text((10, h - head_max + 5), "Head height", fill="blue")
    draw.line([(0, h - eye_max), (w, h - eye_max)], fill="green", width=2)
    draw.line([(0, h - eye_min), (w, h - eye_min)], fill="green", width=2)
    draw.text((10, h - eye_max + 5), "Eye line", fill="green")
    return img

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG)", type=["jpg", "jpeg"])
if uploaded_file:
    try:
        img_bytes = uploaded_file.read()
        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig, caption="Original")

        with col2:
            st.subheader("✅ Fully Automated DV Compliant")
            bg_removed = remove_background(orig)
            processed = auto_adjust_dv_photo(bg_removed)
            final_preview = draw_guidelines(processed.copy())
            st.image(final_preview, caption="DV Compliance Preview")

            # Download button
            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV Photo",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg"
            )
    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
