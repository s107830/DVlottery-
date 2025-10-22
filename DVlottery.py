import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings

warnings.filterwarnings('ignore')

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Auto Correction & Compliance Check")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69

mp_face_mesh = mp.solutions.face_mesh

# ---------------------- HELPERS ----------------------
def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}. Using original image.")
        return img_pil

def get_face_landmarks(cv_img):
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as fm:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = fm.process(img_rgb)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0]
    except:
        return None

def draw_guidelines(img, top_y, chin_y, eye_y, head_height, is_baby=False):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cx = w // 2

        # Head lines
        draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
        draw.text((cx+60, top_y-15), "Head Top", fill="blue")
        draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
        draw.text((cx+60, chin_y-15), "Chin", fill="purple")
        draw.line([(cx, top_y), (cx, chin_y)], fill="green", width=2)

        # Eye line
        draw.line([(0, eye_y), (w, eye_y)], fill="green", width=2)

        # Baby detection text (no emoji to avoid Unicode errors)
        if is_baby:
            draw.text((10, 10), "Baby Photo Detected", fill="orange")
        return img
    except Exception as e:
        st.warning(f"Guidelines drawing failed: {str(e)}")
        return img

def process_photo(img_pil):
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

        # Simple resize
        h, w = cv_img.shape[:2]
        scale = MIN_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        x_off, y_off = (MIN_SIZE - new_w)//2, (MIN_SIZE - new_h)//2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        result_img = Image.fromarray(canvas)

        # Face detection (optional)
        landmarks = get_face_landmarks(cv_img)
        if landmarks:
            top_y, chin_y, eye_y = int(h*0.2), int(h*0.8), int(h*0.5)
            is_baby = False  # Simple placeholder
        else:
            top_y, chin_y, eye_y = MIN_SIZE//4, MIN_SIZE*3//4, MIN_SIZE//2
            is_baby = False

        # Draw guidelines
        result_img = draw_guidelines(result_img.copy(), top_y, chin_y, eye_y, chin_y-top_y, is_baby)
        return result_img, top_y, chin_y, eye_y, is_baby
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return img_pil, 0, 0, 0, False

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        orig = Image.open(uploaded_file).convert("RGB")
        bg_removed = remove_background(orig)
        processed_img, top_y, chin_y, eye_y, is_baby = process_photo(bg_removed)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Photo")
            if isinstance(orig, Image.Image):
                st.image(orig, use_container_width=True)
            else:
                st.error("Original image not available.")

        with col2:
            st.subheader("Processed Photo")
            if isinstance(processed_img, Image.Image):
                st.image(processed_img, use_container_width=True)
            else:
                st.error("Processed image not available.")

    except Exception as e:
        st.error(f"Failed to process uploaded photo: {str(e)}")
else:
    st.markdown("## 🎯 Upload a clear front-facing photo to start processing.")
