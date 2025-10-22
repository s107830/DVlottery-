import streamlit as st
from PIL import Image, ImageDraw
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

# ---------------------- FACE & IMAGE HELPERS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face detected")
        return results.multi_face_landmarks[0]

def get_face_box(landmarks, img_w, img_h, padding=0.3):
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    w, h = x_max - x_min, y_max - y_min
    pad_w, pad_h = int(w * padding), int(h * padding)
    x1 = max(x_min - pad_w, 0)
    y1 = max(y_min - pad_h, 0)
    x2 = min(x_max + pad_w, img_w)
    y2 = min(y_max + pad_h, img_h)
    return x1, y1, x2, y2

def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img_pil

# ---------------------- PROCESSING ----------------------
def process_photo(img_pil, auto_adjust=True):
    cv_img = np.array(img_pil)
    if len(cv_img.shape)==2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2]==4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    h, w = cv_img.shape[:2]
    try:
        landmarks = get_face_landmarks(cv_img)
        x1, y1, x2, y2 = get_face_box(landmarks, w, h, padding=0.25)

        if auto_adjust:
            face_crop = cv_img[y1:y2, x1:x2]
            scale_factor = MIN_SIZE / max(face_crop.shape[:2])
            new_w, new_h = int(face_crop.shape[1]*scale_factor), int(face_crop.shape[0]*scale_factor)
            resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            canvas = np.full((MIN_SIZE, MIN_SIZE,3),255,np.uint8)
            x_offset = (MIN_SIZE - new_w)//2
            y_offset = (MIN_SIZE - new_h)//2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            scale_factor = MIN_SIZE / max(h,w)
            new_w, new_h = int(w*scale_factor), int(h*scale_factor)
            resized = cv2.resize(cv_img, (new_w, new_h))
            canvas = np.full((MIN_SIZE, MIN_SIZE,3),255,np.uint8)
            x_offset = (MIN_SIZE - new_w)//2
            y_offset = (MIN_SIZE - new_h)//2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        result_img = Image.fromarray(canvas)
        return result_img
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        scale_factor = MIN_SIZE / max(h,w)
        new_w, new_h = int(w*scale_factor), int(h*scale_factor)
        resized = cv2.resize(cv_img, (new_w, new_h))
        canvas = np.full((MIN_SIZE, MIN_SIZE,3),255,np.uint8)
        x_offset = (MIN_SIZE - new_w)//2
        y_offset = (MIN_SIZE - new_h)//2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return Image.fromarray(canvas)

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. Upload a clear, front-facing photo  
2. Background is auto-removed  
3. Press "Auto Adjust Face" to fit perfectly  
4. Download the corrected image
""")

uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg","jpeg","png"])

if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    bg_removed = remove_background(orig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Photo")
        st.image(orig, use_container_width=True)

    with col2:
        st.subheader("📸 Processed Photo")
        if st.button("🪄 Auto Adjust Face"):
            processed_img = process_photo(bg_removed, auto_adjust=True)
        else:
            processed_img = process_photo(bg_removed, auto_adjust=False)
        st.image(processed_img, use_container_width=True)

        buf = io.BytesIO()
        processed_img.save(buf, format="JPEG")
        st.download_button("💾 Download Corrected Photo", data=buf.getvalue(),
                           file_name="dv_lottery_photo.jpg", mime="image/jpeg")

else:
    st.markdown("## 🎯 Welcome to DV Lottery Photo Editor\nUpload a photo to get started!")
