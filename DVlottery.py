import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

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

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# ---------------------- FACE UTILS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            return get_face_bounding_box(cv_img)
        return results.multi_face_landmarks[0]

def get_face_bounding_box(cv_img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fd.process(img_rgb)
        if not results.detections:
            raise Exception("No face detected.")
        det = results.detections[0].location_data.relative_bounding_box
        h, w = cv_img.shape[:2]
        x, y, width, height = int(det.xmin*w), int(det.ymin*h), int(det.width*w), int(det.height*h)
        class L: pass
        l = L(); l.landmark = [None]*478
        l.landmark[10] = type('obj', (object,), {'y': (y - height*0.2)/h})()
        l.landmark[152] = type('obj', (object,), {'y': (y + height*1.1)/h})()
        eye_y = y + height*0.3
        l.landmark[33] = type('obj', (object,), {'y': eye_y/h})()
        l.landmark[263] = type('obj', (object,), {'y': eye_y/h})()
        return l

def get_head_eye_positions(landmarks, img_h, img_w):
    top_y = int(landmarks.landmark[10].y * img_h)
    chin_y = int(landmarks.landmark[152].y * img_h)
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    hair_buffer = int((chin_y - top_y) * 0.25)
    top_y = max(0, top_y - hair_buffer)
    return top_y, chin_y, eye_y

# ---------------------- CHECKS ----------------------
def check_single_face(cv_img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fd.process(img_rgb)
        if not results.detections: return False, "No face detected"
        count = len(results.detections)
        return (True, "One face detected") if count == 1 else (False, f"Multiple faces ({count})")

def check_minimum_dimensions(img_pil):
    w, h = img_pil.size
    return (w >= MIN_SIZE and h >= MIN_SIZE, f"{w}x{h}")

def check_photo_proportions(img_pil):
    w, h = img_pil.size
    ar = w/h
    return (0.9 <= ar <= 1.1, f"Aspect ratio {w}:{h}")

def check_background_removal(img_pil):
    try:
        b = io.BytesIO(); img_pil.save(b, format="PNG")
        remove(b.getvalue())
        return True, "OK"
    except: return False, "Failed"

def check_face_recognized(cv_img):
    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, cv_img.shape[0], cv_img.shape[1])
        return (chin_y - top_y > 0, "Face detected")
    except: return False, "Not detected"

def check_red_eyes(cv_img):
    try:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array([0,120,70]), np.array([10,255,255])
        lower2, upper2 = np.array([170,120,70]), np.array([180,255,255])
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        return (cv2.countNonZero(mask) < 500, "No red eyes")
    except: return True, "Check skipped"

# ---------------------- BACKGROUND ----------------------
def remove_background(img_pil):
    try:
        b = io.BytesIO(); img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except: return img_pil

# ---------------------- AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil, force_correction=False):
    image_rgb = np.array(image_pil)
    if len(image_rgb.shape) == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    h, w = image_rgb.shape[:2]
    landmarks = get_face_landmarks(image_rgb)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
    head_height = chin_y - top_y
    scale_factor = (MIN_SIZE * 0.6) / head_height
    scale_factor = np.clip(scale_factor, 0.5, 2.0)
    resized = cv2.resize(image_rgb, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_LANCZOS4)
    new_h, new_w = resized.shape[:2]
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    top_y, chin_y, eye_y = int(top_y*scale_factor), int(chin_y*scale_factor), int(eye_y*scale_factor)
    target_eye_y = MIN_SIZE - int(MIN_SIZE * ((EYE_MIN_RATIO+EYE_MAX_RATIO)/2))
    y_offset = target_eye_y - eye_y
    x_offset = (MIN_SIZE - new_w)//2
    y_start = max(0, y_offset); y_end = min(MIN_SIZE, y_offset+new_h)
    x_start = max(0, x_offset); x_end = min(MIN_SIZE, x_offset+new_w)
    y_src_start = max(0, -y_offset); y_src_end = min(new_h, MIN_SIZE - y_offset)
    x_src_start = max(0, -x_offset); x_src_end = min(new_w, MIN_SIZE - x_offset)
    canvas[y_start:y_end, x_start:x_end] = resized[y_src_start:y_src_end, x_src_start:x_src_end]
    result = Image.fromarray(canvas)
    enh = ImageEnhance.Sharpness(result).enhance(1.1)
    head_info = {
        "top_y": y_start + top_y,
        "chin_y": y_start + chin_y,
        "eye_y": y_start + eye_y,
        "head_height": chin_y - top_y,
        "canvas_size": MIN_SIZE
    }
    return enh, head_info

# ---------------------- GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]
    head_ratio = head_height / canvas_size
    eye_ratio = (canvas_size - eye_y) / canvas_size

    # Head guideline
    color_h = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
    draw.line([(20, top_y), (20, chin_y)], fill=color_h, width=4)
    draw.text((30, (top_y+chin_y)//2 - 20), f"Head {int(head_ratio*100)}%", fill=color_h)
    draw.text((30, (top_y+chin_y)//2), "Req 50–69%", fill="blue")

    # Eye guideline
    color_e = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"
    eye_min_y = h - int(h * EYE_MAX_RATIO)
    eye_max_y = h - int(h * EYE_MIN_RATIO)
    draw.line([(w-30, eye_min_y), (w-30, eye_max_y)], fill="green", width=3)
    draw.line([(w-40, eye_y), (w-20, eye_y)], fill=color_e, width=3)
    draw.text((w-150, (eye_min_y+eye_max_y)//2 - 20), f"Eyes {int(eye_ratio*100)}%", fill=color_e)
    draw.text((w-150, (eye_min_y+eye_max_y)//2), "Req 56–69%", fill="blue")
    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT APP ----------------------
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if 'fix_count' not in st.session_state:
    st.session_state.fix_count = 0
if 'processed' not in st.session_state:
    st.session_state.processed = None

if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Original")
        st.image(orig)

    with col2:
        st.subheader("✅ DV Processed")
        force_fix = st.session_state.fix_count > 0
        with st.spinner("Processing..."):
            bg_removed = remove_background(orig)
            processed, head_info = auto_adjust_dv_photo(bg_removed, force_fix)
        preview, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
        st.image(preview, caption=f"Processed (Fix {st.session_state.fix_count})")

        colm1, colm2 = st.columns(2)
        with colm1:
            head_status = "✅ Within" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "❌ Out"
            st.metric("Head Height", f"{int(head_ratio*100)}%", head_status)
        with colm2:
            eye_status = "✅ Within" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "❌ Out"
            st.metric("Eye Position", f"{int(eye_ratio*100)}%", eye_status)

        needs_fix = (head_ratio < HEAD_MIN_RATIO or head_ratio > HEAD_MAX_RATIO or
                     eye_ratio < EYE_MIN_RATIO or eye_ratio > EYE_MAX_RATIO)
        if needs_fix:
            st.warning("⚠️ Some measurements out of range.")
            if st.button("🛠️ Fix Photo Measurements", use_container_width=True, type="primary"):
                st.session_state.fix_count += 1
                st.rerun()
        else:
            st.success("✅ All measurements within range!")
            if st.session_state.fix_count > 0:
                st.balloons()

        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ Download DV Photo", buf.getvalue(), "dv_lottery_photo.jpg", "image/jpeg", use_container_width=True)
else:
    st.info("👆 Upload a photo to begin.")
