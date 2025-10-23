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
st.title("DV Lottery Photo Editor — Baby Aware + Auto Fit + Official Guidelines")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69
DPI = 300  # 2x2 inch photo at 300 DPI
mp_face_mesh = mp.solutions.face_mesh

# ---------------------- FACE UTILITIES ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    ) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return results.multi_face_landmarks[0]

def get_head_eye_positions(landmarks, img_h, img_w):
    top_y = int(landmarks.landmark[10].y * img_h)
    chin_y = int(landmarks.landmark[152].y * img_h)
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    hair_buffer = int((chin_y - top_y) * 0.25)
    top_y = max(0, top_y - hair_buffer)
    return top_y, chin_y, eye_y

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img_pil

# ---------------------- BABY DETECTION ----------------------
def is_baby_photo(landmarks, img_h, img_w):
    """Detects baby facial proportions based on eye and chin placement."""
    try:
        top_y = landmarks.landmark[10].y * img_h
        chin_y = landmarks.landmark[152].y * img_h
        left_eye_y = landmarks.landmark[33].y * img_h
        right_eye_y = landmarks.landmark[263].y * img_h
        eye_y = (left_eye_y + right_eye_y) / 2
        nose_y = landmarks.landmark[1].y * img_h

        face_height = chin_y - top_y
        eye_to_top = eye_y - top_y
        ratio = eye_to_top / face_height
        jaw_ratio = (chin_y - nose_y) / face_height

        # Babies have higher eyes and smaller chin area
        return ratio > 0.42 and jaw_ratio < 0.33
    except:
        return False

# ---------------------- AUTO CROP ----------------------
def auto_crop_dv(img_pil):
    """Auto-crops photo to DV specs with baby detection and head-fit fix."""
    cv_img = np.array(img_pil)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    h, w = cv_img.shape[:2]
    landmarks = get_face_landmarks(cv_img)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
    head_h = chin_y - top_y

    baby_mode = is_baby_photo(landmarks, h, w)
    target_head = MIN_SIZE * (0.58 if baby_mode else 0.63)
    scale = target_head / head_h

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    target_eye_min = MIN_SIZE - int(EYE_MAX_RATIO * MIN_SIZE)
    target_eye_max = MIN_SIZE - int(EYE_MIN_RATIO * MIN_SIZE)
    target_eye = (target_eye_min + target_eye_max) // 2

    landmarks_resized = get_face_landmarks(resized)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
    y_offset = target_eye - eye_y
    x_offset = (MIN_SIZE - new_w) // 2

    # ----- auto zoom-out if head might get cut -----
    if (top_y + y_offset < 0) or (chin_y + y_offset > MIN_SIZE):
        y_offset = min(max(y_offset, 0), MIN_SIZE - new_h)
        scale *= 0.9  # zoom out slightly
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        landmarks_resized = get_face_landmarks(resized)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
        y_offset = target_eye - eye_y
        x_offset = (MIN_SIZE - new_w) // 2

    # ----- paste to white canvas -----
    y_start_dst = max(0, y_offset)
    y_end_dst = min(MIN_SIZE, y_offset + new_h)
    x_start_dst = max(0, x_offset)
    x_end_dst = min(MIN_SIZE, x_offset + new_w)

    y_start_src = max(0, -y_offset)
    y_end_src = min(new_h, MIN_SIZE - y_offset)
    x_start_src = max(0, -x_offset)
    x_end_src = min(new_w, MIN_SIZE - x_offset)

    canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
        resized[y_start_src:y_end_src, x_start_src:x_end_src]

    final_top_y = top_y + y_offset
    final_chin_y = chin_y + y_offset
    final_eye_y = eye_y + y_offset

    head_info = {
        "top_y": final_top_y,
        "chin_y": final_chin_y,
        "eye_y": final_eye_y,
        "head_height": chin_y - top_y,
        "canvas_size": MIN_SIZE,
        "is_baby": baby_mode
    }
    return Image.fromarray(canvas), head_info

# ---------------------- DRAW DV GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w // 2
    top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_h = head_info["head_height"]
    head_ratio = head_h / h
    eye_ratio = (h - eye_y) / h

    head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
    eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

    eye_min_px = int(1.125 * DPI)
    eye_max_px = int(1.375 * DPI)
    eye_band_top = h - eye_max_px
    eye_band_bottom = h - eye_min_px

    draw.line([(0, top_y), (w, top_y)], fill="red", width=3)
    draw.line([(0, eye_y), (w, eye_y)], fill="red", width=3)
    draw.line([(0, chin_y), (w, chin_y)], fill="red", width=3)

    for x in range(0, w, 20):
        draw.line([(x, eye_band_top), (x + 10, eye_band_top)], fill="green", width=2)
        draw.line([(x, eye_band_bottom), (x + 10, eye_band_bottom)], fill="green", width=2)

    draw.text((10, top_y - 25), "Top of Head", fill="red")
    draw.text((10, eye_y - 15), "Eye Line", fill="red")
    draw.text((10, chin_y - 20), "Chin", fill="red")
    draw.text((w - 240, eye_band_top - 20), "1 inch to 1-3/8 inch", fill="green")
    draw.text((w - 300, eye_band_bottom + 5), "1-1/8 inch to 1-3/8 inch from bottom", fill="green")

    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="black", width=3)
    draw.line([(cx, 0), (cx, h)], fill="gray", width=1)

    # inch rulers
    inch_px = DPI
    for i in range(3):
        y = i * inch_px
        draw.line([(0, y), (20, y)], fill="black", width=2)
        draw.text((25, y - 10), f"{i} in", fill="black")
        draw.line([(w - 20, y), (w, y)], fill="black", width=2)
        draw.text((w - 55, y - 10), f"{i} in", fill="black")

    # ----- PASS / FAIL LOGIC -----
    head_in_frame = (head_info["top_y"] > 5) and (head_info["chin_y"] < h - 5)
    passed = (
        HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
        and EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
        and head_in_frame
    )

    badge_color = "green" if passed else "red"
    status_text = "PASS" if passed else "FAIL"
    draw.rectangle([(10, 10), (200, 75)], fill="white", outline=badge_color, width=3)
    draw.text((20, 20), status_text, fill=badge_color)
    draw.text((20, 40), f"H:{int(head_ratio*100)}%  E:{int(eye_ratio*100)}%", fill="black")

    if not head_in_frame:
        draw.text((20, 60), "Head cropped - FAIL", fill="red")

    if head_info.get("is_baby", False):
        draw.text((20, 80), "Baby Mode Active", fill="orange")

    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a front-facing photo.
2. Background auto-removed & cropped to 2x2 inch.
3. Detects baby faces & adjusts scaling.
4. Automatically fails or zooms out if head cropped.

**DV Requirements:**
- Head height: 50–69%
- Eyes: 1-1/8–1-3/8 inch from bottom
- White background, neutral expression
""")

uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded:
    orig = Image.open(uploaded).convert("RGB")
    with st.spinner("Processing photo..."):
        bg_removed = remove_background(orig)
        processed, head_info = auto_crop_dv(bg_removed)
        overlay, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(orig, use_column_width=True)
    with col2:
        st.subheader("Processed (600x600)")
        st.image(overlay, use_column_width=True)

        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Final 600x600 Photo",
            data=buf.getvalue(),
            file_name="dv_photo_final.jpg",
            mime="image/jpeg"
        )
else:
    st.markdown("""
    ## Welcome to the DV Lottery Photo Editor  
    Upload your photo to generate a perfect 600x600 DV-compliant image.  
    Baby faces are auto-detected and scaling adjusted to fit full head.  
    Cropped heads now automatically FAIL.
    """)

st.markdown("---")
st.caption("DV Lottery Photo Editor | Baby Detection + Auto Fit + Cropped Head Fail Check")
