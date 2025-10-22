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

# ---------------------- HELPER FUNCTIONS ----------------------
def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
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

def get_head_eye_positions(landmarks, img_h, img_w):
    top_y = int(landmarks.landmark[10].y * img_h)
    chin_y = int(landmarks.landmark[152].y * img_h)
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    return top_y, chin_y, eye_y

def is_likely_baby(cv_img, landmarks):
    try:
        h, w = cv_img.shape[:2]
        eye_distance = abs(landmarks.landmark[33].x - landmarks.landmark[263].x) * w
        face_height = (landmarks.landmark[152].y - landmarks.landmark[10].y) * h
        ratio = eye_distance / face_height
        # Only consider ratio < 0.3 and face small for baby
        return ratio < 0.32 and face_height < 220
    except:
        return False

def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w // 2
    top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]
    is_baby = head_info.get("is_baby", False)

    head_ratio = head_height / canvas_size
    eye_ratio = (canvas_size - eye_y) / canvas_size

    head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
    eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

    draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
    draw.text((cx+60, top_y-15), "Head Top", fill="blue")
    draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
    draw.text((cx+60, chin_y-15), "Chin", fill="purple")
    draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=2)
    draw.text((cx + 10, (top_y + chin_y)//2 - 20), f"Head: {int(head_ratio*100)}%", fill=head_color)
    draw.text((cx + 10, (top_y + chin_y)//2), f"Req: {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%", fill="blue")
    
    eye_min_y = h - int(h * EYE_MAX_RATIO)
    eye_max_y = h - int(h * EYE_MIN_RATIO)
    dash_length = 10
    for x in range(0, w, dash_length*2):
        if x + dash_length <= w:
            draw.line([(x, eye_min_y), (x+dash_length, eye_min_y)], fill="green", width=2)
            draw.line([(x, eye_max_y), (x+dash_length, eye_max_y)], fill="green", width=2)
    draw.text((10, eye_min_y-15), "56%", fill="green")
    draw.text((10, eye_max_y-15), "69%", fill="green")
    draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
    draw.text((w-150, eye_y-15), f"Eyes: {int(eye_ratio*100)}%", fill=eye_color)
    
    if is_baby:
        draw.text((10, 10), "👶 Baby Photo Detected", fill="orange")
    
    return img, head_ratio, eye_ratio

# ---------------------- PHOTO PROCESSING ----------------------
def process_photo(img_pil):
    cv_img = np.array(img_pil)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    h, w = cv_img.shape[:2]

    # Resize to canvas
    scale_factor = MIN_SIZE / max(h, w)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    y_offset = (MIN_SIZE - new_h) // 2
    x_offset = (MIN_SIZE - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    result_img = Image.fromarray(canvas)

    # Landmarks
    landmarks = get_face_landmarks(cv_img)
    if landmarks:
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_height = chin_y - top_y
        is_baby = is_likely_baby(cv_img, landmarks)
    else:
        top_y, chin_y, eye_y = MIN_SIZE//4, MIN_SIZE*3//4, MIN_SIZE//2
        head_height = chin_y - top_y
        is_baby = False

    head_info = {
        "top_y": top_y,
        "chin_y": chin_y,
        "eye_y": eye_y,
        "head_height": head_height,
        "canvas_size": MIN_SIZE,
        "is_baby": is_baby
    }

    return result_img, head_info

def auto_fix_photo(img_pil, head_info):
    """Auto-fix head/chin/eye position to DV ratios"""
    cv_img = np.array(img_pil)
    landmarks = get_face_landmarks(cv_img)
    if not landmarks:
        return img_pil, head_info

    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, cv_img.shape[0], cv_img.shape[1])
    head_height = chin_y - top_y
    eye_ratio = (MIN_SIZE - eye_y) / MIN_SIZE
    head_ratio = head_height / MIN_SIZE

    # Scale factor to match target head height
    target_head_height = (HEAD_MIN_RATIO + HEAD_MAX_RATIO)/2 * MIN_SIZE
    scale_factor = target_head_height / head_height
    scaled_w = int(cv_img.shape[1]*scale_factor)
    scaled_h = int(cv_img.shape[0]*scale_factor)
    resized = cv2.resize(cv_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Shift vertically to center eyes in range
    new_top_y = int(top_y*scale_factor)
    new_chin_y = int(chin_y*scale_factor)
    new_eye_y = int(eye_y*scale_factor)
    target_eye_y = int(MIN_SIZE - (EYE_MIN_RATIO + EYE_MAX_RATIO)/2 * MIN_SIZE)
    y_offset = target_eye_y - new_eye_y
    x_offset = (MIN_SIZE - scaled_w)//2

    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    y_start_dst = max(0, y_offset)
    y_end_dst = min(MIN_SIZE, y_offset+scaled_h)
    x_start_dst = max(0, x_offset)
    x_end_dst = min(MIN_SIZE, x_offset+scaled_w)

    y_start_src = max(0, -y_offset)
    y_end_src = min(scaled_h, MIN_SIZE - y_offset)
    x_start_src = max(0, 0)
    x_end_src = x_start_src + (x_end_dst - x_start_dst)

    if y_start_dst < y_end_dst and x_start_dst < x_end_dst:
        canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = resized[y_start_src:y_end_src, x_start_src:x_end_src]
    
    new_top_y += y_offset
    new_chin_y += y_offset
    new_eye_y += y_offset

    result_img = Image.fromarray(canvas)
    new_head_info = {
        "top_y": new_top_y,
        "chin_y": new_chin_y,
        "eye_y": new_eye_y,
        "head_height": target_head_height,
        "canvas_size": MIN_SIZE,
        "is_baby": head_info.get("is_baby", False)
    }
    return result_img, new_head_info

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg","jpeg","png"])

if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    bg_removed = remove_background(orig)
    processed_img, head_info = process_photo(bg_removed)
    processed_with_guides, head_ratio, eye_ratio = draw_guidelines(processed_img.copy(), head_info)

    st.subheader("📷 Original Photo")
    st.image(orig)

    st.subheader("📸 Processed Photo with Guidelines")
    st.image(processed_with_guides)

    st.subheader("📊 Measurements")
    st.write(f"- Head height: {head_ratio*100:.1f}% (50–69%)")
    st.write(f"- Eye position: {eye_ratio*100:.1f}% (56–69%)")

    needs_fix = not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO and EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO)
    if needs_fix:
        st.warning("Photo needs adjustment or replacement")
        if st.button("🔧 Auto-Fix Photo"):
            fixed_img, fixed_head_info = auto_fix_photo(bg_removed, head_info)
            fixed_with_guides, head_ratio, eye_ratio = draw_guidelines(fixed_img.copy(), fixed_head_info)
            st.subheader("✅ Auto-Fixed Photo")
            st.image(fixed_with_guides)
            st.write(f"- Head height: {head_ratio*100:.1f}% (50–69%)")
            st.write(f"- Eye position: {eye_ratio*100:.1f}% (56–69%)")
            buf = io.BytesIO()
            fixed_img.save(buf, format="JPEG")
            st.download_button("⬇️ Download Fixed Photo", buf.getvalue(), "dv_lottery_fixed.jpg", "image/jpeg")
    else:
        st.success("Photo already compliant ✅")
        buf = io.BytesIO()
        processed_img.save(buf, format="JPEG")
        st.download_button("⬇️ Download Photo", buf.getvalue(), "dv_lottery.jpg", "image/jpeg")
else:
    st.info("Upload a clear front-facing photo to start processing")
