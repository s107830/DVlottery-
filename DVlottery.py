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
def get_face_landmarks(cv_img_small):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        results = fm.process(cv_img_small)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        else:
            return None


def scale_landmarks(landmarks, scale_w, scale_h):
    coords = {}
    for i, lm in enumerate(landmarks.landmark):
        coords[i] = (int(lm.x * scale_w), int(lm.y * scale_h))
    return coords


def check_blur_brightness(cv_img_small):
    gray = cv2.cvtColor(cv_img_small, cv2.COLOR_RGB2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)
    issues = []
    if blur_val < 50: issues.append("Image may be blurry")
    if brightness < 40: issues.append("Photo may be too dark")
    if brightness > 220: issues.append("Photo may be overexposed")
    if contrast < 40: issues.append("Low contrast detected")
    return issues


def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img_pil


def is_baby_photo(coords, h):
    eye_dist = abs(coords[33][0] - coords[263][0])
    chin_y = coords[152][1]
    top_y = coords[10][1]
    face_height = chin_y - top_y
    if face_height <= 0: return False
    ratio = eye_dist / face_height
    return ratio > 0.35


def compute_head_eye_info(coords):
    top_y = coords[10][1]
    chin_y = coords[152][1]
    eye_y = (coords[33][1] + coords[263][1]) // 2
    head_height = chin_y - top_y
    return top_y, chin_y, eye_y, head_height


def compliance_check(coords, cv_img_small, head_height):
    issues = check_blur_brightness(cv_img_small)
    face_ratio = head_height / MIN_SIZE
    if not (HEAD_MIN_RATIO <= face_ratio <= HEAD_MAX_RATIO):
        issues.append(f"Head height {int(face_ratio*100)}% not in required range")
    return issues


def draw_guidelines(img_pil, top_y, chin_y, eye_y, head_height, is_baby=False):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    cx = w // 2
    head_ratio = head_height / MIN_SIZE
    eye_ratio = (MIN_SIZE - eye_y) / MIN_SIZE
    head_color = 'green' if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else 'red'
    eye_color = 'green' if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else 'red'
    draw.line([(cx-50, top_y),(cx+50, top_y)], fill='blue', width=3)
    draw.line([(cx-50, chin_y),(cx+50, chin_y)], fill='purple', width=3)
    draw.line([(cx, top_y),(cx, chin_y)], fill=head_color, width=2)
    eye_min_y = h - int(h*EYE_MAX_RATIO)
    eye_max_y = h - int(h*EYE_MIN_RATIO)
    draw.line([(0, eye_min_y),(w, eye_min_y)], fill='green', width=2)
    draw.line([(0, eye_max_y),(w, eye_max_y)], fill='green', width=2)
    draw.line([(0, eye_y),(w, eye_y)], fill=eye_color, width=2)
    if is_baby: draw.text((10,10), '👶 Baby Photo Detected', fill='orange')
    return img_pil

# ---------------------- MAIN PROCESSING ----------------------
def process_photo(img_pil):
    img_small = img_pil.resize((300,300))
    cv_img_small = np.array(img_small)
    if cv_img_small.shape[2]==4: cv_img_small = cv2.cvtColor(cv_img_small, cv2.COLOR_RGBA2RGB)
    landmarks = get_face_landmarks(cv_img_small)
    if landmarks is None:
        return img_pil, None, ['❌ Cannot detect face']
    scale_w = MIN_SIZE / 300
    scale_h = MIN_SIZE / 300
    coords = scale_landmarks(landmarks, MIN_SIZE, MIN_SIZE)
    top_y, chin_y, eye_y, head_height = compute_head_eye_info(coords)
    issues = compliance_check(coords, cv_img_small, head_height)
    is_baby = is_baby_photo(coords, MIN_SIZE)
    result_img = draw_guidelines(img_pil.copy(), top_y, chin_y, eye_y, head_height, is_baby)
    return result_img, {'top_y':top_y,'chin_y':chin_y,'eye_y':eye_y,'head_height':head_height,'is_baby':is_baby}, issues

# ---------------------- STREAMLIT UI ----------------------
with st.sidebar:
    st.header('📋 Instructions')
    st.markdown('Upload photo → Check compliance → Fix if needed → Download')

uploaded_file = st.file_uploader('📤 Upload Your Photo', type=['jpg','jpeg','png'])

if uploaded_file:
    orig = Image.open(uploaded_file).convert('RGB')
    bg_removed = remove_background(orig)
    processed_img, head_info, issues = process_photo(bg_removed)
    
    st.subheader('📸 Processed Photo')
    st.image(processed_img, use_container_width=True)

    st.subheader('🔍 Compliance Check')
    if issues:
        for i in issues: st.error(i)
    else:
        st.success('✅ All checks passed!')

    # Auto-adjust button (lazy evaluation)
    if issues:
        if st.button('🔧 Auto-Adjust'):
            # You can implement process_dv_photo_adjusted() here with scaled coordinates
            st.info('Auto-adjustment would run here.')

    # Download
    buf = io.BytesIO()
    processed_img.save(buf, format='JPEG', quality=95)
    st.download_button('⬇️ Download Photo', buf.getvalue(), file_name='dv_photo.jpg', mime='image/jpeg')

else:
    st.info('👆 Upload your photo to start processing')
