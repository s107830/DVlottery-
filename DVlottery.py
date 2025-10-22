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

# ---------------------- COMPLIANCE CHECKERS ----------------------
def check_facing_direction(landmarks, img_w, img_h):
    try:
        nose_tip = landmarks.landmark[1]
        left_face = landmarks.landmark[234]
        right_face = landmarks.landmark[454]
        nose_center_ratio = (nose_tip.x - left_face.x) / (right_face.x - left_face.x)
        return 0.4 <= nose_center_ratio <= 0.6
    except:
        return True

def check_eyes_open(landmarks, img_h, img_w):
    try:
        left_eye_openness = abs(landmarks.landmark[159].y - landmarks.landmark[145].y) * img_h
        right_eye_openness = abs(landmarks.landmark[386].y - landmarks.landmark[374].y) * img_h
        return left_eye_openness > 0.01*img_h and right_eye_openness > 0.01*img_h
    except:
        return True

def check_neutral_expression(landmarks):
    try:
        mouth_openness = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        return mouth_openness < 0.05
    except:
        return True

def check_hair_covering_eyes(landmarks, img_h, img_w):
    try:
        left_eye_width = abs(landmarks.landmark[33].x - landmarks.landmark[133].x) * img_w
        right_eye_width = abs(landmarks.landmark[263].x - landmarks.landmark[362].x) * img_w
        return left_eye_width >= 0.05*img_w and right_eye_width >= 0.05*img_w
    except:
        return True

def check_image_quality(cv_img):
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        issues = []
        if blur_value < 50:
            issues.append("Image may be blurry")
        if brightness < 40:
            issues.append("Photo may be too dark")
        elif brightness > 220:
            issues.append("Photo may be overexposed")
        if contrast < 40:
            issues.append("Low contrast detected")
        return issues
    except:
        return []

def comprehensive_compliance_check(cv_img, landmarks, head_info):
    issues = []
    h, w = cv_img.shape[:2]
    if not check_facing_direction(landmarks, w, h):
        issues.append("❌ Face not directly facing camera")
    if not check_eyes_open(landmarks, h, w):
        issues.append("❌ Eyes not fully open or visible")
    if not check_neutral_expression(landmarks):
        issues.append("❌ Non-neutral facial expression detected")
    if not check_hair_covering_eyes(landmarks, h, w):
        issues.append("❌ Hair may be covering eyes or face")
    issues.extend([f"❌ {issue}" for issue in check_image_quality(cv_img)])
    
    head_ratio = head_info["head_height"] / head_info["canvas_size"]
    eye_ratio = (head_info["canvas_size"] - head_info["eye_y"]) / head_info["canvas_size"]
    
    if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO):
        issues.append(f"❌ Head height {int(head_ratio*100)}% out of range")
    if not (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO):
        issues.append(f"❌ Eye position {int(eye_ratio*100)}% out of range")
    return issues

# ---------------------- FACE & IMAGE HELPERS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face detected")
        return results.multi_face_landmarks[0]

def get_head_eye_positions(landmarks, img_h, img_w):
    top_y = int(landmarks.landmark[10].y * img_h)
    chin_y = int(landmarks.landmark[152].y * img_h)
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    return top_y, chin_y, eye_y

def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img_pil

def is_likely_baby_photo(cv_img, landmarks):
    try:
        h, w = cv_img.shape[:2]
        eye_distance = abs(landmarks.landmark[33].x - landmarks.landmark[263].x) * w
        face_height = (landmarks.landmark[152].y - landmarks.landmark[10].y) * h
        return (eye_distance / face_height) > 0.35
    except:
        return False

# ---------------------- PROCESSING ----------------------
def process_photo(img_pil):
    cv_img = np.array(img_pil)
    if len(cv_img.shape)==2: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2]==4: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    h, w = cv_img.shape[:2]
    scale_factor = MIN_SIZE / max(h,w)
    new_w, new_h = int(w*scale_factor), int(h*scale_factor)
    resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((MIN_SIZE, MIN_SIZE,3),255,np.uint8)
    x_offset = (MIN_SIZE - new_w)//2
    y_offset = (MIN_SIZE - new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    result_img = Image.fromarray(canvas)
    
    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_height = chin_y - top_y
        head_info = {"top_y": top_y, "chin_y": chin_y, "eye_y": eye_y, "head_height": head_height, "canvas_size": MIN_SIZE, "is_baby": is_likely_baby_photo(cv_img, landmarks)}
        issues = comprehensive_compliance_check(cv_img, landmarks, head_info)
    except:
        head_info = {"top_y":0,"chin_y":0,"eye_y":0,"head_height":0,"canvas_size":MIN_SIZE,"is_baby":False}
        issues = ["❌ Cannot detect face properly"]
    
    return result_img, head_info, issues

def draw_guidelines(img_pil, head_info):
    draw = ImageDraw.Draw(img_pil)
    w,h = img_pil.size
    cx = w//2
    top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]
    is_baby = head_info.get("is_baby", False)

    head_ratio = head_height/canvas_size
    eye_ratio = (canvas_size-eye_y)/canvas_size

    head_color = "green" if HEAD_MIN_RATIO<=head_ratio<=HEAD_MAX_RATIO else "red"
    eye_color = "green" if EYE_MIN_RATIO<=eye_ratio<=EYE_MAX_RATIO else "red"

    draw.line([(cx-50, top_y),(cx+50, top_y)],fill="blue",width=3)
    draw.text((cx+60, top_y-15),"Head Top",fill="blue")
    draw.line([(cx-50, chin_y),(cx+50, chin_y)],fill="purple",width=3)
    draw.text((cx+60, chin_y-15),"Chin",fill="purple")
    draw.line([(cx, top_y),(cx, chin_y)],fill=head_color,width=2)
    draw.text((cx+10,(top_y+chin_y)//2-20),f"Head: {int(head_ratio*100)}%",fill=head_color)

    eye_min_y = h - int(h*EYE_MAX_RATIO)
    eye_max_y = h - int(h*EYE_MIN_RATIO)
    dash_length = 10
    for x in range(0,w,dash_length*2):
        if x+dash_length<=w:
            draw.line([(x, eye_min_y),(x+dash_length, eye_min_y)],fill="green",width=2)
            draw.line([(x, eye_max_y),(x+dash_length, eye_max_y)],fill="green",width=2)
    draw.line([(0, eye_y),(w, eye_y)],fill=eye_color,width=3)

    if is_baby:
        draw.text((10,10),"Baby Photo Detected",fill="orange")

    return img_pil, head_ratio, eye_ratio

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. Upload a clear front-facing photo
2. Check compliance results
3. Press fix button if measurements are out of range
4. Download corrected photo
""")
st.sidebar.header("⚙️ Settings")
enhance_quality = st.sidebar.checkbox("Enhance Image Quality", value=True)

uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg","jpeg","png"])

if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    bg_removed = remove_background(orig)
    processed_img, head_info, issues = process_photo(bg_removed)
    processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed_img.copy(), head_info)
    
    st.subheader("📷 Original Photo")
    st.image(orig, width=600)
    st.subheader("📸 Processed Photo")
    st.image(processed_with_lines, width=600)
    
    st.subheader("🔍 Compliance Check Results")
    if issues:
        for issue in issues:
            st.write(f"- {issue}")
        st.warning("Photo needs adjustment or replacement")
    else:
        st.success("All compliance checks passed!")
    
else:
    st.markdown("## 🎯 Welcome to DV Lottery Photo Editor\nUpload a photo to get started!")
