import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2, io, warnings
import mediapipe as mp
from rembg import remove
warnings.filterwarnings('ignore')

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — Shoulders to Head Only")

MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69
mp_face_mesh = mp.solutions.face_mesh

# ---------------------- FACE LANDMARKS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.3) as fm:
        results = fm.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return results.multi_face_landmarks[0]

def get_head_positions(landmarks, h, w):
    top_y = int(landmarks.landmark[10].y * h)
    chin_y = int(landmarks.landmark[152].y * h)
    left_eye_y = int(landmarks.landmark[33].y * h)
    right_eye_y = int(landmarks.landmark[263].y * h)
    eye_y = (left_eye_y + right_eye_y) // 2
    face_h = chin_y - top_y
    shoulder_y = chin_y + int(face_h * 0.8)
    top_y = max(0, top_y - int(face_h * 0.15))
    return top_y, chin_y, eye_y, shoulder_y

def is_baby_photo(landmarks, h, w):
    try:
        top_y = landmarks.landmark[10].y * h
        chin_y = landmarks.landmark[152].y * h
        left_eye_y = landmarks.landmark[33].y * h
        right_eye_y = landmarks.landmark[263].y * h
        eye_y = (left_eye_y + right_eye_y) / 2
        ratio = (eye_y - top_y) / (chin_y - top_y)
        return ratio > 0.38
    except:
        return False

# ---------------------- BACKGROUND REMOVE ----------------------
def remove_bg(img):
    try:
        b = io.BytesIO()
        img.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img

# ---------------------- AUTO CROP ----------------------
def auto_crop(img_pil):
    cv_img = np.array(img_pil)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    h, w = cv_img.shape[:2]

    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y, shoulder_y = get_head_positions(landmarks, h, w)
        is_baby = is_baby_photo(landmarks, h, w)
    except:
        resized = cv2.resize(cv_img, (MIN_SIZE, MIN_SIZE))
        return Image.fromarray(resized), {"top_y":150,"chin_y":400,"eye_y":270,"shoulder_y":480,"head_height":250,"is_baby":False}

    scale = (MIN_SIZE * 0.85) / (shoulder_y - top_y)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_img, (new_w, new_h))
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)

    y_center = (top_y + shoulder_y) // 2
    y_offset = MIN_SIZE//2 - y_center
    x_offset = (MIN_SIZE - new_w)//2

    y_offset = max(min(y_offset, MIN_SIZE - new_h), 0)
    x_offset = max(min(x_offset, MIN_SIZE - new_w), 0)
    y_end = min(MIN_SIZE, y_offset + new_h)
    x_end = min(MIN_SIZE, x_offset + new_w)
    canvas[y_offset:y_end, x_offset:x_end] = resized[:y_end - y_offset, :x_end - x_offset]

    head_info = {"top_y": top_y + y_offset, "chin_y": chin_y + y_offset,
                 "eye_y": eye_y + y_offset, "shoulder_y": shoulder_y + y_offset,
                 "head_height": chin_y - top_y, "is_baby": is_baby}
    return Image.fromarray(canvas), head_info

# ---------------------- DRAW LINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    t, c, e = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_h = head_info["head_height"]
    head_r, eye_r = head_h / h, (h - e) / h

    if head_info["is_baby"]:
        hmin, hmax, emin, emax = 0.45, 0.72, 0.48, 0.65
    else:
        hmin, hmax, emin, emax = HEAD_MIN_RATIO, HEAD_MAX_RATIO, EYE_MIN_RATIO, EYE_MAX_RATIO

    ok = hmin <= head_r <= hmax and emin <= eye_r <= emax
    color = "green" if ok else "red"

    draw.line([(0, t), (w, t)], fill="green", width=2)
    draw.line([(0, e), (w, e)], fill="orange", width=2)
    draw.line([(0, c), (w, c)], fill="red", width=2)
    draw.rectangle([(10, 10), (220, 85)], outline=color, fill="white", width=3)
    draw.text((20, 15), "PASS" if ok else "FAIL", fill=color)
    draw.text((20, 35), f"Head: {int(head_r*100)}%", fill="black")
    draw.text((20, 50), f"Eyes: {int(eye_r*100)}%", fill="black")
    if head_info["is_baby"]:
        draw.text((20, 65), "BABY MODE", fill="orange")

    return img, head_r, eye_r

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("DV Lottery Requirements")
st.sidebar.markdown("""
- 600×600px, white background  
- Head: 50–69% (baby 45–72%)  
- Eyes: 56–69% from bottom (baby 48–65%)  
- Show shoulders to head only  
- Neutral face, eyes open  
""")

uploaded = st.file_uploader("Upload Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    with st.spinner("Processing..."):
        clean = remove_bg(img)
        cropped, info = auto_crop(clean)
        guided, head_r, eye_r = draw_guidelines(cropped.copy(), info)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(img, use_column_width=True)
    with c2:
        st.subheader("Processed (600×600)")
        st.image(guided, use_column_width=True)
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=95)
        st.download_button("📥 Download DV Photo", buf.getvalue(), "dv_photo.jpg", "image/jpeg")

    st.info(f"""
**Analysis:**  
- Head size: {int(head_r*100)}% {'✅' if 0.45 <= head_r <= 0.72 else '❌'}  
- Eye position: {int(eye_r*100)}% {'✅' if 0.48 <= eye_r <= 0.69 else '❌'}  
- Baby detected: {'Yes 👶' if info.get('is_baby') else 'No'}  
""")
else:
    st.write("📸 Upload a photo to auto-crop to DV Lottery standard (shoulders to head only).")

st.caption("DV Lottery Photo Editor v2.1 — Compact AI Cropping & Compliance Check")
