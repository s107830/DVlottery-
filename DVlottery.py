import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2, io, warnings
import mediapipe as mp
from rembg import remove
warnings.filterwarnings('ignore')

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="DV Lottery Auto Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — AI Auto Adjust v2.6")

MIN_SIZE = 600
mp_face_mesh = mp.solutions.face_mesh
HEAD_STD = (0.50, 0.69)
EYE_STD = (0.56, 0.69)
HEAD_BABY = (0.45, 0.72)
EYE_BABY = (0.48, 0.65)

# ---------------------- FACE UTILITIES ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3) as fm:
        res = fm.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return res.multi_face_landmarks[0]

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

def remove_bg(img):
    try:
        b = io.BytesIO(); img.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img

# ---------------------- AUTO CROP ----------------------
def auto_crop(img_pil, scale_boost=1.0, y_shift=0):
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

    scale = (MIN_SIZE * 0.85 * scale_boost) / (shoulder_y - top_y)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_img, (new_w, new_h))
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)

    y_center = (top_y + shoulder_y) // 2
    y_offset = MIN_SIZE//2 - y_center + int(y_shift)
    x_offset = (MIN_SIZE - new_w)//2
    y_offset = max(min(y_offset, MIN_SIZE - new_h), 0)
    x_offset = max(min(x_offset, MIN_SIZE - new_w), 0)
    y_end = min(MIN_SIZE, y_offset + new_h)
    x_end = min(MIN_SIZE, x_offset + new_w)
    canvas[y_offset:y_end, x_offset:x_end] = resized[:y_end - y_offset, :x_end - x_offset]

    info = {"top_y": top_y + y_offset, "chin_y": chin_y + y_offset,
            "eye_y": eye_y + y_offset, "shoulder_y": shoulder_y + y_offset,
            "head_height": chin_y - top_y, "is_baby": is_baby}
    return Image.fromarray(canvas), info

# ---------------------- DRAW LINES ----------------------
def draw_guidelines(img, info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    t, c, e = info["top_y"], info["chin_y"], info["eye_y"]
    head_h = info["head_height"]
    head_r, eye_r = head_h / h, (h - e) / h
    if info["is_baby"]:
        hmin, hmax, emin, emax = HEAD_BABY + EYE_BABY
    else:
        hmin, hmax, emin, emax = HEAD_STD + EYE_STD
    ok = hmin <= head_r <= hmax and emin <= eye_r <= emax
    color = "green" if ok else "red"
    draw.line([(0, t), (w, t)], fill="green", width=2)
    draw.line([(0, e), (w, e)], fill="orange", width=2)
    draw.line([(0, c), (w, c)], fill="red", width=2)
    draw.rectangle([(10, 10), (220, 85)], outline=color, fill="white", width=3)
    draw.text((20, 15), "PASS" if ok else "FAIL", fill=color)
    draw.text((20, 35), f"Head: {int(head_r*100)}%", fill="black")
    draw.text((20, 50), f"Eyes: {int(eye_r*100)}%", fill="black")
    if info["is_baby"]: draw.text((20, 65), "BABY MODE", fill="orange")
    return img, head_r, eye_r, ok

# ---------------------- AUTO ADJUST LOOP ----------------------
def auto_adjust(img, info, head_r, eye_r):
    target_head, target_eye = 0.60, 0.60
    best_img, best_info = img, info
    for _ in range(3):  # iterative fine tuning
        scale_adj = 1 + (target_head - head_r) * 0.5  # slower scaling
        y_shift = (target_eye - eye_r) * MIN_SIZE * 0.4
        adjusted, new_info = auto_crop(img, scale_boost=scale_adj, y_shift=y_shift)
        _, head_r, eye_r, ok = draw_guidelines(adjusted.copy(), new_info)
        best_img, best_info = adjusted, new_info
        if ok: break
    return best_img, best_info

# ---------------------- STREAMLIT ----------------------
st.sidebar.header("DV Lottery Rules")
st.sidebar.markdown("""
- 600×600px white background  
- Head: 50–69% (baby 45–72%)  
- Eyes: 56–69% (baby 48–65%)  
""")

uploaded = st.file_uploader("Upload Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    with st.spinner("Processing..."):
        clean = remove_bg(img)
        cropped, info = auto_crop(clean)
        guided, head_r, eye_r, ok = draw_guidelines(cropped.copy(), info)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(img, use_column_width=True)
    with c2:
        st.subheader("Processed (600×600)")
        st.image(guided, use_column_width=True)

        if not ok:
            if st.button("🧠 Auto Adjust"):
                fixed, fixed_info = auto_adjust(clean, info, head_r, eye_r)
                fixed_guided, _, _, ok2 = draw_guidelines(fixed.copy(), fixed_info)
                st.image(fixed_guided, use_column_width=True, caption="Auto Adjusted ✅" if ok2 else "Adjusted (retry if needed)")
                buf2 = io.BytesIO()
                fixed.save(buf2, format="JPEG", quality=95)
                st.download_button("📥 Download Adjusted Photo", buf2.getvalue(), "dv_auto_fixed.jpg", "image/jpeg")
        else:
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=95)
            st.download_button("📥 Download DV Photo", buf.getvalue(), "dv_photo.jpg", "image/jpeg")

    st.info(f"""
**Analysis:**  
- Head size: {int(head_r*100)}% {'✅' if ok else '❌'}  
- Eye position: {int(eye_r*100)}% {'✅' if ok else '❌'}  
- Baby detected: {'Yes 👶' if info.get('is_baby') else 'No'}  
""")
else:
    st.write("📸 Upload a photo to auto-crop and auto-fix to DV standard.")
st.caption("DV Lottery Photo Editor v2.6 — Self-Correcting Auto Adjust Mode")
