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

# ---------------------- FACE DETECTION ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return get_face_bounding_box(cv_img)
        return results.multi_face_landmarks[0]

def get_face_bounding_box(cv_img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        if not results.detections:
            raise Exception("No face detected.")
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = cv_img.shape[:2]

        class MockLandmarks:
            def __init__(self): self.landmark = [None]*478
        landmarks = MockLandmarks()
        x, y, width, height = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
        landmarks.landmark[10]  = type('obj', (object,), {'y': max(0,(y-height*0.2))/h})()
        landmarks.landmark[152] = type('obj', (object,), {'y': (y+height*1.1)/h})()
        eye_y = y + height*0.3
        landmarks.landmark[33] = type('obj',(object,),{'y':eye_y/h})()
        landmarks.landmark[263] = type('obj',(object,),{'y':eye_y/h})()
        return landmarks

def get_head_eye_positions(landmarks, img_h, img_w):
    try:
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        left_eye_y = int(landmarks.landmark[33].y * img_h)
        right_eye_y = int(landmarks.landmark[263].y * img_h)
        eye_y = (left_eye_y + right_eye_y)//2
        top_y = max(0, top_y - int((chin_y-top_y)*0.25))
        return top_y, chin_y, eye_y
    except:
        return int(img_h*0.25), int(img_h*0.75), int(img_h*0.5)

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
    try:
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        result = remove(img_bytes)
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        white_bg = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white_bg, fg).convert("RGB")
    except:
        return img_pil

# ---------------------- AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil):
    image_rgb = np.array(image_pil)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    h, w = image_rgb.shape[:2]
    landmarks = get_face_landmarks(image_rgb)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
    head_h = chin_y - top_y

    target_head_h = (HEAD_MIN_RATIO + HEAD_MAX_RATIO) / 2 * MIN_SIZE
    scale = target_head_h / head_h
    scale = np.clip(scale, 0.4, 2.5)

    new_h, new_w = int(h*scale), int(w*scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    target_eye_y = MIN_SIZE - int(MIN_SIZE * ((EYE_MIN_RATIO + EYE_MAX_RATIO)/2))
    y_off = target_eye_y - int(eye_y*scale)
    y_off = np.clip(y_off, -int(top_y*scale), MIN_SIZE - int(chin_y*scale))
    x_off = (MIN_SIZE - new_w)//2

    y_start, y_end = max(0, y_off), min(MIN_SIZE, y_off+new_h)
    x_start, x_end = max(0, x_off), min(MIN_SIZE, x_off+new_w)
    y_src_s, y_src_e = max(0,-y_off), min(new_h, MIN_SIZE-y_off)
    x_src_s, x_src_e = max(0,-x_off), min(new_w, MIN_SIZE-x_off)

    canvas[y_start:y_end, x_start:x_end] = resized[y_src_s:y_src_e, x_src_s:x_src_e]

    result = Image.fromarray(canvas)
    result = ImageEnhance.Sharpness(result).enhance(1.1)
    head_info = {'top_y':y_start+int(top_y*scale),'chin_y':y_start+int(chin_y*scale),
                 'eye_y':y_start+int(eye_y*scale),'head_height':int((chin_y-top_y)*scale),'canvas_size':MIN_SIZE}
    return result, head_info

# ---------------------- DRAW GUIDELINES ----------------------
def draw_guidelines(img, info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    top_y, chin_y, eye_y = info['top_y'], info['chin_y'], info['eye_y']
    head_h, canvas = info['head_height'], info['canvas_size']

    actual_head_ratio = head_h / canvas
    actual_eye_ratio = (h - eye_y) / h
    head_color = "green" if HEAD_MIN_RATIO<=actual_head_ratio<=HEAD_MAX_RATIO else "red"
    eye_color  = "green" if EYE_MIN_RATIO<=actual_eye_ratio<=EYE_MAX_RATIO else "red"

    draw.line([(0, top_y), (w, top_y)], fill="blue", width=2)
    draw.line([(0, chin_y), (w, chin_y)], fill="blue", width=2)
    draw.line([(0, eye_y), (w, eye_y)], fill="green", width=2)
    draw.text((10, 10), f"Head: {actual_head_ratio*100:.1f}% ({head_color})", fill=head_color)
    draw.text((10, 30), f"Eye: {actual_eye_ratio*100:.1f}% ({eye_color})", fill=eye_color)
    return img, actual_head_ratio, actual_eye_ratio

# ---------------------- STREAMLIT UI ----------------------
uploaded = st.file_uploader("📤 Upload photo (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Photo", use_container_width=True)

    bg_removed = remove_background(img)
    processed, head_info = auto_adjust_dv_photo(bg_removed)
    preview, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)

    with col2:
        st.image(preview, caption="DV Compliance Preview", use_container_width=True)

    # status
    st.markdown("### 🧾 Head & Eye Measurement")
    st.write(f"**Head Height:** {head_ratio*100:.1f}% (required 50–69%)")
    st.write(f"**Eye Position:** {eye_ratio*100:.1f}% from bottom (required 56–69%)")

    # Fix button
    if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO and EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO):
        if st.button("🔧 Fix & Re-Adjust"):
            fixed, fixed_info = auto_adjust_dv_photo(processed)
            fixed_preview, new_head, new_eye = draw_guidelines(fixed.copy(), fixed_info)
            st.image(fixed_preview, caption="✅ Fixed & Re-Aligned", use_container_width=True)
            st.success(f"Head: {new_head*100:.1f}% | Eye: {new_eye*100:.1f}%")

            buf = io.BytesIO()
            fixed.save(buf, format="JPEG", quality=98)
            buf.seek(0)
            st.download_button("⬇️ Download Fixed Photo", buf, "dv_fixed.jpg", "image/jpeg")
    else:
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=98)
        buf.seek(0)
        st.download_button("⬇️ Download DV Photo", buf, "dv_ready.jpg", "image/jpeg")

else:
    st.info("👆 Upload a photo to start the automatic DV compliance process.")
