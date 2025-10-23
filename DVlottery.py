import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings("ignore")

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="DV Auto-Crop Photo Tool", layout="wide")
st.title("🇺🇸 DV Lottery — Official Spec Auto Photo Crop & Check (600×600)")

# ---------------- CONSTANTS ----------------
MIN_SIZE = 600
TARGET_HEAD_RATIO = 0.63  # Moderate head size, keeps shoulders
TARGET_HEAD_PX = int(MIN_SIZE * TARGET_HEAD_RATIO)

DPI = 300
EYE_MIN_IN = 1.125
EYE_MAX_IN = 1.375
EYE_MIN_PX = int(EYE_MIN_IN * DPI)  # ~337
EYE_MAX_PX = int(EYE_MAX_IN * DPI)  # ~412
EYE_TARGET_PX = (EYE_MIN_PX + EYE_MAX_PX) // 2  # ~375 mid-point

mp_face_mesh = mp.solutions.face_mesh

# ---------------- HELPERS ----------------
def pil_to_cv(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr

def get_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0]

def landmarks_to_pixels(landmarks, img_w, img_h):
    return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks.landmark]

def get_face_bbox(pts, img_w, img_h, padding=0.5):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x_min, x_max = max(min(xs), 0), min(max(xs), img_w)
    y_min, y_max = max(min(ys), 0), min(max(ys), img_h)
    w, h = x_max - x_min, y_max - y_min
    pad_w, pad_h = int(w * padding), int(h * padding)
    x1 = max(x_min - pad_w, 0)
    y1 = max(y_min - pad_h, 0)
    x2 = min(x_max + pad_w, img_w)
    y2 = min(y_max + pad_h, img_h)
    return x1, y1, x2, y2

def estimate_head_top(pts):
    try:
        forehead_y = pts[10][1]
        chin_y = pts[152][1]
        face_h = chin_y - forehead_y
        return max(int(forehead_y - 0.25 * face_h), 0)
    except Exception:
        return min(p[1] for p in pts)

def remove_background_pil(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except Exception:
        return img_pil

# ---------------- CORE AUTO-CROP ----------------
def auto_crop_to_dv(img_pil):
    cv_img = pil_to_cv(img_pil)
    img_h, img_w = cv_img.shape[:2]
    landmarks = get_landmarks(cv_img)
    if landmarks is None:
        raise Exception("No face detected")

    pts = landmarks_to_pixels(landmarks, img_w, img_h)
    chin_y, top_y = pts[152][1], estimate_head_top(pts)

    left_eye_y = int((pts[159][1] + pts[145][1]) / 2)
    right_eye_y = int((pts[386][1] + pts[374][1]) / 2)
    eye_y = (left_eye_y + right_eye_y) // 2

    x1, y1, x2, y2 = get_face_bbox(pts, img_w, img_h, padding=0.5)
    face_crop = cv_img[y1:y2, x1:x2]

    top_y_crop, chin_y_crop = top_y - y1, chin_y - y1
    head_px_crop = max(1, chin_y_crop - top_y_crop)
    scale_factor = (TARGET_HEAD_PX / head_px_crop) * 0.94  # small zoom-out

    new_w, new_h = int(face_crop.shape[1] * scale_factor), int(face_crop.shape[0] * scale_factor)
    resized_face = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    eye_y_resized = int((eye_y - y1) * scale_factor)
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, dtype=np.uint8)
    x_offset = (MIN_SIZE - new_w) // 2
    desired_eye_top_y = MIN_SIZE - EYE_TARGET_PX
    y_offset = desired_eye_top_y - eye_y_resized

    top_margin, bottom_margin = int(0.10 * MIN_SIZE), int(0.18 * MIN_SIZE)
    if y_offset < -top_margin:
        y_offset = -top_margin
    if y_offset + new_h > MIN_SIZE - bottom_margin:
        y_offset = MIN_SIZE - bottom_margin - new_h

    paste_x1, paste_y1 = x_offset, y_offset
    paste_x2, paste_y2 = paste_x1 + new_w, paste_y1 + new_h
    src_x1 = src_y1 = 0
    src_x2, src_y2 = new_w, new_h
    dst_x1, dst_y1, dst_x2, dst_y2 = paste_x1, paste_y1, paste_x2, paste_y2

    if dst_x1 < 0:
        src_x1 = -dst_x1
        dst_x1 = 0
    if dst_y1 < 0:
        src_y1 = -dst_y1
        dst_y1 = 0
    if dst_x2 > MIN_SIZE:
        src_x2 = new_w - (dst_x2 - MIN_SIZE)
        dst_x2 = MIN_SIZE
    if dst_y2 > MIN_SIZE:
        src_y2 = new_h - (dst_y2 - MIN_SIZE)
        dst_y2 = MIN_SIZE

    try:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_face[src_y1:src_y2, src_x1:src_x2]
    except Exception:
        canvas = cv2.resize(resized_face, (MIN_SIZE, MIN_SIZE))

    final_img_pil = Image.fromarray(canvas)

    # ---------------- OVERLAY ----------------
    overlay = final_img_pil.copy()
    draw = ImageDraw.Draw(overlay)

    top_on_canvas = dst_y1 + int(top_y_crop * scale_factor) - src_y1
    chin_on_canvas = dst_y1 + int(chin_y_crop * scale_factor) - src_y1
    eye_on_canvas = dst_y1 + eye_y_resized
    eye_from_bottom = MIN_SIZE - eye_on_canvas
    head_height = chin_on_canvas - top_on_canvas
    head_ratio = head_height / MIN_SIZE

    # DV guideline red lines
    draw.line([(0, top_on_canvas), (MIN_SIZE, top_on_canvas)], fill="red", width=3)
    draw.line([(0, eye_on_canvas), (MIN_SIZE, eye_on_canvas)], fill="red", width=3)
    draw.line([(0, chin_on_canvas), (MIN_SIZE, chin_on_canvas)], fill="red", width=3)

    # Eye band region (green)
    eye_min_y, eye_max_y = MIN_SIZE - EYE_MIN_PX, MIN_SIZE - EYE_MAX_PX
    draw.line([(0, eye_min_y), (MIN_SIZE, eye_min_y)], fill="green", width=2)
    draw.line([(0, eye_max_y), (MIN_SIZE, eye_max_y)], fill="green", width=2)

    # 2x2 inch box outline
    draw.rectangle([(0, 0), (MIN_SIZE - 1, MIN_SIZE - 1)], outline="black", width=3)

    # Center vertical line
    draw.line([(MIN_SIZE // 2, 0), (MIN_SIZE // 2, MIN_SIZE)], fill="gray", width=1)

    # PASS/FAIL badge
    head_ok = 0.50 <= head_ratio <= 0.69
    eye_ok = EYE_MIN_PX <= eye_from_bottom <= EYE_MAX_PX
    passed = head_ok and eye_ok
    badge_color = "green" if passed else "red"
    status = "PASS ✅" if passed else "FAIL ❌"

    draw.rectangle([(10, 10), (180, 65)], fill="white", outline=badge_color, width=3)
    draw.text((20, 20), f"{status}", fill=badge_color)
    draw.text((20, 45), f"H:{int(head_ratio*100)}%  E:{eye_from_bottom}px", fill="black")

    return overlay, final_img_pil

# ---------------- STREAMLIT UI ----------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
Upload a clear, front-facing photo.
This tool will:
- Auto-remove background (white)
- Scale and center the head per DV specs
- Draw guideline lines exactly like the official reference
- Show a PASS/FAIL badge with reasons
""")

uploaded_file = st.file_uploader("Upload photo (jpg/png)", type=["jpg", "jpeg", "png"])
show_overlay = st.sidebar.checkbox("Show DV guideline overlay", value=True)

if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Processing photo..."):
        bg_removed = remove_background_pil(orig)
        try:
            overlay_img, final_img = auto_crop_to_dv(bg_removed)
        except Exception as e:
            st.error(f"Auto processing failed: {e}")
            final_img = orig.copy().resize((MIN_SIZE, MIN_SIZE), Image.LANCZOS)
            overlay_img = final_img.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(np.array(orig), width=300)
    with col2:
        st.subheader("Processed (600×600)")
        st.image(np.array(overlay_img if show_overlay else final_img), width=300)

        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button("Download 600×600 DV Photo",
                           data=buf.getvalue(),
                           file_name="dv_photo_600x600.jpg",
                           mime="image/jpeg")
else:
    st.markdown("## Welcome — Upload a photo to begin.")
