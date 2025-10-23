import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings("ignore")

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="DV Auto-Crop Photo Tool", layout="wide")
st.title("🇺🇸 DV Lottery — Fully Auto Photo Crop & Fix (600x600)")

# ---------------- CONSTANTS ----------------
MIN_SIZE = 600  # final px
TARGET_HEAD_RATIO = 0.68  # 68% head height of image
TARGET_HEAD_PX = int(MIN_SIZE * TARGET_HEAD_RATIO)

# Eye position constraints (at 300 DPI)
DPI = 300
EYE_MIN_IN = 1.125
EYE_MAX_IN = 1.375
EYE_MIN_PX = int(EYE_MIN_IN * DPI)  # ~337
EYE_MAX_PX = int(EYE_MAX_IN * DPI)  # ~412
EYE_TARGET_PX = (EYE_MIN_PX + EYE_MAX_PX) // 2  # ~375

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

def get_face_bbox(pts, img_w, img_h, padding=0.45):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
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
    """Estimate top of head from Mediapipe landmarks."""
    try:
        forehead_y = pts[10][1]   # forehead
        chin_y = pts[152][1]      # chin
        face_h = chin_y - forehead_y
        top_y = int(forehead_y - 0.25 * face_h)  # add more margin for hair
        return max(top_y, 0)
    except Exception:
        ys = [p[1] for p in pts]
        return min(ys)

def remove_background_pil(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        out = Image.alpha_composite(white, fg).convert("RGB")
        return out
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
    chin_y = pts[152][1]
    top_y = estimate_head_top(pts)
    head_px = max(1, chin_y - top_y)

    left_eye_y = int((pts[159][1] + pts[145][1]) / 2)
    right_eye_y = int((pts[386][1] + pts[374][1]) / 2)
    eye_y = (left_eye_y + right_eye_y) // 2

    x1, y1, x2, y2 = get_face_bbox(pts, img_w, img_h, padding=0.45)
    face_crop = cv_img[y1:y2, x1:x2]

    top_y_crop = top_y - y1
    chin_y_crop = chin_y - y1
    head_px_crop = max(1, chin_y_crop - top_y_crop)
    scale_factor = TARGET_HEAD_PX / head_px_crop

    new_w = max(1, int(face_crop.shape[1] * scale_factor))
    new_h = max(1, int(face_crop.shape[0] * scale_factor))
    resized_face = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    eye_y_crop = eye_y - y1
    eye_y_resized = int(eye_y_crop * scale_factor)

    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, dtype=np.uint8)
    x_offset = (MIN_SIZE - new_w) // 2

    desired_eye_top_y = MIN_SIZE - EYE_TARGET_PX
    y_offset = desired_eye_top_y - eye_y_resized

    # --- Head & shoulder framing adjustment ---
    top_margin = int(0.08 * MIN_SIZE)
    bottom_margin = int(0.12 * MIN_SIZE)
    if y_offset < -top_margin:
        y_offset = -top_margin
    if y_offset + new_h > MIN_SIZE - bottom_margin:
        y_offset = MIN_SIZE - bottom_margin - new_h

    # Paste region calc
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
        canvas_fill = cv2.resize(resized_face, (min(new_w, MIN_SIZE), min(new_h, MIN_SIZE)), interpolation=cv2.INTER_LANCZOS4)
        ch, cw = canvas_fill.shape[:2]
        sx, sy = (MIN_SIZE - cw) // 2, (MIN_SIZE - ch) // 2
        canvas[sy:sy + ch, sx:sx + cw] = canvas_fill
        eye_y_resized = sy + ch // 2

    final_img_pil = Image.fromarray(canvas)

    # ---------------- Overlay Drawing ----------------
    overlay = final_img_pil.copy()
    draw = ImageDraw.Draw(overlay)

    top_on_canvas = dst_y1 + int(top_y_crop * scale_factor) - src_y1
    chin_on_canvas = dst_y1 + int(chin_y_crop * scale_factor) - src_y1
    eye_on_canvas = dst_y1 + eye_y_resized

    head_height_canvas_px = max(1, chin_on_canvas - top_on_canvas)
    head_ratio = head_height_canvas_px / MIN_SIZE

    eye_from_bottom = MIN_SIZE - eye_on_canvas

    # Lines
    draw.line([(0, top_on_canvas), (MIN_SIZE, top_on_canvas)], fill="blue", width=2)
    draw.line([(0, chin_on_canvas), (MIN_SIZE, chin_on_canvas)], fill="purple", width=2)
    draw.line([(0, eye_on_canvas), (MIN_SIZE, eye_on_canvas)], fill="green", width=2)

    # Eye band (dashed)
    eye_min_y_line = MIN_SIZE - EYE_MIN_PX
    eye_max_y_line = MIN_SIZE - EYE_MAX_PX
    for x in range(0, MIN_SIZE, 10):
        if x + 5 <= MIN_SIZE:
            draw.line([(x, eye_min_y_line), (x + 5, eye_min_y_line)], fill="green", width=2)
            draw.line([(x, eye_max_y_line), (x + 5, eye_max_y_line)], fill="green", width=2)

    # DV frame box
    draw.rectangle([(0, 0), (MIN_SIZE - 1, MIN_SIZE - 1)], outline="black", width=3)
    draw.line([(MIN_SIZE // 2, 0), (MIN_SIZE // 2, MIN_SIZE)], fill="gray", width=1)

    # Compliance check
    head_ok = 0.50 <= head_ratio <= 0.69
    eye_ok = EYE_MIN_PX <= eye_from_bottom <= EYE_MAX_PX
    passed = head_ok and eye_ok
    badge_color = "green" if passed else "red"
    status = "PASS ✅" if passed else "FAIL ❌"

    draw.rectangle([(10, 10), (170, 60)], fill="white", outline=badge_color, width=3)
    draw.text((20, 20), f"{status}", fill=badge_color)
    draw.text((20, 40), f"Head:{int(head_ratio*100)}% Eye:{eye_from_bottom}px", fill="black")

    return overlay, final_img_pil

# ---------------- STREAMLIT UI ----------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
Upload a front-facing portrait. This tool will:
- Auto-remove background (white)
- Auto-center and scale so head height meets DV specs
- Position eyes inside DV eye band
- Output 600×600 px JPEG with compliance check
""")

uploaded_file = st.file_uploader("Upload photo (jpg/png)", type=["jpg", "jpeg", "png"])
auto_run = st.sidebar.checkbox("Run auto-adjust immediately", value=True)
show_overlay = st.sidebar.checkbox("Show guidelines overlay on processed preview", value=True)

if uploaded_file:
    try:
        orig = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    with st.spinner("Removing background..."):
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
        st.download_button("Download 600x600 DV Photo", data=buf.getvalue(),
                           file_name="dv_photo_600x600.jpg", mime="image/jpeg")
else:
    st.markdown("## Welcome — DV Auto-Crop Tool\nUpload a photo to start.")
