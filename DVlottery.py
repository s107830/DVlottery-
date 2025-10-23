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
st.title("DV Lottery Photo Editor — Improved Hair Edge & Human Segmentation")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69
DPI = 300  # 2×2 inch at 300 DPI
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

# ---------------------- IMPROVED BACKGROUND REMOVAL ----------------------
def remove_background(img_pil, edge_trim=3, alpha_thresh=200):
    """
    Remove background using human-segmentation model + cleanup:
    - Use hard mask (binary) to eliminate translucent halo.
    - Optionally trim a few pixels (edge_trim) to tighten hairline.
    """
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")

        np_fg = np.array(fg)
        rgb = np_fg[:, :, :3]
        alpha = np_fg[:, :, 3]

        # 1) Binary mask
        mask = np.where(alpha > alpha_thresh, 255, 0).astype(np.uint8)

        # 2) Trim the edge (erode)
        if edge_trim > 0:
            kernel = np.ones((edge_trim, edge_trim), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # 3) Clean mask
        kernel2 = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)

        # 4) Composite on white
        white = np.full_like(rgb, 255, np.uint8)
        composite = np.where(mask[:, :, None] == 255, rgb, white)

        return Image.fromarray(composite.astype(np.uint8))

    except Exception as e:
        st.warning(f"Background cleanup failed ({e}). Using original image.")
        return img_pil

# ---------------------- AUTO CROP ----------------------
def auto_crop_dv(img_pil):
    cv_img = np.array(img_pil)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    h, w = cv_img.shape[:2]
    landmarks = get_face_landmarks(cv_img)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
    head_h = chin_y - top_y

    target_head = MIN_SIZE * 0.63
    scale = target_head / head_h
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    target_eye_min = MIN_SIZE - int(EYE_MAX_RATIO * MIN_SIZE)
    target_eye_max = MIN_SIZE - int(EYE_MIN_RATIO * MIN_SIZE)
    target_eye = (target_eye_min + target_eye_max) // 2

    landmarks_resized = get_face_landmarks(resized)
    top_y_r, chin_y_r, eye_y_r = get_head_eye_positions(landmarks_resized, resized.shape[0], resized.shape[1])
    y_offset = target_eye - eye_y_r
    x_offset = (MIN_SIZE - new_w) // 2

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

    final_top_y = top_y_r + y_offset
    final_chin_y = chin_y_r + y_offset
    final_eye_y = eye_y_r + y_offset

    head_info = {
        "top_y": final_top_y,
        "chin_y": final_chin_y,
        "eye_y": final_eye_y,
        "head_height": chin_y_r - top_y_r,
        "canvas_size": MIN_SIZE
    }
    return Image.fromarray(canvas), head_info

# ---------------------- DRAW GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w // 2
    top_y = head_info["top_y"]
    chin_y = head_info["chin_y"]
    eye_y = head_info["eye_y"]
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

    inch_px = DPI
    for i in range(3):
        y = i * inch_px
        draw.line([(0, y), (20, y)], fill="black", width=2)
        draw.text((25, y - 10), f"{i} in", fill="black")
        draw.line([(w - 20, y), (w, y)], fill="black", width=2)
        draw.text((w - 55, y - 10), f"{i} in", fill="black")

    passed = (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO) and (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO)
    badge_color = "green" if passed else "red"
    status_text = "PASS" if passed else "FAIL"
    draw.rectangle([(10, 10), (170, 60)], fill="white", outline=badge_color, width=3)
    draw.text((25, 20), status_text, fill=badge_color)
    draw.text((25, 40), f"H:{int(head_ratio*100)}%  E:{int(eye_ratio*100)}%", fill="black")

    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a clear front-facing photo.
2. Background is auto-removed & cleaned (human model).
3. Cropped & scaled to 2×2 inch (600×600 px).
4. Official DV guidelines drawn (lines, rulers, PASS badge).

DV Requirements:
- Head height: 50–69% of image
- Eyes: 1-1/8 to 1-3/8 inch from bottom
- Plain white background
- Neutral expression, both eyes open
- No glasses/hats/shadows
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
        st.subheader("Processed (600×600)")
        st.image(overlay, use_column_width=True)

        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Final 600×600 Photo",
            data=buf.getvalue(),
            file_name="dv_photo_final.jpg",
            mime="image/jpeg"
        )
else:
    st.markdown("""
## Welcome to the DV Lottery Photo Editor  
Upload your photo to get a compliant 600×600 image with clean hair edge and official guideline lines.
""")

st.markdown("---")
st.caption("DV Lottery Photo Editor | Human-Segmentation Clean-Edge Edition")
