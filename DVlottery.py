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
TARGET_HEAD_RATIO = 0.68  # aim for 68% head height of image (within 50-69% allowed)
TARGET_HEAD_PX = int(MIN_SIZE * TARGET_HEAD_RATIO)

# Eye position constraints in inches (from bottom):
# DV requires eyes between 1 1/8" and 1 3/8" from bottom. At 300 DPI -> multiply by 300
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
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks.landmark]
    return pts

def get_face_bbox(pts, img_w, img_h, padding=0.25):
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
    # Use landmark 10 (forehead-ish) and shift upward a bit relative to face height to approximate top of head
    # Mediapipe indices: 10 - forehead, 152 - chin
    # If landmarks provided as list of tuples:
    try:
        forehead_y = pts[10][1]
        chin_y = pts[152][1]
        face_h = chin_y - forehead_y
        top_y = int(forehead_y - 0.15 * face_h)  # move a bit upward
        return max(top_y, 0)
    except Exception:
        # fallback to min y of landmarks
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
    """
    Returns (processed_image_with_overlay_pil, final_image_pil(no overlay, exactly 600x600))
    """
    cv_img = pil_to_cv(img_pil)
    img_h, img_w = cv_img.shape[:2]

    landmarks = get_landmarks(cv_img)
    if landmarks is None:
        raise Exception("No face detected")

    pts = landmarks_to_pixels(landmarks, img_w, img_h)

    # key positions
    chin_y = pts[152][1]
    # estimate top of head
    top_y = estimate_head_top(pts)
    head_px = max(1, chin_y - top_y)

    # eye row: average of left/right eye centers (use landmarks 33 and 263 as approximate outer eye corners,
    # and 159/374 or 145/374 for inner/upper points; we'll take average of upper eyelid landmarks to get eye y)
    left_eye_y = int((pts[159][1] + pts[145][1]) / 2)
    right_eye_y = int((pts[386][1] + pts[374][1]) / 2)
    eye_y = (left_eye_y + right_eye_y) // 2

    # compute bounding box to crop (face region)
    x1, y1, x2, y2 = get_face_bbox(pts, img_w, img_h, padding=0.30)
    face_crop = cv_img[y1:y2, x1:x2]

    # compute scale to make head height == TARGET_HEAD_PX after resizing
    # current head in crop coordinates:
    # adjust top_y and chin_y to crop coords
    top_y_crop = top_y - y1
    chin_y_crop = chin_y - y1
    head_px_crop = max(1, chin_y_crop - top_y_crop)
    scale_factor = TARGET_HEAD_PX / head_px_crop

    # resize the face_crop by scale_factor
    new_w = max(1, int(face_crop.shape[1] * scale_factor))
    new_h = max(1, int(face_crop.shape[0] * scale_factor))
    resized_face = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # compute eye_y in resized coordinates
    eye_y_crop = eye_y - y1
    eye_y_resized = int(eye_y_crop * scale_factor)

    # create white canvas 600x600, and compute offsets to paste resized_face so that:
    # 1) center horizontally
    # 2) vertical offset chosen to place eyes at TARGET eye-from-bottom (EYE_TARGET_PX)
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, dtype=np.uint8)

    x_offset = (MIN_SIZE - new_w) // 2

    # desired eye_y on canvas (pixel from top) => canvas_eye_y = MIN_SIZE - EYE_TARGET_PX
    desired_eye_top_y = MIN_SIZE - EYE_TARGET_PX

    # naive y_offset to place resized eye onto desired_eye_top_y
    y_offset = desired_eye_top_y - eye_y_resized

    # Ensure the resized image fully fits onto the canvas; adjust if needed
    if y_offset > 0 and y_offset + new_h > MIN_SIZE:
        # crops bottom overflow
        y_offset = MIN_SIZE - new_h
    if y_offset < 0 and -y_offset > top_y_crop * scale_factor + 1000:
        # ensure not shifting too far (fallback)
        y_offset = max(y_offset, -int(0.2 * new_h))

    # If resized face is larger than canvas, we will center instead (and allow cropping)
    paste_x1 = x_offset
    paste_y1 = y_offset
    paste_x2 = paste_x1 + new_w
    paste_y2 = paste_y1 + new_h

    # If face bigger than canvas, compute overlap region
    src_x1 = 0
    src_y1 = 0
    src_x2 = new_w
    src_y2 = new_h

    dst_x1 = paste_x1
    dst_y1 = paste_y1
    dst_x2 = paste_x2
    dst_y2 = paste_y2

    # adjust if dst starts negative (image above/left)
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

    # Only paste the overlapping region
    try:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_face[src_y1:src_y2, src_x1:src_x2]
    except Exception:
        # fallback: center the resized image (clip)
        canvas_fill = cv2.resize(resized_face, (min(new_w, MIN_SIZE), min(new_h, MIN_SIZE)), interpolation=cv2.INTER_LANCZOS4)
        ch, cw = canvas_fill.shape[:2]
        sx = (MIN_SIZE - cw) // 2
        sy = (MIN_SIZE - ch) // 2
        canvas[sy:sy + ch, sx:sx + cw] = canvas_fill
        # recompute final eye pos roughly
        eye_y_resized = sy + ch // 2

    final_img_pil = Image.fromarray(canvas)

    # Prepare overlay for visualization (not in final downloadable image)
    overlay = final_img_pil.copy()
    draw = ImageDraw.Draw(overlay)

    # draw head top / chin based on our placement:
    # top on canvas is at: top_y_canvas = dst_y1 + top_y_crop * scale_factor - src_y1
    top_on_canvas = dst_y1 + int(top_y_crop * scale_factor) - src_y1
    chin_on_canvas = dst_y1 + int(chin_y_crop * scale_factor) - src_y1
    eye_on_canvas = dst_y1 + eye_y_resized

    # head percentage
    head_height_canvas_px = max(1, chin_on_canvas - top_on_canvas)
    head_ratio = head_height_canvas_px / MIN_SIZE

    # draw lines
    draw.line([(0, top_on_canvas), (MIN_SIZE, top_on_canvas)], fill="blue", width=2)
    draw.text((8, max(0, top_on_canvas - 18)), f"Head Top ({top_on_canvas}px)", fill="blue")
    draw.line([(0, chin_on_canvas), (MIN_SIZE, chin_on_canvas)], fill="purple", width=2)
    draw.text((8, max(0, chin_on_canvas - 18)), f"Chin ({chin_on_canvas}px)", fill="purple")

    # eye line and allowed band
    eye_line_color = "green" if (EYE_MIN_PX <= (MIN_SIZE - eye_on_canvas) <= EYE_MAX_PX) else "red"
    draw.line([(0, eye_on_canvas), (MIN_SIZE, eye_on_canvas)], fill=eye_line_color, width=2)
    # draw band for min/max eyes (from bottom)
    eye_min_y_line = MIN_SIZE - EYE_MIN_PX
    eye_max_y_line = MIN_SIZE - EYE_MAX_PX
    # draw dashed band
    for x in range(0, MIN_SIZE, 10):
        if x + 5 <= MIN_SIZE:
            draw.line([(x, eye_min_y_line), (x + 5, eye_min_y_line)], fill="green", width=2)
            draw.line([(x, eye_max_y_line), (x + 5, eye_max_y_line)], fill="green", width=2)

    draw.text((8, MIN_SIZE - EYE_MIN_PX - 18), f"Eye band top (min) {EYE_MIN_PX}px from bottom", fill="green")
    draw.text((8, MIN_SIZE - EYE_MAX_PX - 18), f"Eye band bottom (max) {EYE_MAX_PX}px from bottom", fill="green")

    # head percent text
    head_color = "green" if 0.50 <= head_ratio <= 0.69 else "red"
    draw.text((MIN_SIZE - 180, 8), f"Head: {int(head_ratio*100)}% -> target {int(TARGET_HEAD_RATIO*100)}%", fill=head_color)

    return overlay, final_img_pil

# ---------------- STREAMLIT UI ----------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
Upload a front-facing portrait. This tool will:
- Auto-remove background (white)
- Auto-center and scale so head height meets DV specs
- Position eyes inside DV eye band
- Output 600 x 600 px JPEG
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

    # remove background (white)
    with st.spinner("Removing background..."):
        bg_removed = remove_background_pil(orig)

    # try to auto crop/center
    try:
        overlay_img, final_img = auto_crop_to_dv(bg_removed)
    except Exception as e:
        st.error(f"Auto processing failed: {e}")
        # fallback: simple center-resize to 600
        final_img = orig.copy().resize((MIN_SIZE, MIN_SIZE), Image.LANCZOS)
        overlay_img = final_img.copy()

    # show side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        try:
            st.image(np.array(orig), width=300)
        except Exception as e:
            st.error(f"Failed to display original image: {e}")

    with col2:
        st.subheader("Processed (600x600)")
        if show_overlay:
            try:
                st.image(np.array(overlay_img), width=300)
            except Exception as e:
                st.error(f"Failed to display processed preview: {e}")
        else:
            try:
                st.image(np.array(final_img), width=300)
            except Exception as e:
                st.error(f"Failed to display processed preview: {e}")

        # provide download (final image - NO overlay)
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button("Download 600x600 DV Photo", data=buf.getvalue(),
                           file_name="dv_photo_600x600.jpg", mime="image/jpeg")

else:
    st.markdown("## Welcome — DV Auto-Crop Tool\nUpload a photo to start.")

# ---------------- END ----------------
